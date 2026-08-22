"""
Evaluation Metrics - Built From Scratch
==========================================
Computes standard object detection metrics:
  - IoU (Intersection over Union) between predicted and ground truth boxes
  - Precision, Recall at a given IoU threshold
  - mAP50 (mean Average Precision at IoU threshold 0.5)
  - mAP50-95 (averaged over IoU thresholds 0.5 to 0.95, step 0.05)
      This is the metric that directly reflects the localization
      quality problem identified earlier - a model can have high
      mAP50 (finds objects) but low mAP50-95 (imprecise boxes).

These metrics only make sense on VALIDATION/TEST data (never on
training data - that would just measure memorization).
"""

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes IoU between every box in boxes1 and every box in boxes2.
    Boxes are in (xc, yc, w, h) normalized format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        iou_matrix: (N, M) - iou_matrix[i, j] = IoU between boxes1[i] and boxes2[j]
    """
    def to_corners(boxes):
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    b1 = to_corners(boxes1)  # (N, 4)
    b2 = to_corners(boxes2)  # (M, 4)

    N, M = b1.shape[0], b2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M))

    # Broadcast to compute all pairs at once
    b1 = b1.unsqueeze(1)  # (N, 1, 4)
    b2 = b2.unsqueeze(0)  # (1, M, 4)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)

    union_area = area1 + area2 - inter_area + 1e-7
    return inter_area / union_area  # (N, M)


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """
    Greedy matching: sort predictions by confidence (highest first),
    match each to the best available (unused) ground-truth box if
    IoU >= threshold. This is the standard approach used by COCO/YOLO
    evaluation - one prediction can only match one GT box, and
    duplicate detections of the same object count as false positives.

    Returns:
        tp: (N,) tensor of 1s and 0s - was this prediction a true positive?
        matched_gt_count: number of unique GT boxes that were matched
    """
    num_preds = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_preds == 0:
        return torch.zeros(0), 0
    if num_gt == 0:
        return torch.zeros(num_preds), 0

    # Sort predictions by confidence, descending
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[order]

    iou_matrix = box_iou(pred_boxes_sorted, gt_boxes)  # (num_preds, num_gt)

    gt_used = torch.zeros(num_gt, dtype=torch.bool)
    tp = torch.zeros(num_preds)

    for i in range(num_preds):
        ious = iou_matrix[i]
        ious[gt_used] = 0  # can't match an already-used GT box
        best_iou, best_gt_idx = ious.max(dim=0)

        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_used[best_gt_idx] = True

    # Re-order tp back to original prediction order
    tp_original_order = torch.zeros(num_preds)
    tp_original_order[order] = tp

    return tp_original_order, gt_used.sum().item()


def compute_precision_recall(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold):
    """
    Computes overall precision and recall across an entire dataset
    (list of per-image predictions and ground truths) at one IoU threshold.
    """
    all_tp = []
    all_scores = []
    total_gt = 0

    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        tp, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        all_tp.append(tp)
        all_scores.append(pred_scores)
        total_gt += len(gt_boxes)

    if len(all_tp) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0  # precision, recall, AP

    all_tp = torch.cat(all_tp)
    all_scores = torch.cat(all_scores)

    # Sort all predictions across the whole dataset by confidence
    order = torch.argsort(all_scores, descending=True)
    tp_sorted = all_tp[order]
    fp_sorted = 1 - tp_sorted

    tp_cumsum = torch.cumsum(tp_sorted, dim=0)
    fp_cumsum = torch.cumsum(fp_sorted, dim=0)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    recalls = tp_cumsum / (total_gt + 1e-7)

    # Average Precision = area under the precision-recall curve
    # (using the standard 101-point interpolation, simplified here to trapezoidal)
    ap = torch.trapz(precisions, recalls).item()

    final_precision = precisions[-1].item() if len(precisions) > 0 else 0.0
    final_recall = recalls[-1].item() if len(recalls) > 0 else 0.0

    return final_precision, final_recall, ap


def compute_map(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresholds=None):
    """
    Computes mAP averaged over a range of IoU thresholds.

    Args:
        all_pred_boxes: list (len=num_images) of (N_i, 4) predicted boxes
        all_pred_scores: list (len=num_images) of (N_i,) confidence scores
        all_gt_boxes: list (len=num_images) of (M_i, 4) ground truth boxes
        iou_thresholds: list of IoU thresholds to average over.
                        Default: [0.5] for mAP50, or 0.5:0.05:0.95 for mAP50-95

    Returns:
        dict with "mAP50", "mAP50_95", "precision", "recall" (all at IoU=0.5)
    """
    # mAP50 - single threshold
    precision_50, recall_50, ap_50 = compute_precision_recall(
        all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5
    )

    # mAP50-95 - averaged over 10 thresholds
    thresholds_50_95 = [0.5 + 0.05 * i for i in range(10)]  # 0.50, 0.55, ..., 0.95
    aps = []
    for thresh in thresholds_50_95:
        _, _, ap = compute_precision_recall(
            all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=thresh
        )
        aps.append(ap)
    map_50_95 = sum(aps) / len(aps)

    return {
        "mAP50": ap_50,
        "mAP50_95": map_50_95,
        "precision": precision_50,
        "recall": recall_50,
    }


if __name__ == "__main__":
    # Sanity check with synthetic data
    print("Running evaluation metrics sanity checks...\n")

    # Test 1: Perfect prediction (IoU matrix should show high overlap)
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    iou = box_iou(pred, gt)
    print(f"Test 1 - Perfect overlap IoU (should be ~1.0): {iou.item():.4f}")
    assert iou.item() > 0.99

    # Test 2: No overlap
    pred2 = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    gt2 = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    iou2 = box_iou(pred2, gt2)
    print(f"Test 2 - No overlap IoU (should be 0.0): {iou2.item():.4f}")
    assert iou2.item() == 0.0

    # Test 3: Full pipeline with a small synthetic dataset
    # Image 1: 1 GT box, 1 correct prediction
    # Image 2: 1 GT box, prediction is a bit off but still IoU > 0.5
    # Image 3: 1 GT box, no prediction (miss)
    all_pred_boxes = [
        torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.52, 0.51, 0.19, 0.21]]),
        torch.zeros((0, 4)),
    ]
    all_pred_scores = [
        torch.tensor([0.9]),
        torch.tensor([0.8]),
        torch.zeros(0),
    ]
    all_gt_boxes = [
        torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
    ]

    results = compute_map(all_pred_boxes, all_pred_scores, all_gt_boxes)
    print(f"\nTest 3 - Synthetic dataset results:")
    print(f"  mAP50:    {results['mAP50']:.4f}")
    print(f"  mAP50-95: {results['mAP50_95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")

    # With 2/3 detections correct, recall should be around 0.67
    assert 0.5 < results['recall'] < 0.8, "Recall should reflect 2 out of 3 GT boxes found"

    print("\nAll evaluation metric sanity checks passed.")