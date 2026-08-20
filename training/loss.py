"""
Custom Loss Functions - Built From Scratch
=============================================
Two losses combine to train the full model:

1. Focal Loss (classification/objectness):
   Standard cross-entropy treats every prediction equally, but in
   object detection the vast majority of grid cells are background
   (no drone). This "easy negative" imbalance can drown out the
   learning signal from the few cells that DO contain a drone.
   Focal Loss down-weights easy/confident predictions and focuses
   training on hard/uncertain ones.

2. CIoU Loss (bounding box localization):
   Directly targets the mAP50 vs mAP50-95 gap problem identified in
   earlier experiments. Plain L1/L2 loss on (x,y,w,h) treats box
   coordinates independently and doesn't correlate well with actual
   overlap quality. CIoU (Complete IoU) instead penalizes:
     - overlap area (standard IoU)
     - distance between box centers
     - aspect ratio consistency
   This produces tighter, more accurate boxes - directly improving
   the "precision at stricter IoU thresholds" (i.e. mAP50-95).
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for objectness/classification.
    FL(p) = -alpha * (1 - p)^gamma * log(p)      for positive samples
    FL(p) = -(1-alpha) * p^gamma * log(1 - p)    for negative samples
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: raw (pre-sigmoid) predictions, any shape
        target: same shape, values in {0, 1}
        """
        prob = torch.sigmoid(pred_logits)

        # Standard binary cross-entropy (numerically stable, from logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none"
        )

        # p_t = prob if target==1 else (1 - prob)
        p_t = prob * target + (1 - prob) * (1 - target)

        # alpha_t = alpha if target==1 else (1 - alpha)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


def compute_ciou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes CIoU between predicted and target boxes.
    Boxes are in (x_center, y_center, width, height) format.

    Returns: CIoU value per box, shape (N,), range roughly [-1, 1]
             (1 = perfect overlap, negative = far apart)
    """
    # Convert center-format to corner-format (x1, y1, x2, y2)
    def to_corners(boxes):
        x_c, y_c, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_boxes)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = to_corners(target_boxes)

    # --- Intersection area ---
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # --- Union area ---
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    # --- Center distance term ---
    pred_cx, pred_cy = pred_boxes[..., 0], pred_boxes[..., 1]
    tgt_cx, tgt_cy = target_boxes[..., 0], target_boxes[..., 1]
    center_dist_sq = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # --- Enclosing box diagonal (smallest box containing both) ---
    enclose_x1 = torch.min(pred_x1, tgt_x1)
    enclose_y1 = torch.min(pred_y1, tgt_y1)
    enclose_x2 = torch.max(pred_x2, tgt_x2)
    enclose_y2 = torch.max(pred_y2, tgt_y2)
    diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # --- Aspect ratio consistency term ---
    pred_w, pred_h = pred_boxes[..., 2], pred_boxes[..., 3]
    tgt_w, tgt_h = target_boxes[..., 2], target_boxes[..., 3]

    v = (4 / (3.14159265 ** 2)) * torch.pow(
        torch.atan(tgt_w / (tgt_h + eps)) - torch.atan(pred_w / (pred_h + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist_sq / diagonal_sq) - alpha * v
    return ciou


class CIoULoss(nn.Module):
    """
    CIoU Loss = 1 - CIoU
    (so a perfect box match -> loss of 0)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        ciou = compute_ciou(pred_boxes, target_boxes)
        loss = 1 - ciou
        return loss.mean()


class DetectionLoss(nn.Module):
    """
    Combines Focal Loss (objectness) + CIoU Loss (bbox) into the
    total loss used during training. Weights are configurable via
    config.yaml (box_loss_weight, cls_loss_weight) so their relative
    importance can be tuned without touching this code.
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config["training"]["loss"]

        self.focal_loss = FocalLoss(
            alpha=loss_cfg["focal_alpha"],
            gamma=loss_cfg["focal_gamma"],
        )
        self.ciou_loss = CIoULoss()

        self.box_weight = loss_cfg["box_loss_weight"]
        self.cls_weight = loss_cfg["cls_loss_weight"]

    def forward(self, pred_objectness: torch.Tensor, target_objectness: torch.Tensor,
                pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> dict:
        """
        pred_objectness / target_objectness: (N,) raw logits / 0-1 labels
        pred_boxes / target_boxes: (N, 4) in (x_center, y_center, w, h) format
                                    - only for POSITIVE samples (cells with a real drone)
        """
        cls_loss = self.focal_loss(pred_objectness, target_objectness)

        if pred_boxes.shape[0] > 0:
            box_loss = self.ciou_loss(pred_boxes, target_boxes)
        else:
            # No positive samples in this batch - box loss contributes 0
            box_loss = torch.tensor(0.0, device=pred_objectness.device)

        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
        }


if __name__ == "__main__":
    import yaml

    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Sanity check 1: Focal Loss ---
    focal = FocalLoss()
    dummy_logits = torch.randn(100)
    dummy_targets = torch.randint(0, 2, (100,)).float()
    fl_value = focal(dummy_logits, dummy_targets)
    print(f"Focal Loss sanity check: {fl_value.item():.4f}")
    assert fl_value.item() >= 0, "Focal loss should be non-negative"

    # --- Sanity check 2: CIoU Loss with a perfect match ---
    perfect_pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    perfect_target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    ciou_loss_fn = CIoULoss()
    perfect_loss = ciou_loss_fn(perfect_pred, perfect_target)
    print(f"CIoU Loss (perfect match, should be ~0): {perfect_loss.item():.4f}")
    assert perfect_loss.item() < 0.01, "Perfect match should give near-zero loss"

    # --- Sanity check 3: CIoU Loss with boxes far apart ---
    far_pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    far_target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    far_loss = ciou_loss_fn(far_pred, far_target)
    print(f"CIoU Loss (far apart, should be higher): {far_loss.item():.4f}")
    assert far_loss.item() > perfect_loss.item(), "Far apart boxes should have higher loss"

    # --- Sanity check 4: Combined DetectionLoss ---
    detection_loss = DetectionLoss(config)
    pred_obj = torch.randn(50)
    tgt_obj = torch.randint(0, 2, (50,)).float()
    pred_boxes = torch.rand(10, 4)
    tgt_boxes = torch.rand(10, 4)

    losses = detection_loss(pred_obj, tgt_obj, pred_boxes, tgt_boxes)
    print(f"\nCombined Detection Loss:")
    print(f"  total_loss: {losses['total_loss'].item():.4f}")
    print(f"  cls_loss:   {losses['cls_loss'].item():.4f}")
    print(f"  box_loss:   {losses['box_loss'].item():.4f}")

    print("\nAll loss function sanity checks passed.")