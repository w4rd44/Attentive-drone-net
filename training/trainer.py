"""
Training Loop - Built From Scratch
=====================================
Ties together Model + Dataset + Loss to actually train the network.

Key piece this file adds: TARGET ASSIGNMENT.
The model produces predictions at every grid cell of 3 different
scales. But our dataset only gives us "here are the drone boxes in
this image" - it doesn't say WHICH grid cell / WHICH scale each box
belongs to. We have to figure that out ourselves before we can
compute a loss.

Assignment strategy (simple, from-scratch anchor-free approach):
  1. For each ground-truth box, look at its size (w, h) to decide
     which SCALE it belongs to:
       - large boxes  -> Scale 1 (20x20 grid, large receptive field)
       - medium boxes -> Scale 2 (40x40 grid)
       - small boxes  -> Scale 3 (80x80 grid, best for tiny/distant drones)
  2. Within that scale's grid, find the single cell that contains
     the box's center point - that cell becomes a "positive" sample
     (objectness target = 1) and is trained to regress that box.
  3. Every other cell (at every scale) is a "negative" sample
     (objectness target = 0, no box to regress).
"""

import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from models.model import build_model
from data.dataset import build_dataloader
from training.loss import DetectionLoss


def assign_box_to_scale(box_w: float, box_h: float, num_scales: int) -> int:
    """
    Decides which scale (0 = largest/deepest, num_scales-1 = smallest/shallowest)
    a ground-truth box should be assigned to, based on its size.
    Sizes are normalized (0-1 relative to image size).
    """
    box_size = max(box_w, box_h)

    if num_scales == 3:
        if box_size > 0.30:
            return 0  # large objects -> scale 1 (20x20 grid)
        elif box_size > 0.12:
            return 1  # medium objects -> scale 2 (40x40 grid)
        else:
            return 2  # small/distant objects -> scale 3 (80x80 grid)
    else:
        # Generic fallback for a different num_scales configuration:
        # split evenly by size percentile.
        idx = int((1 - box_size) * num_scales)
        return min(max(idx, 0), num_scales - 1)


def build_targets(boxes_list: list, labels_list: list, predictions: list, device):
    """
    Builds per-scale target tensors matching the model's prediction shapes.

    Args:
        boxes_list: list (len=batch) of (N_i, 4) tensors - GT boxes per image
        labels_list: list (len=batch) of (N_i,) tensors - GT class ids per image
        predictions: list (len=num_scales) of dicts from the model's forward pass,
                     used only to read grid sizes (H, W) for each scale

    Returns:
        targets: list (len=num_scales) of dicts:
            {
                "objectness": (B, 1, H, W) tensor of 0/1,
                "positive_mask": (B, H, W) bool tensor - which cells are positive,
                "boxes": (B, H, W, 4) tensor - target box at each cell (only valid where positive)
            }
    """
    batch_size = len(boxes_list)
    num_scales = len(predictions)

    targets = []
    for scale_idx, pred in enumerate(predictions):
        _, _, H, W = pred["objectness"].shape
        objectness = torch.zeros((batch_size, 1, H, W), device=device)
        positive_mask = torch.zeros((batch_size, H, W), dtype=torch.bool, device=device)
        box_targets = torch.zeros((batch_size, H, W, 4), device=device)
        targets.append({
            "objectness": objectness,
            "positive_mask": positive_mask,
            "boxes": box_targets,
        })

    for b in range(batch_size):
        boxes = boxes_list[b]
        for box in boxes:
            xc, yc, w, h = box.tolist()
            assigned_scale = assign_box_to_scale(w, h, num_scales)

            _, _, H, W = predictions[assigned_scale]["objectness"].shape
            cell_x = min(int(xc * W), W - 1)
            cell_y = min(int(yc * H), H - 1)

            targets[assigned_scale]["objectness"][b, 0, cell_y, cell_x] = 1.0
            targets[assigned_scale]["positive_mask"][b, cell_y, cell_x] = True
            targets[assigned_scale]["boxes"][b, cell_y, cell_x] = torch.tensor(
                [xc, yc, w, h], device=device
            )

    return targets


def compute_batch_loss(predictions: list, targets: list, loss_fn: DetectionLoss):
    """
    Computes total loss across all scales for one batch by:
      1. Flattening objectness predictions/targets across the whole
         grid (every cell contributes to classification loss)
      2. Gathering ONLY the positive-cell box predictions/targets
         (only cells with a real drone contribute to box loss)
    """
    all_pred_obj = []
    all_tgt_obj = []
    all_pred_boxes = []
    all_tgt_boxes = []

    for pred, tgt in zip(predictions, targets):
        # Objectness: every cell, flattened
        all_pred_obj.append(pred["objectness"].flatten())
        all_tgt_obj.append(tgt["objectness"].flatten())

        # Boxes: only positive cells
        pos_mask = tgt["positive_mask"]  # (B, H, W)
        if pos_mask.any():
            # pred["bbox"] is (B, 4, H, W) -> permute to (B, H, W, 4) to match mask indexing
            pred_bbox = pred["bbox"].permute(0, 2, 3, 1)
            all_pred_boxes.append(pred_bbox[pos_mask])   # (num_positives, 4)
            all_tgt_boxes.append(tgt["boxes"][pos_mask])  # (num_positives, 4)

    pred_obj_flat = torch.cat(all_pred_obj)
    tgt_obj_flat = torch.cat(all_tgt_obj)

    if all_pred_boxes:
        pred_boxes_flat = torch.cat(all_pred_boxes, dim=0)
        tgt_boxes_flat = torch.cat(all_tgt_boxes, dim=0)
    else:
        pred_boxes_flat = torch.zeros((0, 4), device=pred_obj_flat.device)
        tgt_boxes_flat = torch.zeros((0, 4), device=pred_obj_flat.device)

    losses = loss_fn(pred_obj_flat, tgt_obj_flat, pred_boxes_flat, tgt_boxes_flat)
    return losses


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch_num: int):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
    for images, boxes_list, labels_list in progress_bar:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        predictions = model(images)
        targets = build_targets(boxes_list, labels_list, predictions, device)
        losses = compute_batch_loss(predictions, targets, loss_fn)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        total_loss += losses["total_loss"].item()
        total_cls_loss += losses["cls_loss"].item()
        total_box_loss += losses["box_loss"].item()
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{losses['total_loss'].item():.4f}",
            "cls": f"{losses['cls_loss'].item():.4f}",
            "box": f"{losses['box_loss'].item():.4f}",
        })

    return {
        "avg_total_loss": total_loss / num_batches,
        "avg_cls_loss": total_cls_loss / num_batches,
        "avg_box_loss": total_box_loss / num_batches,
    }


def main(num_epochs: int = 5, data_yaml_path: str = "data/raw_video_dataset/data.yaml"):
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nBuilding model...")
    model = build_model(config).to(device)
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")

    print("\nLoading dataset...")
    train_loader, train_dataset = build_dataloader(data_yaml_path, "train", config)
    print(f"  Training images: {len(train_dataset)}")

    loss_fn = DetectionLoss(config)

    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nStarting training for {num_epochs} epochs...\n")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"total_loss: {metrics['avg_total_loss']:.4f}, "
            f"cls_loss: {metrics['avg_cls_loss']:.4f}, "
            f"box_loss: {metrics['avg_box_loss']:.4f}"
        )

        checkpoint_path = f"checkpoints/epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    print("\nTraining complete. Checkpoints saved in checkpoints/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AttentiveDroneNet")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5, for quick sanity test)")
    parser.add_argument("--data", type=str, default="data/raw_video_dataset/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    main(num_epochs=args.epochs, data_yaml_path=args.data)