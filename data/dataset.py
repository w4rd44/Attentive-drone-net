"""
Dataset Loader - Built From Scratch
======================================
Loads images + YOLO-format bounding box labels and prepares them
for training. Works with ANY YOLO-format dataset directory (the
video-extracted frames from Roboflow, and later your existing 14k
image dataset too) - just point `images_dir` / `labels_dir` at it.

YOLO label format (one .txt file per image, same name):
    class_id  x_center  y_center  width  height
    (all values normalized 0-1, relative to image size)

This loader:
  1. Reads the image + its matching label file
  2. Resizes the image to a square (config input_size) using
     LETTERBOX resizing (pad with gray bars, don't stretch) -
     matches what we set up in Roboflow ("Fit within 640x640")
  3. Adjusts box coordinates to match the letterboxed image
  4. Returns (image_tensor, boxes, labels) ready for training

Because different images can have different numbers of drones
(some 1, some 0, some multiple), we use a custom `collate_fn` so
PyTorch's DataLoader can batch images with a variable number of
boxes together.
"""

import os
import cv2
import torch
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader


def letterbox_resize(image: np.ndarray, target_size: int, pad_color=(114, 114, 114)):
    """
    Resizes an image to (target_size, target_size) while preserving
    aspect ratio, padding the rest with a neutral gray color.

    Returns: resized_image, scale_factor, (pad_x, pad_y)
    These are needed to correctly adjust bounding box coordinates.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a square canvas filled with pad_color, then paste resized image centered
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


class DroneDataset(Dataset):
    """
    PyTorch Dataset for YOLO-format drone detection data.

    Args:
        data_yaml_path: path to the dataset's data.yaml (has train/val/test paths + class names)
        split: one of "train", "val", "test"
        input_size: target square size to resize images to (from config)
    """

    def __init__(self, data_yaml_path: str, split: str = "train", input_size: int = 640):
        with open(data_yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)

        self.input_size = input_size
        self.class_names = data_cfg["names"]

        # data.yaml paths are USUALLY relative to the yaml file's own directory,
        # but Roboflow exports sometimes write "../train/images" even when
        # train/valid/test are actually siblings of data.yaml (not one level up).
        # We try the literal relative path first, and fall back to treating it
        # as a sibling folder if that path doesn't actually exist.
        base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        split_key = "val" if split == "val" else split  # data.yaml uses "val" not "valid"
        images_rel_path = data_cfg[split_key]  # e.g. "../valid/images"

        candidate_1 = os.path.normpath(os.path.join(base_dir, images_rel_path))
        # Strip any leading "../" segments and re-join as a sibling of data.yaml
        stripped_path = images_rel_path.replace("../", "").replace("..\\", "")
        candidate_2 = os.path.normpath(os.path.join(base_dir, stripped_path))

        if os.path.isdir(candidate_1):
            self.images_dir = candidate_1
        elif os.path.isdir(candidate_2):
            self.images_dir = candidate_2
        else:
            raise FileNotFoundError(
                f"Could not locate '{split}' images folder. Tried:\n"
                f"  {candidate_1}\n  {candidate_2}"
            )

        self.labels_dir = self.images_dir.replace("images", "labels")

        valid_ext = (".jpg", ".jpeg", ".png")
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_ext)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(image_filename)[0] + ".txt"
        )

        # --- Load and letterbox-resize the image ---
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        resized_image, scale, (pad_x, pad_y) = letterbox_resize(image, self.input_size)

        # --- Load labels (if the file exists; some images may have none) ---
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]

                    if len(values) == 4:
                        # Standard YOLO detection format: xc, yc, w, h
                        xc, yc, w, h = values
                    else:
                        # Polygon/segmentation format (e.g. from SAM3 auto-label):
                        # class x1 y1 x2 y2 x3 y3 ... (many normalized points)
                        # Convert to a bounding box by taking the min/max extent.
                        xs = values[0::2]
                        ys = values[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        xc = (x_min + x_max) / 2
                        yc = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min

                    # Convert normalized coords (relative to ORIGINAL image)
                    # into pixel coords, apply the same letterbox transform,
                    # then re-normalize (relative to the RESIZED square image).
                    xc_px = xc * orig_w * scale + pad_x
                    yc_px = yc * orig_h * scale + pad_y
                    w_px = w * orig_w * scale
                    h_px = h * orig_h * scale

                    xc_norm = xc_px / self.input_size
                    yc_norm = yc_px / self.input_size
                    w_norm = w_px / self.input_size
                    h_norm = h_px / self.input_size

                    boxes.append([xc_norm, yc_norm, w_norm, h_norm])
                    labels.append(class_id)

        # --- Convert to tensors ---
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        return image_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    """
    Custom collate function: since each image can have a different
    number of boxes, we can't just torch.stack() them like normal.
    Images ARE stacked (all same size after letterboxing); boxes and
    labels are kept as a list of variable-length tensors.
    """
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(boxes), list(labels)


def build_dataloader(data_yaml_path: str, split: str, config: dict, shuffle: bool = None):
    """
    Factory function - builds a DataLoader from the config dict,
    consistent with how build_model() works for the model (roadmap Section 9).
    """
    input_size = config["data"]["input_size"]
    batch_size = config["training"]["batch_size"]

    if shuffle is None:
        shuffle = (split == "train")  # shuffle training data, not val/test

    dataset = DroneDataset(data_yaml_path, split=split, input_size=input_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 0 is safest on Windows; increase later on Linux/Kaggle
    )
    return dataloader, dataset


if __name__ == "__main__":
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_yaml_path = "data/raw_video_dataset/data.yaml"

    print("Loading train split...")
    train_loader, train_dataset = build_dataloader(data_yaml_path, "train", config)
    print(f"  Number of training images: {len(train_dataset)}")
    print(f"  Class names: {train_dataset.class_names}")

    print("\nLoading val split...")
    val_loader, val_dataset = build_dataloader(data_yaml_path, "val", config)
    print(f"  Number of validation images: {len(val_dataset)}")

    print("\nFetching one batch from train_loader...")
    images, boxes, labels = next(iter(train_loader))
    print(f"  Batch image tensor shape: {images.shape}  (B, 3, H, W)")
    print(f"  Number of images in batch: {len(boxes)}")
    for i, (b, l) in enumerate(zip(boxes, labels)):
        print(f"    Image {i}: {b.shape[0]} box(es), labels={l.tolist()}")

    assert images.shape[1:] == (3, config["data"]["input_size"], config["data"]["input_size"]), \
        "Image shape doesn't match expected input size"

    print("\nDataset loader sanity check passed.")