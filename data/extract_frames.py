"""
Video Frame Extraction Script
================================
Extracts sparse frames from drone videos to add to the training dataset.

Usage:
    python extract_frames.py --input_dir path/to/videos --output_dir path/to/output --interval 10

This will:
  1. Read every .mp4 (or .mov/.avi) file in input_dir
  2. Extract every Nth frame (default: every 10th frame)
  3. Save frames as .jpg images named: <video_name>_frame_<number>.jpg

After running this, you still need to ANNOTATE the extracted frames
(draw bounding boxes around the drone in each image) before adding
them to your training dataset - this script only extracts images,
it does not label them.
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 10):
    """
    Extracts every `interval`-th frame from a single video file.
    Returns the number of frames saved.
    """
    video_name = Path(video_path).stem  # filename without extension
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  WARNING: Could not open {video_path}, skipping.")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % interval == 0:
            output_filename = f"{video_name}_frame_{frame_idx:05d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()

    print(f"  {video_name}: {total_frames} total frames @ {fps:.1f}fps -> {saved_count} frames extracted")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract sparse frames from drone videos")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Folder containing your .mp4/.mov/.avi video files")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Folder where extracted frames will be saved")
    parser.add_argument("--interval", type=int, default=10,
                         help="Save every Nth frame (default: 10, meaning ~3 frames/sec at 30fps)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_extensions = (".mp4", ".mov", ".avi", ".mkv")
    video_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} video(s) in {args.input_dir}\n")
    print(f"Extracting every {args.interval}th frame...\n")

    total_saved = 0
    for video_file in video_files:
        video_path = os.path.join(args.input_dir, video_file)
        total_saved += extract_frames_from_video(video_path, args.output_dir, args.interval)

    print(f"\nDone! Total frames extracted: {total_saved}")
    print(f"Saved to: {args.output_dir}")
    print("\nNext step: annotate these frames (draw bounding boxes) before")
    print("adding them to your training dataset.")


if __name__ == "__main__":
    main()