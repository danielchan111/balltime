#!/usr/bin/env python3
"""
Interactive labeling script:
- Plays a video for quick reference
- Prompts for timestamp pairs in seconds: start,end (comma-separated)
- Writes binary per-frame labels to CSV as a single row: [video_name, b0, b1, ..., bN]

Usage:
    # Label a single video
    python label_videos.py --video /path/to/video.mp4 --csv video_true_labels.csv

    # Label all videos found in a folder (default: ./videos)
    python label_videos.py --videos-dir ./videos --csv video_true_labels.csv

Controls in viewer:
    Press 'q' to quit playback and proceed to input intervals.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import cv2

# Ensure local module import when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
from video_objects import build_and_write_video_labels_csv_row


def play_video(video_path: Path, max_width: int = 960) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Playing: {video_path.name} | FPS: {fps:.2f}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imshow("Label Preview (press 'q' to stop)", frame)
        # ~30ms delay ~ 33 FPS for preview; adjust as needed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_intervals(prompt: str = "Enter intervals 'start,end' in seconds (one per line). Type 'done' to finish:\n") -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    print(prompt, end="")
    while True:
        line = input().strip()
        if not line:
            # empty line, continue
            continue
        if line.lower() in {"done", "q", "quit", "exit"}:
            break
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                print("Please enter exactly two comma-separated values: start,end")
                continue
            start_s = float(parts[0])
            end_s = float(parts[1])
            intervals.append((start_s, end_s))
        except Exception as e:
            print(f"Could not parse line: {line} | Error: {e}")
    return intervals


def _iter_videos_in_dir(videos_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    return [p for p in sorted(videos_dir.iterdir()) if p.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-frame binary labels from timestamp intervals.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", type=str, help="Path to a single video file")
    group.add_argument("--videos-dir", type=str, help="Directory containing videos to label (default: ./videos)")
    parser.add_argument("--csv", type=str, default="video_true_labels.csv", help="Output CSV path (appends if exists)")
    parser.add_argument("--inclusive", action="store_true", help="Include boundary frames of intervals")
    parser.add_argument("--no-preview", dest="no_preview", action="store_true", help="Skip video preview; go straight to interval input")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    videos: List[Path] = []
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)
        videos = [video_path]
    else:
        # Use directory mode
        videos_dir = Path(args.videos_dir) if args.videos_dir else (PROJECT_ROOT / "videos")
        if not videos_dir.exists():
            print(f"Videos directory not found: {videos_dir}")
            sys.exit(1)
        videos = _iter_videos_in_dir(videos_dir)
        if not videos:
            print(f"No videos found in directory: {videos_dir}")
            sys.exit(0)

    # Loop through videos, preview (optional), collect intervals, and append to CSV
    for vp in videos:
        print(f"\n=== Labeling: {vp.name} ===")
        if not args.no_preview:
            play_video(vp)
        intervals = parse_intervals()
        if not intervals:
            print(f"No intervals provided for {vp.name}. Skipping.")
            continue
        build_and_write_video_labels_csv_row(csv_path, vp, intervals, inclusive=args.inclusive)
        print(f"Saved labels for '{vp.name}' to {csv_path}")


if __name__ == "__main__":
    main()
