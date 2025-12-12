from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class VideoDetectionTensor:
    """
    Holds per-frame confidence tensor for a single video.
    Convention: 0.0 for frames with no ball; >0.0 confidence for detected ball.
    """
    video_path: Path
    fps: float
    frame_count: int
    # 1D tensor of length frame_count or sampled_count (after frame skipping)
    confidences: np.ndarray  # dtype float16/float32, shape: [N]
    sample_rate: int = 1

    def detected_indices(self, threshold: float = 0.0) -> np.ndarray:
        return np.where(self.confidences > threshold)[0]

    def as_numpy(self) -> np.ndarray:
        return self.confidences

    def save_npz(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            video_path=str(self.video_path),
            fps=self.fps,
            frame_count=self.frame_count,
            sample_rate=self.sample_rate,
            confidences=self.confidences,
        )

    @staticmethod
    def load_npz(in_path: Path) -> "VideoDetectionTensor":
        data = np.load(in_path, allow_pickle=True)
        return VideoDetectionTensor(
            video_path=Path(str(data["video_path"])) ,
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            sample_rate=int(data["sample_rate"]),
            confidences=data["confidences"],
        )


class VideoObject:
    """
    Represents a video and its derived detection tensor.
    """
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.tensor: Optional[VideoDetectionTensor] = None

    def attach_tensor(self, tensor: VideoDetectionTensor) -> None:
        self.tensor = tensor

    def get_tensor(self) -> VideoDetectionTensor:
        if self.tensor is None:
            raise ValueError("Tensor not computed/attached for this video.")
        return self.tensor


def videos_to_objects(folder: Path) -> List[VideoObject]:
    """Scan a folder and create VideoObject entries for common video extensions."""
    exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    folder = Path(folder)
    if not folder.exists():
        return []
    return [VideoObject(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def train_test_split_videos(
    video_objects: List[VideoObject],
    test_ratio: float = 0.2,
    min_train: int = 16,
    min_test: int = 4,
) -> Tuple[List[VideoObject], List[VideoObject]]:
    """
    Deterministic split: first N for train, last M for test.
    Guarantees at least min_train/min_test when possible.
    """
    n = len(video_objects)
    if n == 0:
        return [], []
    # Compute test size by ratio, then ensure minimums
    test_size = max(int(n * test_ratio), min_test)
    train_size = max(n - test_size, min_train)
    # Adjust to not exceed total
    if train_size + test_size > n:
        test_size = n - train_size
        if test_size < 0:
            test_size = 0
    train = video_objects[:train_size]
    test = video_objects[train_size:train_size + test_size]
    return train, test


# -----------------------------
# Label utilities (binary per-frame CSV)
# -----------------------------
def _load_video_meta(video_path: Path) -> Tuple[int, float]:
    """Return (frame_count, fps) for the given video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, fps


def build_binary_labels_for_video(
    video_path: Path,
    timestamp_pairs_seconds: List[Tuple[float, float]],
    *,
    inclusive: bool = True,
) -> np.ndarray:
    """
    Create a binary array over frames: 1 if frame time (in seconds) falls within
    any of the provided (start_s, end_s) pairs, else 0. Timestamps are inclusive
    at boundaries when inclusive=True.

    Args:
        video_path: Path to the video file.
        timestamp_pairs_seconds: List of (start_seconds, end_seconds) pairs.
        inclusive: Include boundary frames when True.

    Returns:
        np.ndarray of shape [frame_count], dtype=np.uint8 with 0/1 values.
    """
    frame_count, fps = _load_video_meta(video_path)
    labels = np.zeros(frame_count, dtype=np.uint8)
    # Pre-normalize intervals so start <= end
    intervals = []
    for s, e in timestamp_pairs_seconds:
        if s > e:
            s, e = e, s
        intervals.append((max(0.0, float(s)), float(e)))

    # Fill labels by interval
    for s, e in intervals:
        # Convert seconds to frame indices
        if inclusive:
            start_idx = int(np.floor(s * fps))
            end_idx = int(np.floor(e * fps))
        else:
            start_idx = int(np.ceil(s * fps))
            end_idx = int(np.floor(e * fps) - 1)
        start_idx = max(0, start_idx)
        end_idx = min(frame_count - 1, end_idx)
        if end_idx >= start_idx:
            labels[start_idx:end_idx + 1] = 1
    return labels


def write_video_binary_labels_row(
    csv_path: Path,
    video_path: Path,
    labels: np.ndarray,
) -> None:
    """Append a CSV row: [video_name, b0, b1, ..., bN]."""
    import csv
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["video", "labels..."])  # simple header
        row = [Path(video_path).name] + [int(x) for x in labels.tolist()]
        w.writerow(row)


def build_and_write_video_labels_csv_row(
    csv_path: Path,
    video_path: Path,
    timestamp_pairs_seconds: List[Tuple[float, float]],
    inclusive: bool = True,
) -> np.ndarray:
    """
    Generate binary labels for a video based on timestamp pairs and append
    a single row to the CSV.

    Returns the labels array for further use.
    """
    labels = build_binary_labels_for_video(video_path, timestamp_pairs_seconds, inclusive=inclusive)
    write_video_binary_labels_row(csv_path, video_path, labels)
    return labels
