"""
video_tools.py — PRE-PROCESSING
Extracts frames from raw .mp4 files and resizes to 480×832 for Wan 2.1 VAE.
Designed for Google Colab GPU environment.
"""

import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image


def extract_frames(video_path: str, num_frames: int = 8) -> list[np.ndarray]:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to .mp4 file
        num_frames: Number of frames to extract (default 8)

    Returns:
        List of BGR numpy arrays (raw OpenCV frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        raise ValueError(
            f"Video {video_path} has only {total_frames} frames, need {num_frames}"
        )

    # Calculate evenly-spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")

    cap.release()
    return frames


def resize_and_crop(frame: np.ndarray, target_w: int = 832, target_h: int = 480) -> np.ndarray:
    """
    Resize frame to target dimensions with center-crop to maintain aspect ratio.

    Args:
        frame: BGR numpy array
        target_w: Target width (default 832)
        target_h: Target height (default 480)

    Returns:
        Resized & cropped BGR numpy array of shape (target_h, target_w, 3)
    """
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    frame_aspect = w / h

    if frame_aspect > target_aspect:
        # Frame is wider — crop width
        new_h = h
        new_w = int(h * target_aspect)
    else:
        # Frame is taller — crop height
        new_w = w
        new_h = int(w / target_aspect)

    # Center crop
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]

    # Resize to exact target
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return resized


def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """
    Convert list of BGR numpy frames to a normalized float16 tensor.

    Args:
        frames: List of BGR numpy arrays (all same shape)

    Returns:
        Tensor of shape (num_frames, 3, H, W), normalized to [-1, 1], dtype float16
    """
    processed = []
    for frame in frames:
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        normalized = (rgb.astype(np.float32) / 127.5) - 1.0
        # HWC → CHW
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        processed.append(tensor)

    # Stack into (N, C, H, W) and convert to float16 to save VRAM
    batch = torch.stack(processed).half()
    return batch


def prepare_video_batch(raw_dir: str, num_frames: int = 8, target_w: int = 832, target_h: int = 480):
    """
    Scan raw directory, pair .mp4 with .txt captions, extract + resize frames.

    Args:
        raw_dir: Path to data/raw/ containing .mp4 and .txt files
        num_frames: Frames to extract per video
        target_w: Target frame width
        target_h: Target frame height

    Yields:
        Tuple of (frames_tensor, caption_text, video_stem_name)
    """
    video_files = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))

    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {raw_dir}")

    for video_path in video_files:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        caption_path = os.path.join(raw_dir, f"{stem}.txt")

        if not os.path.exists(caption_path):
            raise FileNotFoundError(
                f"Missing caption for {stem}.mp4 — expected {caption_path}"
            )

        # Read caption
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Extract and resize frames
        raw_frames = extract_frames(video_path, num_frames)
        resized_frames = [resize_and_crop(fr, target_w, target_h) for fr in raw_frames]

        # Convert to tensor
        frames_tensor = frames_to_tensor(resized_frames)

        yield frames_tensor, caption, stem


def scan_raw_directory(raw_dir: str) -> list[dict]:
    """
    Scan data/raw/ and return status of all video-caption pairs.
    Used by the Gradio UI Data Manager tab.

    Returns:
        List of dicts with keys: video, caption_file, has_caption, video_stem
    """
    video_files = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))
    results = []

    for video_path in video_files:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        caption_path = os.path.join(raw_dir, f"{stem}.txt")
        has_caption = os.path.exists(caption_path)

        results.append({
            "video": os.path.basename(video_path),
            "caption_file": f"{stem}.txt",
            "has_caption": has_caption,
            "video_stem": stem,
        })

    return results
