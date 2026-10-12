#!/usr/bin/env python3
"""
Quick test of motion analysis on a video file.
Requires only opencv-python and numpy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def analyze_motion(video_path: str, sample_fps: int = 5, motion_threshold: float = 0.02):
    """Analyze video for motion and find dead spots."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {Path(video_path).name}")
    print(f"  Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print()

    frame_skip = max(1, int(fps / sample_fps))
    print(f"Sampling every {frame_skip} frames ({fps / frame_skip:.1f} fps effective)")
    print()

    motion_data = []
    prev_gray = None
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_energy = np.mean(diff) / 255.0
                timestamp = frame_idx / fps
                motion_data.append((timestamp, motion_energy))

            prev_gray = gray
            processed += 1

            # Progress
            if processed % 50 == 0:
                progress = frame_idx / total_frames * 100
                print(f"\rProcessing: {progress:.0f}%", end="", flush=True)

        frame_idx += 1

    cap.release()
    print("\rProcessing: 100%")
    print()

    # Find dead spots
    print("Motion Analysis Results:")
    print("-" * 40)

    # Stats
    energies = [e for _, e in motion_data]
    print(
        f"  Motion energy - min: {min(energies):.4f}, max: {max(energies):.4f}, avg: {np.mean(energies):.4f}"
    )
    print(f"  Threshold: {motion_threshold}")
    print()

    # Find low motion periods
    dead_spots = []
    in_dead = False
    spot_start = 0

    for ts, energy in motion_data:
        if energy < motion_threshold:
            if not in_dead:
                in_dead = True
                spot_start = ts
        else:
            if in_dead:
                duration = ts - spot_start
                if duration >= 5:  # At least 5 seconds
                    dead_spots.append((spot_start, ts, duration))
                in_dead = False

    if dead_spots:
        print(f"Found {len(dead_spots)} potential dead spots (>5s of low motion):")
        for start, end, dur in dead_spots:
            print(f"  {start:.1f}s - {end:.1f}s ({dur:.1f}s)")
    else:
        print("No significant dead spots found")

    print()

    # Show motion over time (simple ASCII graph)
    print("Motion over time (ASCII graph):")
    print("-" * 60)

    # Downsample for display
    bucket_size = max(1, len(motion_data) // 50)
    buckets = []
    for i in range(0, len(motion_data), bucket_size):
        chunk = motion_data[i : i + bucket_size]
        avg_energy = np.mean([e for _, e in chunk])
        avg_time = np.mean([t for t, _ in chunk])
        buckets.append((avg_time, avg_energy))

    max_energy = max(e for _, e in buckets)
    for ts, energy in buckets:
        bar_len = int(energy / max_energy * 40) if max_energy > 0 else 0
        bar = "#" * bar_len
        indicator = " " if energy >= motion_threshold else "*"
        print(f"{ts:6.1f}s {indicator} |{bar}")

    print()
    print("Legend: * = below motion threshold (potential dead spot)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_motion.py <video_path>")
        sys.exit(1)

    analyze_motion(sys.argv[1])
