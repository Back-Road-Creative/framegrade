#!/usr/bin/env python3
"""
Quick test of OCR on video frames.
Extracts frames at intervals and runs PaddleOCR on them.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def test_ocr(video_path: str, interval_seconds: float = 1.0, max_frames: int = 20):
    """Extract frames and run OCR."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
        return

    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(lang="en")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {Path(video_path).name}")
    print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}")
    print(f"Extracting frames every {interval_seconds}s (max {max_frames} frames)")
    print()

    frame_interval = int(fps * interval_seconds)
    results = []
    frame_idx = 0
    frames_processed = 0

    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            print(f"Processing frame at {timestamp:.1f}s...", end=" ")

            # Run OCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ocr_result = ocr.predict(rgb_frame)

            texts = []
            if ocr_result:
                # New PaddleOCR returns dict with 'rec_texts', 'rec_scores', etc.
                if isinstance(ocr_result, dict):
                    rec_texts = ocr_result.get("rec_texts", [])
                    rec_scores = ocr_result.get("rec_scores", [])
                    for text, score in zip(rec_texts, rec_scores):
                        if score >= 0.5 and len(text) >= 2:
                            texts.append((text, score))
                # Old API compatibility
                elif isinstance(ocr_result, list) and ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        if line and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, tuple) and len(text_info) >= 2:
                                text, conf = text_info[0], text_info[1]
                                if conf >= 0.5 and len(text) >= 2:
                                    texts.append((text, conf))

            if texts:
                print(f"Found {len(texts)} text regions:")
                for text, conf in texts:
                    print(f"    [{conf:.2f}] {text}")
                results.append((timestamp, texts))
            else:
                print("No text found")

            frames_processed += 1

        frame_idx += 1

    cap.release()

    print()
    print("=" * 50)
    print("Summary of detected text:")
    print("=" * 50)

    if results:
        for timestamp, texts in results:
            combined = " | ".join([t for t, _ in texts])
            print(f"{timestamp:6.1f}s: {combined}")
    else:
        print("No text detected in any frames")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <video_path> [interval_seconds] [max_frames]")
        sys.exit(1)

    video_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    test_ocr(video_path, interval, max_frames)
