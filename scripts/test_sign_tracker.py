#!/usr/bin/env python3
"""
Sign tracker - when text is found, look at nearby frames to get best read.

Strategy:
1. Scan at coarse interval (e.g., every 1 second)
2. When text is found, go back and scan nearby frames at fine interval
3. Pick the frame with highest confidence / most text
4. Deduplicate to avoid logging same sign multiple times
"""

import sys
from pathlib import Path

import cv2


def read_frame_text(reader, cap, frame_num, min_conf=0.3):
    """Read text from a specific frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = reader.readtext(rgb)
    return [(text, conf) for (_, text, conf) in result if conf >= min_conf and len(text) >= 2]


def track_signs(
    video_path: str,
    coarse_interval: float = 1.0,
    fine_interval: float = 0.1,
    look_back: float = 1.5,
    look_ahead: float = 0.5,
):
    """
    Two-pass sign detection:
    1. Coarse scan to find frames with text
    2. Fine scan around those frames to get best reading
    """
    try:
        import easyocr
    except ImportError:
        print("EasyOCR not installed. Install with: pip install easyocr")
        return

    print("Initializing EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {Path(video_path).name}")
    print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}")
    print(f"Pass 1: Coarse scan every {coarse_interval}s", flush=True)

    # Pass 1: Coarse scan - just find timestamps with text
    coarse_hits = []
    t = 0.0
    while t < duration:
        frame_num = int(t * fps)
        texts = read_frame_text(reader, cap, frame_num)
        if texts:
            coarse_hits.append(t)
            print(f"  {t:.1f}s: {len(texts)} texts found", flush=True)
        t += coarse_interval

    print(f"\nPass 1: {len(coarse_hits)} frames with text", flush=True)

    if not coarse_hits:
        print("No text found in video")
        cap.release()
        return

    # Merge nearby hits into search windows
    windows = []
    for ts in coarse_hits:
        window_start = max(0, ts - look_back)
        window_end = min(duration, ts + look_ahead)
        if windows and window_start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], window_end)
        else:
            windows.append((window_start, window_end))

    print(f"Merged into {len(windows)} search windows", flush=True)

    # Pass 2: Fine scan within windows
    print(f"\nPass 2: Fine scan every {fine_interval}s", flush=True)

    all_detections = []

    for win_idx, (win_start, win_end) in enumerate(windows):
        print(f"\nWindow {win_idx + 1}: {win_start:.1f}s - {win_end:.1f}s", flush=True)

        best_timestamp = None
        best_texts = []
        best_score = 0

        t = win_start
        while t <= win_end:
            frame_num = int(t * fps)
            texts = read_frame_text(reader, cap, frame_num)

            if texts:
                score = sum(conf for _, conf in texts)
                print(f"    {t:.2f}s: {len(texts)} texts (score={score:.2f})", flush=True)
                for txt, conf in texts:
                    print(f"         [{conf:.2f}] {txt}", flush=True)

                if score > best_score:
                    best_score = score
                    best_timestamp = t
                    best_texts = texts

            t += fine_interval

        if best_texts:
            all_detections.append((best_timestamp, best_texts, best_score))
            print(f"  -> Best: {best_timestamp:.2f}s (score={best_score:.2f})", flush=True)

    cap.release()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for timestamp, texts, score in all_detections:
        print(f"\n{timestamp:.2f}s:")
        for text, conf in sorted(texts, key=lambda x: -x[1]):
            print(f"  [{conf:.2f}] {text}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_sign_tracker.py <video_path> [coarse_interval] [fine_interval]")
        print()
        print("  coarse_interval: seconds between frames in initial scan (default: 1.0)")
        print("  fine_interval: seconds between frames in detailed scan (default: 0.1)")
        sys.exit(1)

    video_path = sys.argv[1]
    coarse = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    fine = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    track_signs(video_path, coarse_interval=coarse, fine_interval=fine)


if __name__ == "__main__":
    main()
