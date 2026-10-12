"""Visual scoring signals for drive pipeline segment selection.

Individual per-frame scoring functions, each returning 0-1 normalized scores.
Designed for batch processing at 1fps across entire video files.

Signals:
  - Color vibrancy: HSV saturation histogram mean
  - Scene change: Frame-to-frame histogram correlation distance
  - Sky ratio: Upper-frame brightness for openness detection
  - Golden hour: Warm color temperature in specific brightness range
"""

import cv2
import numpy as np


def score_color_vibrancy(frame: np.ndarray) -> float:
    """Score color vibrancy from HSV saturation.

    High saturation = fall colors, ocean, sunset. Low = overcast, gray.

    Args:
        frame: BGR uint8 image (any resolution).

    Returns:
        Score 0-1 (1 = highly vibrant).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_saturation = float(np.mean(hsv[:, :, 1]))
    # Scale: 0-200 saturation range mapped to 0-1
    # Typical driving footage: 40-120 saturation
    return min(avg_saturation / 180.0, 1.0)


def score_scene_change(prev_hist: np.ndarray, curr_hist: np.ndarray) -> float:
    """Score magnitude of scene change between two frames.

    Uses histogram correlation distance. Large distance = dramatic scenery
    change (came around a bend, exited forest into valley).

    Args:
        prev_hist: Normalized histogram (256 bins, sums to 1.0).
        curr_hist: Normalized histogram (256 bins, sums to 1.0).

    Returns:
        Score 0-1 (1 = maximum scene change).
    """
    # cv2.compareHist uses correlation: 1.0 = identical, -1.0 = opposite
    correlation = cv2.compareHist(
        prev_hist.astype(np.float32),
        curr_hist.astype(np.float32),
        cv2.HISTCMP_CORREL,
    )
    # Convert correlation [-1, 1] to distance [0, 1]
    distance = (1.0 - correlation) / 2.0
    return float(np.clip(distance, 0.0, 1.0))


def score_sky_ratio(frame: np.ndarray, upper_fraction: float = 0.33) -> float:
    """Score sky openness from upper-frame brightness.

    Bright open sky = scenic vista. Dark overhead = tunnel or dense trees.

    Args:
        frame: BGR uint8 image.
        upper_fraction: Fraction of frame height to analyze (default top third).

    Returns:
        Score 0-1 (1 = bright open sky).
    """
    h = frame.shape[0]
    upper_rows = int(h * upper_fraction)
    if upper_rows < 1:
        return 0.5

    upper_region = frame[:upper_rows]
    gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)

    # Brightness of upper region
    avg_brightness = float(np.mean(gray_upper))

    # Sky is typically bright (>150) with some texture (not solid)
    # Map 50-220 brightness to 0-1
    brightness_score = np.clip((avg_brightness - 50.0) / 170.0, 0.0, 1.0)

    # Penalize solid uniform upper region (overcast or solid color)
    std_brightness = float(np.std(gray_upper))
    # Some variation (clouds, gradients) is better than flat
    texture_bonus = np.clip(std_brightness / 50.0, 0.0, 1.0)

    # Combine: 70% brightness, 30% texture
    score = 0.7 * brightness_score + 0.3 * texture_bonus
    return float(np.clip(score, 0.0, 1.0))


def score_golden_hour(frame: np.ndarray) -> float:
    """Score golden hour warm color temperature.

    Golden hour: high red/orange warmth, moderate brightness, low blue.

    Args:
        frame: BGR uint8 image.

    Returns:
        Score 0-1 (1 = strong golden hour characteristics).
    """
    # Split channels (BGR order)
    b_mean = float(np.mean(frame[:, :, 0]))
    g_mean = float(np.mean(frame[:, :, 1]))
    r_mean = float(np.mean(frame[:, :, 2]))

    # Warmth ratio: high R relative to B indicates warm light
    # Avoid division by zero
    if b_mean < 1.0:
        warmth = r_mean / 1.0
    else:
        warmth = r_mean / b_mean

    # Golden hour warmth typically has R/B ratio > 1.3
    # Map ratio 1.0-2.5 to 0-1
    warmth_score = np.clip((warmth - 1.0) / 1.5, 0.0, 1.0)

    # Brightness check: golden hour is moderate brightness (not too dark, not blown out)
    overall_brightness = (r_mean + g_mean + b_mean) / 3.0
    # Optimal brightness range: 80-200
    if 80 <= overall_brightness <= 200:
        brightness_factor = 1.0
    elif overall_brightness < 80:
        brightness_factor = overall_brightness / 80.0
    else:
        brightness_factor = max(0.0, 1.0 - (overall_brightness - 200) / 55.0)

    # Green channel should be between R and B for golden tones
    # (not higher than R, which would indicate green foliage dominance)
    if r_mean > 1.0 and g_mean < r_mean:
        color_balance = 1.0
    else:
        color_balance = 0.3  # Still some score but not golden

    score = warmth_score * brightness_factor * color_balance
    return float(np.clip(score, 0.0, 1.0))


def compute_grayscale_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute normalized grayscale histogram for scene change detection.

    Args:
        frame: BGR uint8 image.

    Returns:
        Normalized 256-bin histogram (sums to 1.0).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
