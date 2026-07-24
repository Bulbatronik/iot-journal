from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def resolve_detection_mode(dataset_name: str = None, detection_mode: str = "auto") -> str:
    if detection_mode != "auto":
        return detection_mode

    dataset_name = (dataset_name or "").lower()
    if dataset_name == "swat":
        return "pointwise"
    return "persistent"


def resolve_score_transform(
    dataset_name: str = None,
    score_transform: str = "auto",
    detection_mode: str = "auto",
) -> str:
    if score_transform != "auto":
        return score_transform

    dataset_name = (dataset_name or "").lower()
    if dataset_name == "swat":
        resolved_mode = resolve_detection_mode(dataset_name, detection_mode)
        if resolved_mode == "pointwise":
            return "raw"
        return "rolling_delta"
    return "raw"


def apply_score_transform(
    scores: np.ndarray,
    score_transform: str = "raw",
    drift_window: int = None,
) -> np.ndarray:
    scores = np.asarray(scores).squeeze().astype(float)

    if score_transform == "raw":
        return scores

    if score_transform == "rolling_delta":
        window = max(1, int(drift_window) if drift_window is not None else 1)
        baseline = pd.Series(scores).rolling(window=window, min_periods=1).median().to_numpy()
        return np.clip(scores - baseline, 0.0, None)

    raise ValueError(f"Unknown score_transform: {score_transform}")


def apply_score_filter(scores: np.ndarray, filter_size: int = None) -> np.ndarray:
    scores = np.asarray(scores).squeeze()
    if filter_size is None or int(filter_size) <= 1:
        return scores
    return median_filter(scores, size=int(filter_size))


def prepare_scores(
    scores: np.ndarray,
    filter_size: int = None,
    score_transform: str = "raw",
    drift_window: int = None,
) -> np.ndarray:
    scores = apply_score_transform(
        scores,
        score_transform=score_transform,
        drift_window=drift_window,
    )
    return apply_score_filter(scores, filter_size)


def _enforce_min_consecutive(binary: np.ndarray, min_consecutive: int) -> np.ndarray:
    binary = np.asarray(binary).astype(int).reshape(-1)
    if min_consecutive <= 1 or binary.size == 0:
        return binary

    result = np.zeros_like(binary)
    padded = np.pad(binary, (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        if (end - start) >= min_consecutive:
            result[start:end] = 1

    return result


def predict_anomalies(
    scores: np.ndarray,
    threshold: float,
    detection_mode: str = "persistent",
    min_consecutive: int = 1,
) -> np.ndarray:
    scores = np.asarray(scores).squeeze()
    above = (scores > threshold).astype(int)

    if detection_mode == "pointwise":
        return _enforce_min_consecutive(above, min_consecutive)

    if detection_mode != "persistent":
        raise ValueError(f"Unknown detection_mode: {detection_mode}")

    above = _enforce_min_consecutive(above, min_consecutive)
    indices = np.flatnonzero(above == 1)
    pred = np.zeros_like(above)
    if indices.size:
        pred[indices[0]:] = 1
    return pred


def first_positive_index(values: Sequence[int]) -> float:
    indices = np.flatnonzero(np.asarray(values).reshape(-1) == 1)
    return float(indices[0]) if indices.size else np.nan


def get_anomaly_spans(values: Sequence[int]) -> List[Tuple[int, int]]:
    values = np.asarray(values).astype(int).reshape(-1)
    if values.size == 0 or not np.any(values):
        return []

    padded = np.pad(values, (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def compute_event_stats(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> dict:
    true_labels = np.asarray(true_labels).astype(int).reshape(-1)
    pred_labels = np.asarray(pred_labels).astype(int).reshape(-1)

    true_spans = get_anomaly_spans(true_labels)
    pred_spans = get_anomaly_spans(pred_labels)

    matched_pred = set()
    tp_events = 0
    fn_events = 0
    delta_early = []
    delta_late = []

    for true_start, true_end in true_spans:
        overlaps = []
        for pred_idx, (pred_start, pred_end) in enumerate(pred_spans):
            if pred_idx in matched_pred:
                continue
            if pred_end <= true_start or pred_start >= true_end:
                continue
            overlaps.append((pred_idx, pred_start, pred_end))

        if not overlaps:
            fn_events += 1
            continue

        pred_idx, pred_start, _ = min(overlaps, key=lambda item: item[1])
        matched_pred.add(pred_idx)
        tp_events += 1

        if pred_start < true_start:
            delta_early.append(true_start - pred_start)
        elif pred_start > true_start:
            delta_late.append(pred_start - true_start)

    fp_events = len(pred_spans) - len(matched_pred)

    return {
        "TP": int(tp_events),
        "FP": int(fp_events),
        "FN": int(fn_events),
        "delta_early": delta_early,
        "delta_late": delta_late,
        "num_true_events": len(true_spans),
        "num_pred_events": len(pred_spans),
    }
