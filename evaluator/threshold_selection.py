import json
from copy import deepcopy
from typing import Dict
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real, Integer

from evaluator.detection import (
    compute_event_stats,
    first_positive_index,
    prepare_scores,
    predict_anomalies,
    resolve_detection_mode,
    resolve_score_transform,
)
from evaluator.aramis_cost import aramis_metric


def _to_sequence_list(values) -> list:
    return [np.asarray(seq).squeeze() for seq in values.tolist()]


def predict_changepoint(scores: np.ndarray, threshold: float, min_consecutive: int = 1) -> np.ndarray:
    """
    Predict changepoint/anomaly based on threshold and minimum consecutive anomalies.
    
    This function identifies the first occurrence of 'min_consecutive' consecutive values
    above the threshold and marks all points after that as anomalies.
    
    Parameters:
        scores: Array of mean squared errors or anomaly scores
        threshold: Threshold value above which a point is considered anomalous
        min_consecutive: Minimum number of consecutive points required to trigger detection
        
    Returns:
        np.ndarray: Binary array where 1 indicates anomaly, starting from the first detected point
    """
    return predict_anomalies(
        scores=scores,
        threshold=threshold,
        detection_mode="persistent",
        min_consecutive=min_consecutive,
    )


def dtau(
    score_tensor: np.ndarray,
    label_tensor: np.ndarray,
    threshold: float,
    filter_size: int = None,
    detection_mode: str = "persistent",
    min_consecutive: int = 1,
    score_transform: str = "raw",
    drift_window: int = None,
) -> float:
    """
    Evaluate detection performance using delta-tau metric from ARAMIS paper.
    
    This function calculates the difference between the true and predicted anomaly
    detection points (tau). The delta-tau metric measures how close the predicted
    anomaly point is to the actual anomaly point in time.
    
    Parameters:
        score_tensor: Array of anomaly score sequences
        label_tensor: Array of ground truth label sequences
        threshold: Threshold value for anomaly detection
        min_consecutive: Minimum number of consecutive points above threshold to trigger detection
        filter_size: Size of median filter to apply to scores (None for no filtering)
        
    Returns:
        float: Mean delta-tau across all sequences (lower is better)
    """
    dtau_total=0.0
    cost_late, cost_early = 1, 1 # weights to penalize late vs early detections differently 
    for score, label_true in zip(score_tensor, label_tensor):
        min_len = min(len(score), len(label_true))

        # 1. Apply score transform + median filter (as in evaluation)
        score = prepare_scores(
            score,
            filter_size=filter_size,
            score_transform=score_transform,
            drift_window=drift_window,
        )

        # 2. Get Predictions
        label_pred = predict_anomalies(
            score,
            threshold,
            detection_mode=detection_mode,
            min_consecutive=min_consecutive,
        )
        
        # 3. Find first change points (Tau)
        # We slice by [:min_len] to ensure alignment
        tau_true = first_positive_index(label_true[:min_len])
        tau_pred = first_positive_index(label_pred[:min_len])
        
        has_anomaly = not np.isnan(tau_true)
        has_prediction = not np.isnan(tau_pred)
        
        # --- SCENARIOS ---
        # Case A: True Negative (Normal sample, correctly ignored)
        if not has_anomaly and not has_prediction:
            dtau_total += 0.0
            
        # Case B: False Positive (Normal sample, but we predicted an anomaly)
        elif not has_anomaly and has_prediction:
            dtau_total += cost_early * min_len

        # Case C: False Negative (Anomaly exists, but we missed it)
        elif has_anomaly and not has_prediction:
            dtau_total += cost_late * min_len
            
        # Case D: True Positive (Anomaly exists, and we predicted one)
        else:
            t_true = tau_true
            t_pred = tau_pred
            
            # Late detection
            if t_pred > t_true:
                dtau = cost_late * (t_pred - t_true)
            # Early detection (Premature alarm)
            else:
                dtau = cost_early * (t_true - t_pred)
                
            dtau_total += dtau

    # Return mean delta-tau across all sequences
    return dtau_total


def f1(
    score_tensor: np.ndarray,
    label_tensor: np.ndarray,
    tau_tensor: np.ndarray,
    threshold: float,
    filter_size: int = None,
    detection_mode: str = "persistent",
    min_consecutive: int = 1,
    score_transform: str = "raw",
    drift_window: int = None,
) -> float:
    """Raw point-wise F1, identical to the metric reported by evaluate_model.

    Persistent (ARAMIS): steps before the onset tau (and every step of a normal
    sequence) form the normal pool, steps from tau onward the anomalous pool.
    Pointwise (SWaT): the split follows the per-timestep label. The threshold is
    applied raw (no persistence). Returns the negative F1 for minimization.
    """
    norm_scores, anom_scores = [], []
    for score, label, tau in zip(score_tensor, label_tensor, tau_tensor):
        s = np.asarray(prepare_scores(
            score,
            filter_size=filter_size,
            score_transform=score_transform,
            drift_window=drift_window,
        )).reshape(-1)
        if detection_mode == "persistent":
            if tau is None or np.isnan(tau):
                norm_scores.append(s)
            else:
                t = int(tau)
                norm_scores.append(s[:t])
                anom_scores.append(s[t:])
        else:
            lab = np.asarray(label).reshape(-1)
            m = min(len(s), len(lab))
            s, lab = s[:m], lab[:m]
            norm_scores.append(s[lab == 0])
            anom_scores.append(s[lab == 1])

    norm = np.concatenate(norm_scores) if norm_scores else np.array([])
    anom = np.concatenate(anom_scores) if anom_scores else np.array([])
    tp = float(np.sum(anom > threshold))
    fp = float(np.sum(norm > threshold))
    fn = float(np.sum(anom <= threshold))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return -2 * precision * recall / (precision + recall)


def event_f1(
    score_tensor: np.ndarray,
    label_tensor: np.ndarray,
    threshold: float,
    filter_size: int = None,
    detection_mode: str = "persistent",
    min_consecutive: int = 1,
    score_transform: str = "raw",
    drift_window: int = None,
) -> float:
    tp_total, fp_total, fn_total = 0, 0, 0

    for score, label in zip(score_tensor, label_tensor):
        min_len = min(len(score), len(label))
        score = prepare_scores(
            score,
            filter_size=filter_size,
            score_transform=score_transform,
            drift_window=drift_window,
        )
        pred = predict_anomalies(
            score,
            threshold,
            detection_mode=detection_mode,
            min_consecutive=min_consecutive,
        )

        stats = compute_event_stats(
            np.asarray(label[:min_len]).reshape(-1),
            np.asarray(pred[:min_len]).reshape(-1),
        )
        tp_total += stats["TP"]
        fp_total += stats["FP"]
        fn_total += stats["FN"]

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return -score


def aramis_cost(
    score_tensor: np.ndarray,
    label_tensor: np.ndarray,
    tau_tensor: np.ndarray,
    threshold: float,
    filter_size: int = None,
    detection_mode: str = "persistent",
    min_consecutive: int = 1,
    score_transform: str = "raw",
    drift_window: int = None,
    T: float = 1000.0,
    a1: float = 10.0,
    a2: float = 13.0,
) -> float:
    """
    Asymmetric ARAMIS cost objective for threshold calibration.

    Replicates the evaluation-time cost: for each sequence the predicted onset
    tau_pred is the first detected change point and tau_true is the stored onset
    (NaN for normal sequences), which are scored by ``aramis_metric`` with the
    same asymmetric penalties used at evaluation (late detections and misses
    penalized more than early detections/false alarms). Lower is better.
    """
    tau_true_list, tau_pred_list = [], []
    for score, tau in zip(score_tensor, tau_tensor):
        score = prepare_scores(
            score,
            filter_size=filter_size,
            score_transform=score_transform,
            drift_window=drift_window,
        )
        pred = predict_anomalies(
            score,
            threshold,
            detection_mode=detection_mode,
            min_consecutive=min_consecutive,
        )
        tau_true_list.append(float(tau) if tau is not None and not np.isnan(tau) else np.nan)
        tau_pred_list.append(first_positive_index(pred))

    return aramis_metric(
        np.asarray(tau_true_list, dtype=float),
        np.asarray(tau_pred_list, dtype=float),
        T=T,
        a1=a1,
        a2=a2,
        k_false=7 * T,
        k_missed=10 * T,
    )


def select_threshold(
    df_valid: pd.DataFrame,
    dataset_name: str = None,
    n_calls: int = 10,  
    random_state: int = 0,
    verbose: bool = False,
    log_dir: str = None,
    logger: logging.Logger = None,
    detection_mode: str = "auto",
    objective_name: str = "auto",
    min_consecutive: int = 1,
    optimize_min_consecutive: bool = False,
    min_consecutive_bounds: tuple = (1, 50),
    score_transform: str = "auto",
    drift_window: int = None,
) -> Dict[str, float]:
    """Select optimal threshold parameters using Bayesian optimization."""
    n_calls = max(int(n_calls), 10)
    n_initial_points = min(10, max(1, n_calls - 1))
    
    # Extract anomaly scores and labels from DataFrame
    score_tensor = _to_sequence_list(df_valid['anom_score'])
    label_tensor = _to_sequence_list(df_valid['y'])
    # True onset per sequence (NaN for normal); present only for ARAMIS
    if 'tau' in df_valid.columns:
        tau_tensor = np.asarray(df_valid['tau'].values, dtype=float)
    else:
        tau_tensor = np.full(len(score_tensor), np.nan, dtype=float)

    resolved_mode = resolve_detection_mode(dataset_name, detection_mode)
    resolved_score_transform = resolve_score_transform(
        dataset_name,
        score_transform,
        detection_mode=resolved_mode,
    )
    # Paper (IV-C / V-A): the persistent-anomaly PdM setting (ARAMIS) calibrates
    # the threshold by minimizing the average time offset between true and
    # predicted anomalies; SWaT (pointwise) uses F1.
    resolved_objective = objective_name
    if objective_name == "auto":
        resolved_objective = "dtau" if resolved_mode == "persistent" else "f1"

    # Define search space for optimization
    max_score = max(float(np.max(seq)) for seq in score_tensor)
    
    space = [
        Real(0, 0.95*max_score, name='threshold'),
        Integer(1, 50, name='filter_size')
    ]
    if optimize_min_consecutive:
        low, high = min_consecutive_bounds
        space.append(Integer(int(low), int(high), name='min_consecutive'))
    
    def objective(params):
        """Objective function for Bayesian optimization."""
        threshold = params[0]
        filter_size = int(params[1])
        current_min_consecutive = int(params[2]) if optimize_min_consecutive else int(min_consecutive)

        if resolved_objective == "aramis_cost":
            return aramis_cost(
                score_tensor,
                label_tensor,
                tau_tensor,
                threshold,
                filter_size,
                detection_mode=resolved_mode,
                min_consecutive=current_min_consecutive,
                score_transform=resolved_score_transform,
                drift_window=drift_window,
            )
        if resolved_objective == "dtau":
            return dtau(
                score_tensor,
                label_tensor,
                threshold,
                filter_size,
                detection_mode=resolved_mode,
                min_consecutive=current_min_consecutive,
                score_transform=resolved_score_transform,
                drift_window=drift_window,
            )
        if resolved_objective == "f1":
            return f1(
                score_tensor,
                label_tensor,
                tau_tensor,
                threshold,
                filter_size,
                detection_mode=resolved_mode,
                min_consecutive=current_min_consecutive,
                score_transform=resolved_score_transform,
                drift_window=drift_window,
            )
        if resolved_objective == "event_f1":
            return event_f1(
                score_tensor,
                label_tensor,
                threshold,
                filter_size,
                detection_mode=resolved_mode,
                min_consecutive=current_min_consecutive,
                score_transform=resolved_score_transform,
                drift_window=drift_window,
            )
        raise ValueError(f"Unknown threshold objective: {resolved_objective}")
    
    # Perform Bayesian optimization
    result = gp_minimize(
        objective, 
        space, 
        n_calls=n_calls, 
        n_initial_points=n_initial_points,
        random_state=random_state, 
        verbose=verbose,
    )

    # Extract best parameters
    best_threshold = result.x[0]
    best_filter_size = result.x[1]
    best_min_consecutive = result.x[2] if optimize_min_consecutive else int(min_consecutive)
    best_obj = result.fun
    
    if logger is not None:
        logger.info("\nBayesian Optimization Results:")
        logger.info("Detection mode: %s", resolved_mode)
        logger.info("Score transform: %s", resolved_score_transform)
        logger.info("Objective: %s", resolved_objective)
        logger.info(f"Best threshold: {best_threshold:.4f}")
        logger.info(f"Best filter size: {best_filter_size}")
        logger.info(f"Best min_consecutive: {best_min_consecutive}")
        logger.info(f"Best objective score: {best_obj:.4f}")
    
    # Convert numpy types to Python native types for JSON serialization
    threshold_params = {
        "thr": float(best_threshold), 
        "filter_size": int(best_filter_size),
        "objective": float(best_obj),
        "objective_name": resolved_objective,
        "detection_mode": resolved_mode,
        "min_consecutive": int(best_min_consecutive),
        "score_transform": resolved_score_transform,
        "drift_window": None if drift_window is None else int(drift_window),
    }
    # Save results to log directory if specified as json
    if log_dir:
        with open(f"{log_dir}/threshold_params.json", 'w') as f:
            json.dump(threshold_params, f, indent=4)
            
    # Return optimal parameters
    return threshold_params
