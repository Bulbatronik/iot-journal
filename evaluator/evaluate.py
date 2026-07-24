"""
Evaluation logic for centralized models.

This module contains functions to load trained models, reconstruct sequences,
evaluate model performance, and visualize results. It implements the evaluation
pipeline for centralized anomaly detection models.
"""

from typing import Any, Dict

import pandas as pd
import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from datasets.aramis import AramisDataset
from datasets.postprocess import reconstruct_score_series, reconstruct_sequence, reconstruct_time_series
from datasets.preprocess import create_sequences
from evaluator.detection import (
    compute_event_stats,
    first_positive_index,
    prepare_scores,
    predict_anomalies,
    resolve_detection_mode,
    resolve_score_transform,
)

from .aramis_cost import aramis_metric

# Configure tqdm for pandas apply operations
tqdm.pandas(desc="Reconstructing sequences")


def _safe_binary_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if y_true.size == 0 or y_score.size == 0:
        return {"ROC_AUC": None, "PR_AUC_TRAP": None}

    unique = np.unique(y_true)
    if unique.size < 2:
        return {"ROC_AUC": None, "PR_AUC_TRAP": None}

    roc_auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc_trap = auc(recall[::-1], precision[::-1])
    return {
        "ROC_AUC": float(roc_auc),
        "PR_AUC_TRAP": float(pr_auc_trap),
    }


def _prefix_metric_names(metrics: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    return {f"{key}({suffix})": value for key, value in metrics.items()}

def evaluate_model(df_test: pd.DataFrame,
                   threshold_params: Dict[str, float],
                   dataset_name: str = None) -> Dict[str, Any]:
    
    """Evaluate model on test data with threshold parameters.
    
    Args:
        df_test: Test data with scores and labels
        threshold_params: Threshold parameters
        results: Training results
        
    Returns:
        Updated results with evaluation metrics
    """
    tau_true, tau_pred = [], []
    delta_early, delta_late = [], []
    scores_all_anom, scores_all_norm = np.array([]), np.array([])
    detection_mode = resolve_detection_mode(
        dataset_name,
        threshold_params.get("detection_mode", "auto"),
    )
    score_transform = resolve_score_transform(
        dataset_name,
        threshold_params.get("score_transform", "auto"),
        detection_mode=detection_mode,
    )
    
    for _, row in df_test.iterrows():
        # Extract scores, ground truth, and anomaly location
        score = row['anom_score']
        anom_true = row['y']
        tau = row['tau']
        
        tau_true.append(tau) # To compute the aramis cost 
        
        # Apply median filter to the anomaly scores
        score_filtered = prepare_scores(
            score,
            filter_size=threshold_params['filter_size'],
            score_transform=score_transform,
            drift_window=threshold_params.get('drift_window'),
        )
        
        
        # Old way
        if detection_mode == "persistent" and np.isnan(tau):
            scores_all_norm = np.concatenate((scores_all_norm, score_filtered.squeeze()))
        elif detection_mode == "persistent":
            scores_all_norm = np.concatenate((scores_all_norm, score_filtered.squeeze()[:int(tau)]))
            scores_all_anom = np.concatenate((scores_all_anom, score_filtered.squeeze()[int(tau):]))
        else:
            anom_true_np = np.asarray(anom_true).reshape(-1)
            min_len = min(len(score_filtered), len(anom_true_np))
            scores_all_norm = np.concatenate((scores_all_norm, score_filtered.squeeze()[:min_len][anom_true_np[:min_len] == 0]))
            scores_all_anom = np.concatenate((scores_all_anom, score_filtered.squeeze()[:min_len][anom_true_np[:min_len] == 1]))
            
        
        # Detect anomaly/changepoint
        anom_pred = predict_anomalies(
            scores=score_filtered,
            threshold=threshold_params['thr'],
            detection_mode=detection_mode,
            min_consecutive=threshold_params.get('min_consecutive', 1),
        )
        
        anom_true_np = np.asarray(anom_true)
        anom_pred_np = np.asarray(anom_pred)
        if len(anom_pred_np) < len(anom_true_np):
            anom_pred_np = np.pad(anom_pred_np, (0, len(anom_true_np) - len(anom_pred_np)), 'constant', constant_values=0)
        if len(anom_pred_np) > len(anom_true_np):
            anom_pred_np = anom_pred_np[:len(anom_true_np)]

        event_stats = compute_event_stats(anom_true_np, anom_pred_np)
        delta_early.extend(event_stats["delta_early"])
        delta_late.extend(event_stats["delta_late"])

        tau_pred.append(first_positive_index(anom_pred))
                
    avg_delta_early = np.mean(delta_early) if delta_early else 0
    avg_delta_late = np.mean(delta_late) if delta_late else 0

    TP_OLD = sum(scores_all_anom > threshold_params['thr'])
    FP_OLD = sum(scores_all_norm > threshold_params['thr'])
    FN_OLD = sum(scores_all_anom <= threshold_params['thr'])
    TN_OLD = sum(scores_all_norm <= threshold_params['thr'])

    precision_OLD = TP_OLD / (TP_OLD + FP_OLD) if (TP_OLD + FP_OLD) > 0 else 0
    recall_OLD = TP_OLD / (TP_OLD + FN_OLD) if (TP_OLD + FN_OLD) > 0 else 0
    f1_OLD = 2 * (precision_OLD * recall_OLD) / (precision_OLD + recall_OLD) if (precision_OLD + recall_OLD) > 0 else 0
    f2_OLD = 5 * (precision_OLD * recall_OLD) / (4*precision_OLD + recall_OLD) if (precision_OLD + recall_OLD) > 0 else 0

    cost = (
        aramis_metric(tau_true, tau_pred, T=1000, a1=10, a2=13, k_false=7*1000, k_missed=10*1000)
        if detection_mode == "persistent"
        else 0.0
    )

    # Threshold-independent AUCs over the pooled per-timestep scores (normal vs anomalous)
    auc_metrics = _safe_binary_auc_metrics(
        np.concatenate([np.zeros(len(scores_all_norm)), np.ones(len(scores_all_anom))]),
        np.concatenate([scores_all_norm, scores_all_anom]),
    )

    results = {
        'TP': int(TP_OLD),
        'FP': int(FP_OLD),
        'TN': int(TN_OLD),
        'FN': int(FN_OLD),
        'Precision': float(precision_OLD),
        'Recall': float(recall_OLD),
        'F1': float(f1_OLD),
        'F2': float(f2_OLD),

        'PR_AUC': auc_metrics['PR_AUC_TRAP'],
        'ROC_AUC': auc_metrics['ROC_AUC'],

        'd_TAU_early': float(avg_delta_early),
        'd_TAU_late': float(avg_delta_late),
        'd_TAU': float(np.mean(delta_early + delta_late)) if (delta_early or delta_late) else 0.0,
        'Cost': float(cost),

        'eval_mode_is_persistent': float(detection_mode == "persistent"),
        'score_transform_is_raw': float(score_transform == "raw"),
    }

    return results



def reconstruct_dataset(
    model: Any, 
    dataset: AramisDataset, 
    config: Dict[str, Any], 
    args: Any,
    subset: str,
    log_dir: str = None
    ) -> AramisDataset:
    """
    Reconstruct the sequences in the validation/test DataFrame using the trained model.
    
    Parameters:
        model: The trained model to use for reconstruction
        dataset: Object containing the validation and test datasets
        config: Configuration parameters for reconstruction
        args: Additional arguments including model_name and batch_size
        subset: Which subset to reconstruct ('valid' or 'test')
        log_dir: Directory to save logs and outputs (if any)
    
    Returns:
        dataset: Updated dataset object with reconstructed sequences
    """
    # Scale the 'sens' column in the specific subset
    dataset.scale_data(split=subset, column='sens')
    
    ds = dataset.__getattribute__(f"df_{subset}")
    # Reconstruct each sequence in the dataframe

    #reconstruction_results = ds.progress_apply(
    #    lambda row: reconstruct_sequence(
    #        model=model,
    #        seq=row,
    #        window_size=config["dataset"]["window_size"],
    #        stride=config["dataset"]["stride"],
    #        **config[args.model_name]["generation"]),
    #    axis=1, result_type='expand'
    #)
    # FASTER VERSION
    rows = ds.to_dict('records')
    window_size = config["dataset"]["window_size"]
    stride = config["dataset"]["stride"]
    device = next(model.parameters()).device
    seq_wind_list = []
    lengths = []
    for row in rows:
        seq_wind = create_sequences(row['sens'].reshape(1, *row['sens'].shape), window_size=window_size, stride=stride)
        seq_wind_list.append(seq_wind)
        lengths.append(len(seq_wind))
    big_input = torch.cat([torch.tensor(sw, dtype=torch.float32) for sw in seq_wind_list], dim=0).to(device)
    # Process in batches to avoid OOM
    batch_size = 2*1024  # Adjust based on memory
    big_reconstr_list = []
    big_anom_list = []
    pbar = tqdm(range(0, big_input.size(0), batch_size), desc="Reconstructing sequences", unit="batch")
    for i in range(0, big_input.size(0), batch_size):
        end_i = min(i + batch_size, big_input.size(0))
        batch_input = big_input[i:end_i]
        reconstr, anom = model.gen_seq(input_seq=batch_input, **config[args.model_name]["generation"])
        big_reconstr_list.append(reconstr.detach().cpu().numpy())
        big_anom_list.append(anom.detach().cpu().numpy())
        
        pbar.update(1)
    big_reconstr = np.concatenate(big_reconstr_list, axis=0)
    big_anom = np.concatenate(big_anom_list, axis=0)
    # Split back
    start = 0
    results = []
    filter_size = config[args.model_name]["generation"].get("filter_size")
    for i, length in enumerate(lengths):
        end = start + length
        reconstr_data = big_reconstr[start:end]
        anom_score = big_anom[start:end]
        predict_flat = reconstruct_time_series(reconstr_data, window_size, stride)
        anom_score_flat = reconstruct_score_series(anom_score, window_size, stride)
        if filter_size is not None:
            anom_score_flat = median_filter(anom_score_flat, size=filter_size)
        results.append(pd.Series({'sens_rec': predict_flat, 'anom_score': anom_score_flat}))
        start = end
    reconstruction_results = pd.DataFrame(results)
    reconstruction_results.index = ds.index
    
    # Add the reconstruction results to the dataframe using .loc to avoid SettingWithCopyWarning
    dataset.__getattribute__(f"df_{subset}").loc[:, ['sens_rec', 'anom_score']] = reconstruction_results
    
    if log_dir is not None:
        # Save processed data to CSV files
        dataset.__getattribute__(f"df_{subset}").to_csv(f"{log_dir}/df_{subset}.csv", index=False)
    
    return dataset
