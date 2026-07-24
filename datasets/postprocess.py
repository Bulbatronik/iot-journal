import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from .preprocess import create_sequences


# def reconstruct_sequence(model, seq, window_size, stride, filter_size=None, **kwargs):
#     """
#     Reconstructs a time series from overlapping windows using a trained model.

#     Args:
#         model (torch.nn.Module): The trained PyTorch model used for reconstruction.
#         seq (dict): A dictionary containing the time series data, with the key 'sens' 
#                     representing the sensor data as a numpy array.
#         gen_seq (callable): A function that generates sequences from the model.
#         window_size (int): The size of each window used for splitting the time series.
#         stride (int): The step size between consecutive windows.
#         filter_size (int, optional): The size of the median filter to apply to the anomaly score.
#         **kwargs: Additional keyword arguments to pass to the `gen_seq` function.

#     Returns:
#         numpy.ndarray: The reconstructed time series as a numpy array.
#     """
#     # Convert the time series to a set of windows
    
#     seq_wind = create_sequences(seq['sens'].reshape(1, *seq['sens'].shape), window_size=window_size, stride=stride)
#     #print(seq_wind.shape)

#     # Convert the windows to PyTorch tensors
#     input_seq = torch.tensor(seq_wind, dtype=torch.float32).to(next(model.parameters()).device)

#     reconstr_data, anom_score = model.gen_seq(input_seq=input_seq, **kwargs)
#     reconstr_data = reconstr_data.detach().cpu().numpy()
#     #print(reconstr_data.shape)
#     anom_score = anom_score.detach().cpu().numpy()

#     # Convert the predictions for each window to a complete time series
#     predict_flat = reconstruct_time_series(reconstr_data, window_size, stride)
#     #print(predict_flat.shape)
#     anom_score_flat = reconstruct_time_series(anom_score, window_size, stride)
    
#     # Remove spikes frin the anomaly score
#     if filter_size is not None:
#         anom_score_flat = median_filter(anom_score_flat, size=filter_size)
    
#     # T = seq['sens'].shape[0]
#     # print("T:", T)
#     # print("remainder:", (T - window_size) % stride)
#     # print("n_windows:", seq_wind.shape[0])
#     # print("reconstructed_len:", predict_flat.shape[0])
#     # print(seq['sens'].shape)
#     #T_min = min(seq['sens'].shape[0], predict_flat.shape[0])
    
#     return pd.Series({'sens_rec': predict_flat, 'anom_score': anom_score_flat})


# def reconstruct_time_series(windowed_data, window_size, stride):
#     n_windows, _, n_features = windowed_data.shape
#     time_length = (n_windows - 1) * stride + window_size

#     out = np.full((time_length, n_features), np.nan, dtype=windowed_data.dtype)
#     for i, window in enumerate(windowed_data):
#         start = i * stride
#         end = start + window_size

#         current = out[start:end]

#         abs_window = np.abs(window)
#         abs_current = np.abs(current)

#         # Match your original behavior:
#         # - take larger absolute value
#         # - if equal abs, take the smaller numeric value
#         take_window = (
#             np.isnan(current) |
#             (abs_window > abs_current) |
#             ((abs_window == abs_current) & (window < current))
#         )

#         out[start:end] = np.where(take_window, window, current)

#     return out


def _merge_windows_inplace(out, windowed_data, start_window_idx, window_size, stride):
    """
    Merge a batch of reconstructed windows into the output time series,
    preserving your original 'largest absolute value wins' rule.
    """
    for i, window in enumerate(windowed_data):
        start = (start_window_idx + i) * stride
        end = start + window_size

        current = out[start:end]

        abs_window = np.abs(window)
        abs_current = np.abs(current)

        take_window = (
            np.isnan(current) |
            (abs_window > abs_current) |
            ((abs_window == abs_current) & (window < current))
        )

        out[start:end] = np.where(take_window, window, current)


def _average_merge_windows(windowed_data, window_size, stride):
    """
    Merge overlapping windows by averaging them pointwise.

    This is appropriate for anomaly scores, where taking the largest absolute
    value across overlapping windows artificially inflates long-horizon trends.
    """
    windowed_data = np.asarray(windowed_data)
    if windowed_data.ndim != 3:
        raise ValueError(
            f"windowed_data must have shape (n_windows, window_size, n_features), got {windowed_data.shape}"
        )

    n_windows, _, n_features = windowed_data.shape
    time_length = (n_windows - 1) * stride + window_size

    sums = np.zeros((time_length, n_features), dtype=np.float64)
    counts = np.zeros((time_length, n_features), dtype=np.float64)

    for i, window in enumerate(windowed_data):
        start = i * stride
        end = start + window_size
        sums[start:end] += window
        counts[start:end] += 1.0

    counts[counts == 0] = 1.0
    merged = sums / counts
    return merged.astype(windowed_data.dtype, copy=False)


def _iter_window_batches(x, window_size, stride, batch_size):
    """
    Yield batches of windows without building the full window tensor in memory.
    x: array of shape (T, F)
    """
    T = len(x)
    starts = range(0, T - window_size + 1, stride)

    batch = []
    batch_start_window_idx = None
    window_idx = 0

    for s in starts:
        if batch_start_window_idx is None:
            batch_start_window_idx = window_idx

        batch.append(x[s:s + window_size])

        if len(batch) == batch_size:
            yield batch_start_window_idx, np.stack(batch, axis=0)
            batch = []
            batch_start_window_idx = None

        window_idx += 1

    if batch:
        yield batch_start_window_idx, np.stack(batch, axis=0)


def reconstruct_sequence(model, seq, window_size, stride, filter_size=None, batch_size=256, **kwargs):
    """
    Memory-safe reconstruction of a long time series.
    """
    sens = np.asarray(seq['sens'], dtype=np.float32)

    if sens.ndim != 2:
        raise ValueError(f"seq['sens'] must have shape (T, F), got {sens.shape}")

    T, n_features = sens.shape
    if T < window_size:
        raise ValueError(f"Sequence length {T} is smaller than window_size={window_size}")

    n_windows = 1 + (T - window_size) // stride
    time_length = (n_windows - 1) * stride + window_size

    sens_rec = np.full((time_length, n_features), np.nan, dtype=np.float32)
    anom_score_flat = np.full((time_length, 1), np.nan, dtype=np.float32)

    device = next(model.parameters()).device
    model.eval()

    with torch.inference_mode():
        for start_window_idx, batch_np in _iter_window_batches(
            sens, window_size=window_size, stride=stride, batch_size=batch_size
        ):
            batch_t = torch.from_numpy(batch_np).to(device)

            recon_batch, score_batch = model.gen_seq(input_seq=batch_t, **kwargs)

            recon_batch = recon_batch.detach().cpu().numpy()
            score_batch = score_batch.detach().cpu().numpy()

            _merge_windows_inplace(
                sens_rec, recon_batch, start_window_idx, window_size, stride
            )
            _merge_windows_inplace(
                anom_score_flat, score_batch, start_window_idx, window_size, stride
            )

    if filter_size is not None:
        anom_score_flat = median_filter(anom_score_flat, size=filter_size)

    return pd.Series({
        'sens_rec': sens_rec,
        'anom_score': anom_score_flat
    })


def reconstruct_time_series(windowed_data, window_size, stride):
    """
    Reconstruct a full time series from overlapping windows using the same
    merge rule as `_merge_windows_inplace`.
    """
    windowed_data = np.asarray(windowed_data)

    if windowed_data.ndim != 3:
        raise ValueError(
            f"windowed_data must have shape (n_windows, window_size, n_features), got {windowed_data.shape}"
        )

    n_windows, _, n_features = windowed_data.shape
    time_length = (n_windows - 1) * stride + window_size
    out = np.full((time_length, n_features), np.nan, dtype=windowed_data.dtype)

    _merge_windows_inplace(out, windowed_data, 0, window_size, stride)
    return out


def reconstruct_score_series(windowed_data, window_size, stride):
    """
    Reconstruct a full anomaly-score series from overlapping score windows by
    averaging overlaps.
    """
    return _average_merge_windows(windowed_data, window_size, stride)


# def reconstruct_time_series(windowed_data, window_size, stride):
#     """
#     Reconstructs a complete time series from overlapping windows.

#     Args:
#         windowed_data (numpy.ndarray): The windowed data of shape (n_windows, window_size, n_features).
#         window_size (int): The size of each window in the input sequence.
#         stride (int): The step size between consecutive windows.

#     Returns:
#         numpy.ndarray: The reconstructed time series of shape (time_length, n_features), where 
#                        time_length is determined by the number of windows, window size, and stride.
#     """
#     result = np.nan * np.zeros(shape=((windowed_data.shape[0] - 1) * stride + window_size, 
#                                       windowed_data.shape[0], 
#                                       windowed_data.shape[2]),dtype="float16")  # (time_length, num_windows, num_features)
    
#     for index, window in enumerate(windowed_data):
#         start_idx = index * stride
#         end_idx = start_idx + window_size
        
#         result[start_idx:end_idx, index, :] = window
    
#     # Perform max pooling and min pooling to handle overlapping regions
#     reconstructed_data_max = np.nanmax(result, axis=1)
#     reconstructed_data_min = np.nanmin(result, axis=1)
    
#     # Combine max and min pooling results based on absolute values
#     reconstructed_data = np.where(np.abs(reconstructed_data_max) > np.abs(reconstructed_data_min), 
#                                    reconstructed_data_max, 
#                                    reconstructed_data_min)
#     return reconstructed_data  # Note: The last window may not be fully reconstructed
