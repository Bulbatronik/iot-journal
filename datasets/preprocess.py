import numpy as np


def create_sequences(ts, y=None, window_size=50, stride=1):
    """
    Args:
        ts : numpy array of shape (n_samples, n_features)
        y: binary array of shape (n_samples,) or None [1 - anom, 0 - norm]
        window_size: window size of the input sequence
        stride : stride for data extraction
        shuffle : Flag to shuffle the data

    Returns:
        dataset: numpy array of shape (n_windows, window_size, n_features)
    """

    dataset = []
    if y is not None:
        target = []
    
    for x in ts:
        idx = 0
        while idx+window_size <= len(x):
            dataset.append(x[idx:idx+window_size])
            
            if y is not None:
                target.append(y[idx:idx+window_size])
            
            idx += stride
    
    if y is None:
        return np.array(dataset)
    else:
        return np.array(dataset), np.array(target)