import random
import gc
import torch
import numpy as np
import json

def set_seed(seed):
    """Fix random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_cuda():
    """Clean CUDA cache"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.ipc_collect()    


def get_device():
    """Get the current device"""
    # List GPU names
    gpu_names = []
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(i))
        print(f'GPUs available: {gpu_names}')
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU.')
    print(f'Using device: {device}')
    return device



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)