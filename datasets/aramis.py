import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Subset
from .preprocess import create_sequences

# Get or create logger
logger = logging.getLogger(__name__)


class AramisDataset:
    """
    A class to handle the processing and loading of the Aramis dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the Aramis dataset CSV file
    num_features : int
        Number of sensor features in the dataset
    delta_tau : int, optional (default=50)
        Margin of samples to truncate sensitivity values for abnormal components
    window_size : int, optional (default=20)
        Size of the sliding window for sequence creation
    stride : int, optional (default=5)
        Stride length for sliding window
    batch_size : int, optional (default=128)
        Batch size for DataLoader
    device : str, optional (default='cuda' if available else 'cpu')
        Device to load the tensors on
    random_state : int, optional (default=42)
        Random seed for train-test splitting
    """
    
    def __init__(self, data_path, delta_tau=50, window_size=20,
                 stride=5, batch_size=128, num_clients=None, device=None, seed=42,
                 train_count=30, val_count=65, test_count=65, ref_clients=5):
        self.data_path = data_path
        self.num_features = None
        self.delta_tau = delta_tau
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.num_clients = num_clients if num_clients else 1
        # Per-client trajectory budgets (paper V-A): 30 train / 65 val / 65 test.
        self.train_count = train_count
        self.val_count = val_count
        self.test_count = test_count
        self.ref_clients = ref_clients

        # Load and process data
        self.df = self._load_and_preprocess_data()
        self._compute_anomaly_labels()

        # Build per-client train/val/test splits (one partition per client)
        self.datasets_client = self._build_client_splits()
        self.df_train = self._combine_client_split(index=0)
        self.df_valid = self._combine_client_split(index=1)
        self.df_test = self._combine_client_split(index=2)
        self._truncate_training_data() # If train data has abnormal "tails", they will be removed

    def _load_and_preprocess_data(self):
        """Load and preprocess the raw data into a compact format."""
        df_raw = pd.read_csv(self.data_path)
        df_raw.columns = df_raw.columns.map(lambda x: x.lower())
        self.num_features = len([c for c in df_raw.columns if c.startswith('s_')])

        df = pd.DataFrame()
        gk = df_raw.groupby(["sys", "comp"])
        
        for _, group in gk:
            aggregate_values = {
                'sys': group['sys'].iloc[0],
                'comp': group['comp'].iloc[0],
                'sens': [group[[f's_{i}' for i in range(self.num_features)]].values],
                'y': [group['y'].values],
                'tau': group['tau'].iloc[0]
            }
            df = pd.concat([df, pd.DataFrame(aggregate_values)], ignore_index=True)
        
        return df


    def _compute_anomaly_labels(self):
        """Compute anomaly labels for components and systems."""
        # Component-level anomaly
        self.df['abn_comp'] = self.df.tau.apply(lambda x: not np.isnan(x))
        
        # System-level anomaly
        self.df['abn_sys'] = (self.df.groupby('sys')['abn_comp']
                             .transform('sum')
                             .apply(lambda x: x > 0))
        
        # Store normal and abnormal IDs
        self.normal_systems = shuffle(self.df[~self.df.abn_sys].sys.unique(),
                                      random_state=self.seed) 
        self.abnormal_systems = shuffle(self.df[self.df.abn_sys].sys.unique(),
                                        random_state=self.seed)
        
        self.normal_components = shuffle([(int(s), int(c)) for s, c in 
                                self.df[~self.df.abn_comp][['sys', 'comp']].values],
                                         random_state=self.seed)
        self.abnormal_components = shuffle([(int(s), int(c)) for s, c in
                                  self.df[self.df.abn_comp][['sys', 'comp']].values],
                                           random_state=self.seed)


    def _assign_systems_to_groups(self, n_groups):
        """Assign whole systems to n_groups client groups (non-IID).

        Systems (and their degradation processes) are kept together on a single
        client; assignment is greedy by the total number of components per
        system so that each client's total trajectory budget (train + val + test)
        stays balanced, which is what the fixed per-client counts require.
        """
        total_per_sys = self.df.groupby('sys').size()
        systems = shuffle(list(self.df['sys'].unique()), random_state=self.seed)
        systems = sorted(systems, key=lambda s: int(total_per_sys.get(s, 0)), reverse=True)

        group_load = [0] * n_groups
        groups = [[] for _ in range(n_groups)]
        for s in systems:
            g = min(range(n_groups), key=lambda k: group_load[k])
            groups[g].append(s)
            group_load[g] += int(total_per_sys.get(s, 0))
        return groups

    def _split_group(self, sys_group):
        """Split one client's systems into train/val/test frames (paper V-A counts).

        Training uses ``train_count`` normal-condition trajectories; validation
        and test use ``val_count`` / ``test_count`` trajectories mixing normal and
        abnormal components (both needed to calibrate/score the change point).
        """
        in_group = self.df[self.df['sys'].isin(sys_group)]
        normal = shuffle(in_group[~in_group['abn_comp']], random_state=self.seed).reset_index(drop=True)
        abnormal = shuffle(in_group[in_group['abn_comp']], random_state=self.seed).reset_index(drop=True)

        if len(normal) < self.train_count:
            raise ValueError(
                f"A client has {len(normal)} normal trajectories but train_count="
                f"{self.train_count}; reduce counts or num_clients."
            )
        df_train = normal.iloc[:self.train_count].reset_index(drop=True)

        # Remaining normal + all abnormal form the calibration/test pool
        pool = pd.concat([normal.iloc[self.train_count:], abnormal], ignore_index=True)
        pool = shuffle(pool, random_state=self.seed).reset_index(drop=True)
        need = self.val_count + self.test_count
        if len(pool) < need:
            raise ValueError(
                f"A client has {len(pool)} calibration/test trajectories but needs "
                f"val_count+test_count={need}; reduce counts or num_clients."
            )
        df_valid = pool.iloc[:self.val_count].reset_index(drop=True)
        df_test = pool.iloc[self.val_count:need].reset_index(drop=True)
        return df_train, df_valid, df_test

    def _build_client_splits(self):
        """Build per-client (train, val, test) frames, one partition per client."""
        n_groups = self.num_clients if self.num_clients > 1 else self.ref_clients
        groups = self._assign_systems_to_groups(n_groups)
        ref_splits = [self._split_group(sys_group) for sys_group in groups]

        if self.num_clients == 1:
            # Centralized baseline: a single node pooling all clients' data
            pooled = tuple(
                pd.concat([s[i] for s in ref_splits], ignore_index=True)
                for i in range(3)
            )
            return [pooled]
        return ref_splits

    def _combine_client_split(self, index):
        """Concatenate the given split (0=train, 1=valid, 2=test) across clients."""
        frames = []
        for client_id, splits in enumerate(self.datasets_client):
            df_client = splits[index].copy()
            df_client.insert(0, "client", client_id)
            frames.append(df_client)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


    def _truncate_training_data(self):
        """Truncate training data sensitivity values."""
        
        def truncate_sens(x, delta_tau):
            """Truncate sensitivity values for abnormal components."""
            if np.isnan(x['tau']):
                return x['sens']
            else:
                return x['sens'][:int(x['tau'] - delta_tau)]
            
        #self.df_train['sens'] = self.df_train.apply(self._truncate_sens, axis=1)
        self.df_train.loc[:, 'sens'] = self.df_train.apply(truncate_sens, args=(self.delta_tau,), axis=1)

    
    def get_data_loader(self, split='train', column='sens'):
        """Return Train/Test DataLoader."""
        
        split = split.lower()
        if split not in ['train', 'valid', 'test']:#, 'all']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")
        
        df_split = self.__getattribute__(f'df_{split}')
        
        logger.info(f'Number of normal components: {len(df_split[df_split.abn_comp == False])}')
        logger.info(f'Number of abnormal components: {len(df_split[df_split.abn_comp == True])}')
        
        dataset = df_split.loc[:, column]
        X_train = create_sequences(dataset.values, window_size=self.window_size, stride=self.stride)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
    
        train_dataset = TensorDataset(X_train)
        return DataLoader(train_dataset, 
                         batch_size=self.batch_size, 
                         shuffle=True)

    
    def get_data_clients(self, num_clients=1, split='train', column='sens', stratify_col=None, return_loaders=True):
        """Return per-client data for the requested split.

        Each client's train/val/test frames are the per-partition splits built
        in ``_build_client_splits`` (paper V-A). With ``return_loaders=True`` a
        list of windowed DataLoaders is returned; otherwise the per-client
        DataFrames are returned (used for client-specific reconstruction).
        """
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")

        index = {'train': 0, 'valid': 1, 'test': 2}[split]
        df_clients = [self.datasets_client[c][index] for c in range(self.num_clients)]

        if not return_loaders:
            return df_clients

        client_loaders = []
        for i, df_client in enumerate(df_clients):
            X_client = create_sequences(
                df_client[column].values,
                window_size=self.window_size,
                stride=self.stride,
            )
            if len(X_client) == 0:
                raise ValueError(f"Client {i} produced no windows for split '{split}'.")
            X_client = torch.tensor(X_client, dtype=torch.float32).to(self.device)
            client_loaders.append(
                DataLoader(TensorDataset(X_client),
                           batch_size=self.batch_size,
                           shuffle=(split == 'train'))
            )
        return client_loaders


    def get_dataset_info(self):
        """Return basic information about the dataset."""
        return {
            'num_systems': self.df.sys.nunique(),
            'num_components': self.df.comp.nunique(),
            'num_features': self.num_features,
            
            'num_normal_systems': len(self.normal_systems),
            'normal_systems_ids': self.normal_systems,
            
            'num_abnormal_systems': len(self.abnormal_systems),
            'abnormal_systems_ids': self.abnormal_systems,
            
            'num_normal_components': len(self.normal_components),
            'normal_components_ids': self.normal_components,
            
            'num_abnormal_components': len(self.abnormal_components),
            'abnormal_components_ids': self.abnormal_components,
            
            'train_components': len(self.comp_train),
            'valid_components': len(self.comp_valid),
            'test_components': len(self.comp_test),
            #'x_train_shape': self.train_loader.dataset.tensors[0].shape
        }
