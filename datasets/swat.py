import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


def create_sequences(values, window_size=20, stride=5):
    """
    Create sliding-window sequences from a 2D array of shape [time, features].
    Returns an array of shape [num_windows, window_size, features].
    """
    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError("values must be a 2D array of shape [time, features]")

    n_rows = len(values)
    if n_rows < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32)

    windows = [
        values[start:start + window_size]
        for start in range(0, n_rows - window_size + 1, stride)
    ]
    return np.stack(windows).astype(np.float32)


def _to_attack_binary(series):
    s = series.astype(str).str.strip().str.lower()

    if s.str.fullmatch(r"[01]").all():
        return s.astype(int)

    normal_tokens = {"normal", "0", "false", "benign"}
    return (~s.isin(normal_tokens)).astype(int)


def _contiguous_ranges(n, k):
    bounds = np.linspace(0, n, k + 1, dtype=int)
    return [(bounds[i], bounds[i + 1]) for i in range(k)]


class SWaTDataset:
    """
    ARAMIS-style utility for loading SWaT from a merged CSV and partitioning it
    chronologically for federated learning.

    Split strategy:
      - [start, cutoff)  -> train pool
      - [cutoff, end]    -> later pool
      - train pool       -> contiguous split among clients
      - later pool       -> contiguous split among clients
      - each client's later split is divided chronologically into valid/test

    Parameters
    ----------
    data_path : str
        Path to merged SWaT CSV.
    cutoff_timestamp : str or pandas.Timestamp
        Boundary between early normal interval and later interval.
    timestamp_col : str, default=" Timestamp"
        Timestamp column in the CSV.
    label_col : str, default="Normal/Attack"
        Label column in the CSV.
    feature_cols : list[str] or None
        Columns to use as model features. If None, uses every column except
        timestamp, label, and the derived 'attack' column.
    window_size : int, default=20
        Sequence length.
    stride : int, default=5
        Sliding-window stride.
    batch_size : int, default=128
        Batch size for DataLoaders.
    num_clients : int, default=5
        Number of federated clients.
    device : str or None
        Torch device. If None, auto-detects.
    seed : int, default=42
        Random seed.
    drop_attacks_from_train : bool, default=True
        If True, removes attack-labelled rows from the pre-cutoff train pool.
    scale : bool, default=True
        If True, fits a global scaler on pooled train rows and scales all splits.
    """

    def __init__(self, data_path, window_size=20, stride=5, 
                 batch_size=128, num_clients=5, device=None, seed=42, **kwargs):
        self.data_path = data_path
        self.num_features = None
        # self.delta_tau = delta_tau
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        self.cutoff_timestamp = pd.to_datetime("2015-12-28 07:17:00")
        self.timestamp_col = "Timestamp"
        self.label_col = "Normal/Attack"
        self.feature_cols = None
        self.drop_attacks_from_train = True
        self.scale_continuous = kwargs.get("scale_continuous", True)
        self.continuous_scaler_name = kwargs.get("continuous_scaler", "standard").lower()
        self.clip_continuous = kwargs.get("clip_continuous", False)
        self.binary_feature_cols = []
        self.continuous_feature_cols = []
        self.scalers = []
        self.scaler = None

        self.num_clients = num_clients
        self.info = None
        self.datasets_client = self._load_and_preprocess_data()
        self.df_train = self._combine_client_split(index=0)
        self.df_valid = self._combine_client_split(index=1)
        self.df_test = self._combine_client_split(index=2)
        
        
        
    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)

        # Remove " " from each column
        df.columns = df.columns.str.strip()
        #print(df.isnull().sum())
        self.feature_cols = [
            c for c in df.columns if c not in {self.timestamp_col, self.label_col}
        ]
        self.num_features = len(self.feature_cols)
        
        print("Converting data type")
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
   
        
        print("Cleaning timestamp")
        # Parse + sort by time
        df[self.timestamp_col] = pd.to_datetime(
                        df[self.timestamp_col].astype(str).str.strip(),
                        dayfirst=True,
                        format="mixed",
                        errors="coerce",
        )
        df = df.dropna(subset=[self.timestamp_col]).sort_values(self.timestamp_col).reset_index(drop=True)
  
        #df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])#, errors="coerce")
        #df = df.dropna(subset=[self.timestamp_col]).sort_values(self.timestamp_col).reset_index(drop=True)
        # Binary label
        df["y"] = _to_attack_binary(df[self.label_col])
        # Drop label column
        df = df.drop(columns=[self.label_col])
        
        print("Filling NaN values")
        #print(df.isnull().sum())
        # Binary actuators (fill with 0 for OFF state)
        binary_actuators = ['MV101', 'MV201', 'P201', 'P202', 'P204', 'MV303']
        for col in binary_actuators:
            matching_cols = [c for c in self.feature_cols if c.strip() == col]
            if matching_cols:
                actual_col = matching_cols[0]
                df[actual_col] = df[actual_col].fillna(0)
                df[actual_col] = df[actual_col].astype(int)
        
        # Critical continuous sensors (interpolate)
        critical_sensors = ['LIT101', 'AIT201', 'AIT202', 'FIT401', 'PIT501']
        for col in critical_sensors:
            matching_cols = [c for c in self.feature_cols if c.strip() == col]
            if matching_cols:
                actual_col = matching_cols[0]
                df[actual_col] = df[actual_col].interpolate(method='linear', limit_direction='both') #Linear interpolation fills missing values based on surrounding values.
        
        # Check if any nans are present
        assert df.isnull().sum().sum() == 0, f"Data contains NaNs, \n {df.isnull().sum()}"
        
        print("Splitting into train/valid/test")
        # Interval split
        self.cutoff_timestamp_start = pd.to_datetime("2015-12-22 19:30:00") # Remove some initial noise
        train_pool = df[(df[self.timestamp_col] < self.cutoff_timestamp) & (df[self.timestamp_col] >= self.cutoff_timestamp_start)].copy().reset_index(drop=True)
        later_pool = df[df[self.timestamp_col] >= self.cutoff_timestamp].copy().reset_index(drop=True)

        if self.drop_attacks_from_train:
            train_pool = train_pool[train_pool["y"] == 0].copy().reset_index(drop=True)

        train_ranges = _contiguous_ranges(len(train_pool), self.num_clients)
        later_ranges = _contiguous_ranges(len(later_pool), self.num_clients)

        summary = []
        datasets_client = []
        print("Splitting among clients")
        for client_id in range(self.num_clients):
            #client_dir = out_dir / f"client_{client_id + 1}"
            #client_dir.mkdir(parents=True, exist_ok=True)

            # Train chunk from early interval
            s_tr, e_tr = train_ranges[client_id]
            train_df = train_pool.iloc[s_tr:e_tr].reset_index(drop=True)

            # Later chunk from post-cutoff interval
            s_l, e_l = later_ranges[client_id]
            later_df = later_pool.iloc[s_l:e_l].reset_index(drop=True)

            # Split later chunk into val/test halves (chronological)
            mid = len(later_df) // 2
            val_df = later_df.iloc[:mid].reset_index(drop=True)
            test_df = later_df.iloc[mid:].reset_index(drop=True)

            summary.append(
            {
                "client": client_id,
                "train_rows": len(train_df),
                "train_anom_rows": int(train_df["y"].sum()),
                "val_rows": len(val_df),
                "val_anom_rows": int(val_df["y"].sum()),
                "test_rows": len(test_df),
                "test_anom_rows": int(test_df["y"].sum()),
                "train_start": train_df[self.timestamp_col].min() if len(train_df) else None,
                "train_end": train_df[self.timestamp_col].max() if len(train_df) else None,
                "val_start": val_df[self.timestamp_col].min() if len(val_df) else None,
                "val_end": val_df[self.timestamp_col].max() if len(val_df) else None,
                "test_start": test_df[self.timestamp_col].min() if len(test_df) else None,
                "test_end": test_df[self.timestamp_col].max() if len(test_df) else None,
            }
            )
            
            # Convert to aramis format (only one row)
            train_df = pd.DataFrame({'sens': [train_df[[col for col in self.feature_cols]].values],
                                    'y': [train_df['y'].values],
                                     'tau': np.nan})
            val_df = pd.DataFrame({'sens': [val_df[[col for col in self.feature_cols]].values],
                                'y': [val_df['y'].values]})
            test_df = pd.DataFrame({'sens': [test_df[[col for col in self.feature_cols]].values],
                                'y': [test_df['y'].values],
                                'tau': np.nan})

            self._set_feature_groups(train_df.iloc[0]['sens'])

            scaler = self._make_continuous_scaler()
            train_sens = train_df.iloc[0]['sens']
            val_sens = val_df.iloc[0]['sens']
            test_sens = test_df.iloc[0]['sens']

            if len(train_sens) == 0:
                raise ValueError(f"Client {client_id} has empty train split.")

            if len(val_sens) > 0:
                pass
            else:
                raise ValueError(f"Client {client_id} has empty validation split.")

            if len(test_sens) > 0:
                pass
            else:
                raise ValueError(f"Client {client_id} has empty test split.")

            train_scaled, val_scaled, test_scaled, client_scaler = self._scale_client_splits(
                train_sens=train_sens,
                val_sens=val_sens,
                test_sens=test_sens,
                scaler=scaler,
            )

            train_df.at[0, 'sens'] = train_scaled
            val_df.at[0, 'sens'] = val_scaled
            test_df.at[0, 'sens'] = test_scaled

            self.scalers.append(client_scaler)
            if client_id == 0:
                self.scaler = client_scaler

            datasets_client.append((train_df, val_df, test_df))
                
        self.info = summary
        print("Dataset is preprocessed")
        return datasets_client

    def _make_continuous_scaler(self):
        if not self.scale_continuous or not self.continuous_feature_cols:
            return None
        if self.continuous_scaler_name == "minmax":
            return MinMaxScaler()
        if self.continuous_scaler_name == "standard":
            return StandardScaler()
        raise ValueError("continuous_scaler must be either 'standard' or 'minmax'")

    def _set_feature_groups(self, train_sens):
        if self.binary_feature_cols or self.continuous_feature_cols:
            return

        binary_cols = []
        continuous_cols = []

        for i, col in enumerate(self.feature_cols):
            values = train_sens[:, i]
            values = values[~np.isnan(values)]
            unique = np.unique(values)
            is_integer_valued = np.allclose(unique, np.round(unique))

            if is_integer_valued and len(unique) <= 3:
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        self.binary_feature_cols = binary_cols
        self.continuous_feature_cols = continuous_cols

    def _transform_discrete(self, values, train_values):
        values = np.asarray(values, dtype=np.float32).copy()
        train_values = np.asarray(train_values, dtype=np.float32)

        for col in self.binary_feature_cols:
            idx = self.feature_cols.index(col)
            unique = np.unique(train_values[:, idx][~np.isnan(train_values[:, idx])])
            if len(unique) == 2:
                low, high = np.min(unique), np.max(unique)
                denom = high - low
                if denom != 0:
                    values[:, idx] = (values[:, idx] - low) / denom
            elif len(unique) == 1:
                values[:, idx] = 0.0

        return values

    def _transform_continuous(self, values, scaler):
        values = np.asarray(values, dtype=np.float32).copy()
        if scaler is None or not self.continuous_feature_cols:
            return values

        idxs = [self.feature_cols.index(col) for col in self.continuous_feature_cols]
        values_cont = values[:, idxs]
        values_cont = scaler.transform(values_cont)

        if self.clip_continuous and self.continuous_scaler_name == "minmax":
            values_cont = np.clip(values_cont, 0.0, 1.0)

        values[:, idxs] = values_cont
        return values

    def _scale_client_splits(self, train_sens, val_sens, test_sens, scaler):
        train_sens = np.asarray(train_sens, dtype=np.float32)
        val_sens = np.asarray(val_sens, dtype=np.float32)
        test_sens = np.asarray(test_sens, dtype=np.float32)

        continuous_scaler = scaler
        if continuous_scaler is not None and self.continuous_feature_cols:
            idxs = [self.feature_cols.index(col) for col in self.continuous_feature_cols]
            continuous_scaler.fit(train_sens[:, idxs])

        train_scaled = self._transform_discrete(train_sens, train_sens)
        val_scaled = self._transform_discrete(val_sens, train_sens)
        test_scaled = self._transform_discrete(test_sens, train_sens)

        train_scaled = self._transform_continuous(train_scaled, continuous_scaler)
        val_scaled = self._transform_continuous(val_scaled, continuous_scaler)
        test_scaled = self._transform_continuous(test_scaled, continuous_scaler)

        client_scaler = {
            "continuous_scaler": continuous_scaler,
            "continuous_feature_cols": list(self.continuous_feature_cols),
            "binary_feature_cols": list(self.binary_feature_cols),
        }
        return train_scaled, val_scaled, test_scaled, client_scaler

    def _combine_client_split(self, index):
        frames = []
        for client_id, splits in enumerate(self.datasets_client):
            df_client = splits[index].copy()
            df_client.insert(0, "client", client_id)
            frames.append(df_client)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _windows_from_df(self, df_split, column='sens'):
        windows = []
        for values in df_split[column].values:
            seq_windows = create_sequences(values, window_size=self.window_size, stride=self.stride)
            if len(seq_windows):
                windows.append(seq_windows)

        if not windows:
            return np.empty((0, self.window_size, self.num_features), dtype=np.float32)
        return np.concatenate(windows, axis=0).astype(np.float32)

    def get_data_loader(self, split='train', column='sens'):
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")

        df_split = getattr(self, f'df_{split}')
        X_split = self._windows_from_df(df_split, column=column)
        X_split = torch.tensor(X_split, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
        )

    @staticmethod
    def _split_dataframe_contiguously(df, n_splits):
        if n_splits == 1:
            return [df.reset_index(drop=True)]
        ranges = _contiguous_ranges(len(df), n_splits)
        return [df.iloc[start:end].reset_index(drop=True) for start, end in ranges]


    def get_data_clients(self, num_clients=None, split='train', column='sens', stratify_col=None, return_loaders=True, **kwargs):
        """Return list of DataLoaders for each client by splitting the dataset."""
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")

        num_clients = self.num_clients if num_clients is None else num_clients

        if num_clients == self.num_clients:
            index = {'train': 0, 'valid': 1, 'test': 2}[split]
            df_clients = [self.datasets_client[client_id][index] for client_id in range(self.num_clients)]
        else:
            df_split = getattr(self, f'df_{split}')
            df_clients = self._split_dataframe_contiguously(df_split, num_clients)

        if return_loaders:
            client_loaders = []
            for df_client in df_clients:
                X_client = self._windows_from_df(df_client, column=column)
                X_client = torch.tensor(X_client, dtype=torch.float32).to(self.device)
                client_dataset = TensorDataset(X_client)
                client_loader = DataLoader(
                    client_dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == 'train')
                )
                client_loaders.append(client_loader)
            return client_loaders
        return df_clients

    def get_dataset_info(self):
        return {
            "num_features": self.num_features,
            "feature_cols": self.feature_cols,
            "binary_feature_cols": self.binary_feature_cols,
            "continuous_feature_cols": self.continuous_feature_cols,
            "num_clients": self.num_clients,
            "clients": self.info,
        }

    def fit_scaler(self, split='train', column='sens', client_id=0):
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")
        if not (0 <= client_id < len(self.scalers)):
            raise IndexError("client_id out of range")

        scaler_bundle = self.scalers[client_id]
        scaler = scaler_bundle["continuous_scaler"]
        if scaler is None or not scaler_bundle["continuous_feature_cols"]:
            return

        df_split = self.datasets_client[client_id][{'train': 0, 'valid': 1, 'test': 2}[split]]
        sens = np.asarray(df_split.iloc[0][column], dtype=np.float32)
        idxs = [self.feature_cols.index(col) for col in scaler_bundle["continuous_feature_cols"]]
        scaler.fit(sens[:, idxs])

    def scale_data(self, split='train', column='sens', client_id=0):
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")
        if not (0 <= client_id < len(self.scalers)):
            raise IndexError("client_id out of range")

        split_index = {'train': 0, 'valid': 1, 'test': 2}[split]
        df_split = self.datasets_client[client_id][split_index]
        sens = np.asarray(df_split.iloc[0][column], dtype=np.float32)
        train_sens = np.asarray(self.datasets_client[client_id][0].iloc[0][column], dtype=np.float32)
        scaler_bundle = self.scalers[client_id]

        sens_scaled = self._transform_discrete(sens, train_sens)
        sens_scaled = self._transform_continuous(sens_scaled, scaler_bundle["continuous_scaler"])
        df_split.at[df_split.index[0], column] = sens_scaled

    def inverse_scale(self, split='train', column='sens', client_id=0):
        split = split.lower()
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be either 'train', 'valid' or 'test'")
        if not (0 <= client_id < len(self.scalers)):
            raise IndexError("client_id out of range")

        split_index = {'train': 0, 'valid': 1, 'test': 2}[split]
        df_split = self.datasets_client[client_id][split_index]
        sens = np.asarray(df_split.iloc[0][column], dtype=np.float32).copy()
        scaler_bundle = self.scalers[client_id]
        scaler = scaler_bundle["continuous_scaler"]
        if scaler is None:
            return

        idxs = [self.feature_cols.index(col) for col in scaler_bundle["continuous_feature_cols"]]
        sens[:, idxs] = scaler.inverse_transform(sens[:, idxs])
        df_split.at[df_split.index[0], column] = sens
    
