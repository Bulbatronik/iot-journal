import argparse


def get_fed_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Federated training for generative anomaly detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['vae', 'wgan_gp', 'ddpm', 'fedsw_tsad'],
        help='Model architecture to train (LSTM-VAE, TAnoWGAN, TAnoDDPM, FedSW-TSAD)'
    )

    # Experiment identification
    parser.add_argument(
        '--experiment_id',
        type=int,
        default=0,
        help='Unique experiment run identifier for logging and output separation'
    )

    # Number of clients
    parser.add_argument(
        '--num_clients',
        type=int,
        default=5,
        help='Number of federated clients (1 for centralized training)'
    )

    # Number of rounds
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=30,
        help='Number of federated communication rounds'
    )

    # Federation type
    parser.add_argument(
        '--fed_type',
        type=str,
        default='full',
        choices=['full', 'enc', 'dec'],
        help='Type of federation: full model, encoder/critic only, or decoder/generator only'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='aramis',
        choices=['aramis', 'swat'],
        help='Dataset to use for training and evaluation'
    )

    # Number of epochs
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of local training epochs per round'
    )

    # Batch size
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility of results'
    )

    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience in communication rounds (0 disables early stopping)'
    )

    # Output and logging
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save experiment results and logs'
    )

    # Optional Weights & Biases tracking
    parser.add_argument(
        '--wandb',
        action='store_true',
        default=False,
        help='Enable Weights & Biases experiment tracking'
    )

    # Verbose output
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output during training'
    )

    # Configuration
    parser.add_argument(
        '--base_config',
        type=str,
        default='base_config',
        help='Base configuration name (without .yaml extension) to load'
    )

    # Evaluation scope
    parser.add_argument(
        '--eval_scope',
        type=str,
        default='per_client',
        choices=['shared', 'per_client'],
        help="Final calibration/evaluation scope: 'shared' global test set or "
             "'per_client' local shards"
    )

    # Threshold-calibration objective (overrides thr_select.objective in the config)
    parser.add_argument(
        '--objective',
        type=str,
        default=None,
        choices=['dtau', 'f1'],
        help="Threshold-calibration objective: 'dtau' (average time offset, ARAMIS) "
             "or 'f1' (point-wise F1, SWaT). Defaults to the config when not set."
    )

    args, unknown = parser.parse_known_args()
    return args, unknown
