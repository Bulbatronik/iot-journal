import os
import json
import warnings
from copy import deepcopy
from pprint import pformat

import numpy as np
import torch

from datasets.aramis import AramisDataset
from datasets.swat import SWaTDataset

from utils.config import validate_config, load_config
from utils.experiment_utils import get_fed_argparser
from utils.logger import setup_logger, log_experiment_header
from utils.model_utils import create_model, create_optimizer
from utils.utils import get_device, clean_cuda, set_seed, NpEncoder

from trainers.training import train_models, compute_validation_performance
from evaluator.evaluate import reconstruct_dataset, evaluate_model
from evaluator.threshold_selection import select_threshold
from evaluator.detection import resolve_detection_mode

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)


def create_dataset(args, config, device):
    dataset_name = args.dataset_name.lower()

    dataset_config = dict(config["dataset"])
    if dataset_name == "swat":
        dataset_config["path"] = "data/merged.csv"
        dataset = SWaTDataset(
            data_path=dataset_config["path"],
            window_size=dataset_config["window_size"],
            stride=dataset_config["stride"],
            batch_size=args.batch_size,
            num_clients=args.num_clients,
            device=device,
            seed=args.seed,
        )
    elif dataset_name == "aramis":
        dataset = AramisDataset(
            data_path=dataset_config["path"],
            window_size=dataset_config["window_size"],
            stride=dataset_config["stride"],
            delta_tau=dataset_config["delta_tau"],
            batch_size=args.batch_size,
            num_clients=args.num_clients,
            device=device,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    config["dataset"]["path"] = dataset_config["path"]
    config["dataset"]["num_features"] = dataset.num_features
    return dataset


def main() -> None:
    """Run complete federated experiment workflow."""
    results = {}

    # Parse arguments
    args, _ = get_fed_argparser()

    # Load and validate configuration
    config = load_config(args.model_name, args.base_config)
    validate_config(config, args.model_name)

    # Set up logger for this experiment
    logger, log_dir = setup_logger(args.model_name, args.experiment_id, args.results_dir)

    # GPU-related
    device = get_device()
    clean_cuda()
    set_seed(args.seed)

    # Log configuration and experiment header
    logger.info(f"Configuration loaded for model {args.model_name}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Arguments: {vars(args)}")
    log_experiment_header(logger, args, device)

    # Load dataset
    dataset = create_dataset(args, config, device)

    # Optional Weights & Biases tracking
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="iot-journal",
            name=f'feder-{args.fed_type}_data-{args.dataset_name}_model-{args.model_name}_C-{args.num_clients}_R-{args.num_rounds}_E-{args.epochs}_B-{args.batch_size}_exp-{args.experiment_id}',
            group=f"feder-{args.fed_type}_{args.dataset_name}_{args.model_name}",
            config={**vars(args), **config},
        )

    # Per-client training and validation data
    num_clients = args.num_clients
    train_loaders = dataset.get_data_clients(num_clients=num_clients, split='train', column='sens', stratify_col=None, return_loaders=True)
    df_valid_clients = dataset.get_data_clients(num_clients=num_clients, split='valid', return_loaders=False)
    thr_select_config = config.get('thr_select', {})
    if args.objective is not None:
        thr_select_config = {**thr_select_config, 'objective': args.objective}
    num_samples_per_client = [len(loader.dataset) for loader in train_loaders]
    logger.info(f"Training set size: {num_samples_per_client} sequences")

    # Global model and per-client clones with individual optimizers
    model_global, criterion, training_config = create_model(config, args, device)
    results.update({
        'run_id': args.experiment_id,
        'model_name': args.model_name,
        'log_dir': log_dir,
        'num_parameters': sum(p.numel() for p in model_global.parameters()),
    })
    models = [deepcopy(model_global) for _ in range(num_clients)]
    optimizers_client = [create_optimizer(models[i], config, args) for i in range(num_clients)]

    # Shared parameter set: full model, analysis (encoder/critic), or synthesis (decoder/generator)
    if args.fed_type == 'full':
        federate_weight_names = [name for name, _ in model_global.named_parameters()]
        logger.info("Federated learning type: Full model aggregation")
    elif args.fed_type == 'enc':
        federate_weight_names = model_global.get_encoder_weights_names()
        logger.info("Federated learning type: Encoder/Critic aggregation")
    elif args.fed_type == 'dec':
        federate_weight_names = model_global.get_decoder_weights_names()
        logger.info("Federated learning type: Decoder/Generator aggregation")
    else:
        raise ValueError(f"Unknown federated type: {args.fed_type}")

    results['num_shared_parameters'] = sum(
        p.numel() for name, p in model_global.named_parameters() if name in federate_weight_names
    )
    logger.info(f"Shared weights: {federate_weight_names}")

    # Initialize all clients identically
    for name, param in model_global.named_parameters():
        for i in range(num_clients):
            models[i].state_dict()[name].copy_(param.data)

    def sync_shared_weights():
        for name, param in model_global.named_parameters():
            if name in federate_weight_names:
                for i in range(num_clients):
                    models[i].state_dict()[name].copy_(param.data)

    # Federated training with early stopping and best-validation-checkpoint selection
    logger.info("Starting federated training...")
    best_val_perf = -1.0
    best_round = 0
    best_global_state = deepcopy(model_global.state_dict())
    best_client_states = [deepcopy(m.state_dict()) for m in models]
    rounds_without_improvement = 0

    for r in range(args.num_rounds):
        logger.info(f"--- ROUND {r+1}/{args.num_rounds} ---")

        # Broadcast shared parameters, train locally
        sync_shared_weights()
        train_models(models, train_loaders, optimizers_client, criterion, training_config, args.epochs)

        # FedAvg aggregation of the shared parameters, weighted by client data volume
        logger.info(f"Aggregating model weights for round {r+1}...")
        total_samples = sum(num_samples_per_client)
        count = 0
        for name, param in model_global.named_parameters():
            if name in federate_weight_names:
                aggregated_weight = sum(
                    (num_samples_per_client[i] / total_samples) * models[i].state_dict()[name]
                    for i in range(num_clients)
                )
                param.data.copy_(aggregated_weight)
                count += 1
        assert count == len(federate_weight_names), "Not all federated weights were aggregated"

        # Validation for early stopping: clients evaluate with the updated shared weights
        sync_shared_weights()
        val_scores = [
            compute_validation_performance(models[i], dataset, df_valid_clients[i], config, args, thr_select_config)
            for i in range(num_clients)
        ]
        val_perf = float(np.mean(val_scores))
        logger.info(f"Round {r+1} validation F1: {val_perf:.5f} (best: {best_val_perf:.5f} @ round {best_round})")
        if wandb_run:
            wandb_run.log({'round': r + 1, 'val_f1': val_perf})

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            best_round = r + 1
            best_global_state = deepcopy(model_global.state_dict())
            best_client_states = [deepcopy(m.state_dict()) for m in models]
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if args.patience > 0 and rounds_without_improvement >= args.patience:
                logger.info(f"Early stopping at round {r+1} (no improvement for {args.patience} rounds)")
                break

    # Restore the best-validation checkpoints for evaluation
    logger.info(f"Restoring best-validation checkpoint from round {best_round} (val F1 {best_val_perf:.5f})")
    model_global.load_state_dict(best_global_state)
    for i in range(num_clients):
        models[i].load_state_dict(best_client_states[i])
    results['best_round'] = best_round
    results['best_val_f1'] = best_val_perf

    # Client-specific threshold calibration on the validation set (offline, post training)
    logger.info("Selecting client-specific thresholds...")
    assert config[args.model_name]["generation"]['filter_size'] is None, "The reconstruction of the dataset should have no filtering"

    shared_eval = args.eval_scope == 'shared'
    df_test_clients = dataset.get_data_clients(num_clients=num_clients, split='test', return_loaders=False)

    threshold_detection_mode = resolve_detection_mode(
        args.dataset_name,
        thr_select_config.get('detection_mode', 'auto'),
    )
    threshold_params_clients = []
    for i in range(num_clients):
        dataset_client = deepcopy(dataset)
        dataset_client.df_valid = df_valid_clients[i].copy()
        eval_model = model_global if args.fed_type == 'full' else models[i]
        dataset_client = reconstruct_dataset(eval_model, dataset_client, config, args, subset="valid", log_dir=log_dir)
        df_valid = dataset_client.df_valid
        assert 'anom_score' in df_valid.columns, "Dataset is not reconstructed"
        logger.info(f"Selecting threshold for client {i}...")
        threshold_params = select_threshold(
            df_valid,
            dataset_name=args.dataset_name,
            n_calls=thr_select_config.get('n_calls', 50),
            logger=logger,
            log_dir=log_dir,
            detection_mode=thr_select_config.get('detection_mode', 'auto'),
            objective_name=thr_select_config.get('objective', 'auto'),
            min_consecutive=thr_select_config.get('min_consecutive', 1),
            optimize_min_consecutive=thr_select_config.get('optimize_min_consecutive', False),
            min_consecutive_bounds=tuple(thr_select_config.get('min_consecutive_bounds', [1, 50])),
            score_transform=thr_select_config.get('score_transform', 'auto'),
            drift_window=thr_select_config.get('drift_window'),
        )
        threshold_params_clients.append(threshold_params)
        logger.info(f"Client {i} selected threshold parameters: {threshold_params}")
    logger.info("Resolved threshold detection mode: %s", threshold_detection_mode)

    # Final evaluation on the configured scope (global model under full
    # federation, personalized model under partial federation)
    logger.info("Evaluating on the test set...")
    for i in range(num_clients):
        dataset_client = deepcopy(dataset)
        dataset_client.df_test = (dataset.df_test if shared_eval else df_test_clients[i]).copy()
        eval_model = model_global if args.fed_type == 'full' else models[i]
        dataset_client = reconstruct_dataset(eval_model, dataset_client, config, args, subset="test", log_dir=log_dir)
        df_test = dataset_client.df_test
        assert 'anom_score' in df_test.columns, "Dataset is not reconstructed"
        results_eval = evaluate_model(df_test, threshold_params_clients[i], dataset_name=args.dataset_name)
        logger.info(f"Evaluation results for client {i}:\n%s", pformat(results_eval))
        results[f'eval_client_{i}'] = results_eval

    # Average the results across clients
    avg_results = {}
    for i in range(num_clients):
        for key, value in results[f'eval_client_{i}'].items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            avg_results[key] = avg_results.get(key, 0.0) + value / num_clients

    results['eval_avg'] = avg_results
    logger.info("AVERAGE EVALUATION RESULTS:\n%s", pformat(avg_results))
    if wandb_run:
        wandb_run.log(avg_results)
        wandb_run.finish()

    # Save results to disk (log_dir)
    results_path = os.path.join(log_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
