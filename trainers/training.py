import logging
from copy import deepcopy

from evaluator.evaluate import reconstruct_dataset, evaluate_model
from evaluator.threshold_selection import select_threshold


def train_model(model, train_loader, optimizer, criterion, training_config, epochs):
    model.fit(train_loader, optimizer, criterion, epochs, **training_config)


def train_models(models_list, train_loaders_list, optimizers_list, criterion, training_config, epochs, active=None):
    """Train each client model locally. `active` optionally masks out early-stopped clients."""
    logger = logging.getLogger(__name__)
    for i in range(len(models_list)):
        if active is not None and not active[i]:
            logger.info(f"Skipping client {i+1}/{len(models_list)} (early-stopped)")
            continue
        logger.info(f"Training model for client {i+1}/{len(models_list)}: {models_list[i].__class__.__name__}")
        train_model(
            models_list[i],
            train_loaders_list[i],
            optimizers_list[i],
            criterion,
            training_config,
            epochs,
        )


def compute_validation_loss(model, val_loader, criterion):
    val_criterion = None if isinstance(criterion, dict) else criterion
    return model.validation_loss(val_loader, criterion=val_criterion)


def compute_validation_performance(model, dataset, df_valid_client, config, args,
                                   thr_select_config, f1_key='F1'):
    dataset_client = deepcopy(dataset)
    dataset_client.df_valid = df_valid_client.copy()
    dataset_client = reconstruct_dataset(model, dataset_client, config, args, subset="valid", log_dir=None)
    df_valid = dataset_client.df_valid

    threshold_params = select_threshold(
        df_valid,
        dataset_name=args.dataset_name,
        n_calls=thr_select_config.get('n_calls', 50),
        detection_mode=thr_select_config.get('detection_mode', 'auto'),
        objective_name=thr_select_config.get('objective', 'auto'),
        min_consecutive=thr_select_config.get('min_consecutive', 1),
        optimize_min_consecutive=thr_select_config.get('optimize_min_consecutive', False),
        min_consecutive_bounds=tuple(thr_select_config.get('min_consecutive_bounds', [1, 50])),
        score_transform=thr_select_config.get('score_transform', 'auto'),
        drift_window=thr_select_config.get('drift_window'),
    )
    results_eval = evaluate_model(df_valid, threshold_params, dataset_name=args.dataset_name)
    return float(results_eval.get(f1_key, 0.0))
