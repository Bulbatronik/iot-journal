from typing import Dict, Any, Tuple
import os
import logging


def setup_logger(model_name: str, run_id: int, results_dir: str) -> tuple[logging.Logger, str]:
    # Create the log directory path and ensure it exists
    log_dir = os.path.join(results_dir, model_name, f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler for logging to a file
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger so all child loggers propagate to it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get or create a logger with a unique name for this experiment
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    return logger, log_dir


def log_experiment_header(logger: logging.Logger, args: Any, device: str) -> None:
    """Log formatted experiment header with key details.
    
    Args:
        logger: Logger instance
        args: Command-line arguments
        device: Computing device
    """
    separator = "=" * 60
    
    logger.info(separator)
    logger.info("CENTRALIZED TRAINING & EVALUATION EXPERIMENT")
    logger.info(separator)
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Experiment ID: {args.experiment_id}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Device: {device}")
    logger.info(separator)
    
    # Log additional experiment parameters
    logger.info(f"Starting experiment {args.experiment_id} for model {args.model_name}")


def log_experiment_summary(logger: logging.Logger, result: Dict[str, Any]) -> None:
    """Log experiment results summary.
    
    Args:
        logger: Logger instance
        result: Results dictionary
    """
    logger.info(f"{'='*60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Experiment ID: {result.get('run_id', '-')}")
    
    logger.info(f"Training time: {result['training_time']:.2f} seconds" if result['training_time'] else "Training skipped")
    logger.info(f"Model parameters: {result['num_parameters']:,}" if result['num_parameters'] else "N/A")
    
    # Log evaluation results if available
    eval_results = result['eval']
    
    logger.info(f"{'='*60}")
    logger.info("EVALUATION RESULTS ON TEST DATASET")
    logger.info(f"{'='*60}")
    
    # Per-sample (micro) statistics
    micro = eval_results['per_sample']
    logger.info("Per-sample (micro) statistics:")
    logger.info(f"True Positives (TP): {micro['TP']}")
    logger.info(f"True Negatives (TN): {micro['TN']}")
    logger.info(f"False Positives (FP): {micro['FP']}")
    logger.info(f"False Negatives (FN): {micro['FN']}")
    logger.info(f"Precision: {micro['P']:.4f}")
    logger.info(f"Recall: {micro['R']:.4f}")
    logger.info(f"F1 Score: {micro['F1']:.4f}")
    
    # Detection timing statistics
    logger.info(f"Average Delta Early Detection (FP): {micro['mean_delta_early']:.4f}+-{micro['std_delta_early']:.4f}")
    logger.info(f"Average Delta for Late detection (FN): {micro['mean_delta_late']:.4f}+-{micro['std_delta_late']:.4f}")
    
    # Per-sequence (macro) statistics
    logger.info(f"{'-'*60}")
    logger.info("Per-sequence (macro) statistics:")
    macro = eval_results['per_sequence']
    logger.info(f"True Positives (TP): {macro['TP']}")
    logger.info(f"True Negatives (TN): {macro['TN']}")
    logger.info(f"False Positives (FP): {macro['FP']}")
    logger.info(f"False Negatives (FN): {macro['FN']}")
    logger.info(f"Precision: {macro['P']:.4f}")
    logger.info(f"Recall: {macro['R']:.4f}")
    logger.info(f"F1 Score: {macro['F1']:.4f}")
   
    logger.info(f"{'='*60}")
    logger.info("Experiment completed!")
    logger.info(f"Results saved in: {result.get('log_dir', '-')}")
    logger.info(f"{'='*60}")
