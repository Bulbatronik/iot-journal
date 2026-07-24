"""
Model and optimizer creation utilities for centralized/federated training.

This module provides functions for creating models, optimizers, and loss functions
based on configuration parameters. It supports the four generative architectures
evaluated in the paper: LSTM-VAE, TAnoWGAN (WGAN-GP), TAnoDDPM, and FedSW-TSAD.
"""
from typing import Dict, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.optim import Adam

from models import LSTM_VAE, VaeLoss, WGAN, DDPM, FedSWTSAD

# Type aliases for better readability
ModelType = Union[LSTM_VAE, WGAN, DDPM, FedSWTSAD]
OptimizerType = Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]
CriterionType = Union[nn.Module, Dict[str, nn.Module], None]
ConfigType = Dict[str, Any]


def _create_vae_model(
    num_features: int,
    window_size: int,
    config: ConfigType,
    device: torch.device
) -> Tuple[ModelType, CriterionType, ConfigType]:
    """Create the LSTM-VAE model with its loss function."""
    vae_config = config['vae']
    arch_config = vae_config['architecture']

    model = LSTM_VAE(
        num_feat=num_features,
        seq_len=window_size,
        hidden_dims_encoder=arch_config['hidden_dims_encoder'],
        hidden_dims_decoder=arch_config['hidden_dims_decoder'],
        latent_dim=arch_config['latent_dim']
    ).to(device)

    criterion = VaeLoss(beta=vae_config['beta'])
    training_config = {}

    return model, criterion, training_config


def _create_wgan_model(
    num_features: int,
    window_size: int,
    config: ConfigType,
    device: torch.device
) -> Tuple[ModelType, CriterionType, ConfigType]:
    """Create the TAnoWGAN (WGAN-GP) model."""
    wgan_config = config['wgan_gp']
    arch_config = wgan_config['architecture']

    model = WGAN(
        num_feat=num_features,
        seq_len=window_size,
        hidden_dims_generator=arch_config['hidden_dims_generator'],
        hidden_dims_discriminator=arch_config['hidden_dims_discriminator'],
        latent_dim=arch_config['latent_dim']
    ).to(device)

    # Wasserstein losses are computed inside the model
    criterion = {'Generator': None, 'Discriminator': None}
    training_config = {
        'n_critic': arch_config['n_critic'],
        'lambda_gp': arch_config['lambda_gp'],
    }

    return model, criterion, training_config


def _create_fedsw_tsad_model(
    num_features: int,
    window_size: int,
    config: ConfigType,
    device: torch.device
) -> Tuple[ModelType, CriterionType, ConfigType]:
    """Create FedSW-TSAD model with its training config."""
    model_config = config['fedsw_tsad']
    arch_config = model_config['architecture']

    model = FedSWTSAD(
        num_feat=num_features,
        seq_len=window_size,
        conditioning_len=arch_config.get('conditioning_length'),
        hidden_dims_generator=arch_config['hidden_dims_generator'],
        hidden_dims_discriminator=arch_config['hidden_dims_discriminator'],
        predictor_hidden_dim=arch_config['predictor_hidden_dim'],
        predictor_num_layers=arch_config.get('predictor_num_layers', 2),
        kernel_size=arch_config.get('kernel_size', 3),
        dilations=arch_config.get('dilations'),
        generator_dropout=arch_config.get('generator_dropout', 0.1),
        discriminator_dropout=arch_config.get('discriminator_dropout', 0.1),
        predictor_dropout=arch_config.get('predictor_dropout', 0.1),
        negative_slope=arch_config.get('negative_slope', 0.2),
    ).to(device)

    criterion = {'Generator': None, 'Discriminator': None, 'Predictor': nn.MSELoss()}
    training_config = {
        'n_critic': arch_config.get('n_critic', 5),
        'lambda_gp': arch_config.get('lambda_gp', 10.0),
        'generator_noise_std': arch_config.get('generator_noise_std', 1.0),
        'recon_loss_weight': arch_config.get('recon_loss_weight', 0.0),
    }

    return model, criterion, training_config


def _create_ddpm_model(
    num_features: int,
    window_size: int,
    config: ConfigType,
    device: torch.device
) -> Tuple[ModelType, CriterionType, ConfigType]:
    """Create the TAnoDDPM model."""
    ddpm_config = config['ddpm']
    arch_config = ddpm_config['architecture']

    model = DDPM(
        unet_configs=arch_config['unet_configs'],
        num_timesteps=ddpm_config['beta_schedule']['num_timesteps'],
        beta_start=ddpm_config['beta_schedule']['beta_start'],
        beta_end=ddpm_config['beta_schedule']['beta_end']
    ).to(device)

    # DDPM uses the denoising MSE loss internally
    criterion = None
    training_config = {}

    return model, criterion, training_config


def create_model(
    config: ConfigType,
    args: Any,
    device: torch.device,
    weights_path: str = None,
) -> Tuple[ModelType, CriterionType, ConfigType]:
    """
    Instantiate and return a model (without creating its optimizer).

    Intended for FL workflows where you create a model once and then
    clone it C times and create a separate optimizer for each cloned model.

    Returns:
        Tuple containing `(model, criterion, training_config)`
    """
    dataset_params = config.get('dataset', {})
    num_features = dataset_params.get('num_features', 10)
    window_size = dataset_params.get('window_size', 20)

    if args.model_name == 'vae':
        model, criterion, training_config = _create_vae_model(
            num_features, window_size, config, device
        )
    elif args.model_name == 'wgan_gp':
        model, criterion, training_config = _create_wgan_model(
            num_features, window_size, config, device
        )
    elif args.model_name == 'ddpm':
        model, criterion, training_config = _create_ddpm_model(
            num_features, window_size, config, device
        )
    elif args.model_name == 'fedsw_tsad':
        model, criterion, training_config = _create_fedsw_tsad_model(
            num_features, window_size, config, device
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    if weights_path:
        if hasattr(model, 'load_model'):
            model.load_model(weights_path)
        else:
            try:
                state = torch.load(weights_path, map_location=device)
                model.load_state_dict(state)
            except Exception:
                raise RuntimeError(f"Failed to load weights from {weights_path}")

    return model, criterion, training_config


def create_optimizer(
    model: ModelType,
    config: ConfigType,
    args: Any,
) -> OptimizerType:
    """
    Create and return an optimizer (or dict of optimizers) for a given model,
    so that per-client optimizers can be created for cloned models in FL.
    """
    if args.model_name == 'vae':
        vae_config = config['vae']
        optimizer = Adam(
            model.parameters(),
            lr=vae_config['optimizer']['learning_rate'],
            betas=vae_config['optimizer']['betas'],
        )
    elif args.model_name == 'wgan_gp':
        wgan_config = config['wgan_gp']
        optimizer = {
            'Generator': Adam(
                model.generator.parameters(),
                lr=wgan_config['optimizer']['learning_rate'],
                betas=wgan_config['optimizer']['betas'],
            ),
            'Discriminator': Adam(
                model.discriminator.parameters(),
                lr=wgan_config['optimizer']['learning_rate'],
                betas=wgan_config['optimizer']['betas'],
            ),
        }
    elif args.model_name == 'ddpm':
        ddpm_config = config['ddpm']
        optimizer = Adam(
            model.parameters(),
            lr=ddpm_config['optimizer']['learning_rate'],
            betas=ddpm_config['optimizer']['betas'],
        )
    elif args.model_name == 'fedsw_tsad':
        model_config = config['fedsw_tsad']
        learning_rate = model_config['optimizer']['learning_rate']
        betas = tuple(model_config['optimizer']['betas'])
        optimizer = {
            'Generator': Adam(
                model.generator.parameters(),
                lr=learning_rate,
                betas=betas,
            ),
            'Discriminator': Adam(
                model.discriminator.parameters(),
                lr=learning_rate,
                betas=betas,
            ),
            'Predictor': Adam(
                model.predictor.parameters(),
                lr=learning_rate,
                betas=betas,
            ),
        }
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    return optimizer
