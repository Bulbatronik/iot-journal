import math
import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class Chomp1d(nn.Module):
    """Remove the trailing padding introduced by causal convolutions."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual causal TCN block used by the SWGAN generator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            Chomp1d(0),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            Chomp1d(0),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return self.activation(out + residual)


class TCNGenerator(nn.Module):
    """Generator that reconstructs the target horizon from a noise-corrupted target segment."""

    def __init__(
        self,
        num_feat: int,
        target_len: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError('hidden_dims must contain at least one element')
        self.num_feat = num_feat
        self.target_len = target_len
        self.hidden_dims = hidden_dims

        dilations = dilations or [2 ** i for i in range(len(hidden_dims))]
        if len(dilations) < len(hidden_dims):
            repeats = math.ceil(len(hidden_dims) / len(dilations))
            dilations = (dilations * repeats)[: len(hidden_dims)]

        blocks = []
        in_channels = num_feat
        for out_channels, dilation in zip(hidden_dims, dilations):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(in_channels, num_feat)
        #self.tanh = nn.Tanh()

    def forward(self, x_noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_noisy: (B, target_len, M)
        x = x_noisy.transpose(1, 2)  # (B, M, target_len)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (B, target_len, H)
        features = self.dropout(self.layer_norm(x))
        recon = self.output_layer(features)
        #recon = self.tanh(recon)
        return recon, features


class ConvCritic(nn.Module):
    """Convolutional discriminator/critic used by the SWGAN branch."""

    def __init__(
        self,
        num_feat: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError('hidden_dims must contain at least one element')

        layers = []
        in_channels = num_feat
        padding = kernel_size // 2
        for out_channels in hidden_dims:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(in_channels, in_channels // 2 if in_channels > 1 else 1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(in_channels // 2 if in_channels > 1 else 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, M)
        feats = self.feature_extractor(x.transpose(1, 2)).transpose(1, 2)  # (B, L, H)
        scores = self.mlp(feats)
        return scores, feats


class LSTMPredictor(nn.Module):
    """Two-layer LSTM predictor followed by a fully-connected output layer."""

    def __init__(
        self,
        num_feat: int,
        conditioning_len: int,
        target_len: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_feat = num_feat
        self.conditioning_len = conditioning_len
        self.target_len = target_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=num_feat,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, target_len * num_feat)

    def forward(self, conditioning_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, (h_n, _c_n) = self.lstm(conditioning_seq)
        last_hidden = h_n[-1]
        pred = self.output_layer(last_hidden).view(conditioning_seq.size(0), self.target_len, self.num_feat)
        return pred, outputs


class FedSWTSAD(nn.Module):
    """Faithful SWGAN + predictor backbone adapted to the user's repo interface."""

    def __init__(
        self,
        num_feat: int,
        seq_len: int,
        hidden_dims_generator: List[int],
        hidden_dims_discriminator: List[int],
        predictor_hidden_dim: int = 256,
        conditioning_len: Optional[int] = None, 
        predictor_num_layers: int = 2,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        generator_dropout: float = 0.1,
        discriminator_dropout: float = 0.1,
        predictor_dropout: float = 0.1,
        negative_slope: float = 0.2,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.conditioning_len = conditioning_len if conditioning_len is not None else seq_len // 2
        if not (1 <= self.conditioning_len < seq_len):
            raise ValueError(f'conditioning_len must be in [1, seq_len-1], got {self.conditioning_len} for seq_len={seq_len}')
        self.target_len = seq_len - self.conditioning_len

        self.generator = TCNGenerator(
            num_feat=num_feat,
            target_len=self.target_len,
            hidden_dims=hidden_dims_generator,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=generator_dropout,
            negative_slope=negative_slope,
        )
        self.discriminator = ConvCritic(
            num_feat=num_feat,
            hidden_dims=hidden_dims_discriminator,
            kernel_size=kernel_size,
            dropout=discriminator_dropout,
            negative_slope=negative_slope,
        )
        self.predictor = LSTMPredictor(
            num_feat=num_feat,
            conditioning_len=self.conditioning_len,
            target_len=self.target_len,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=predictor_dropout,
        )

    def forward(self, x: torch.Tensor):
        raise NotImplementedError('Use fit(...) for training and gen_seq(...) for inference.')

    def get_encoder_weights_names(self) -> List[str]:
        names = [f'discriminator.{name}' for name, _ in self.discriminator.named_parameters()]
        #names.extend([f'predictor.{name}' for name, _ in self.predictor.named_parameters()])
        return names

    def get_pred_weights_names(self) -> List[str]:
        names = [f'predictor.{name}' for name, _ in self.predictor.named_parameters()]
        return names

    def get_decoder_weights_names(self) -> List[str]:
        return [f'generator.{name}' for name, _ in self.generator.named_parameters()]

    def _split_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conditioning = x[:, : self.conditioning_len, :]
        target = x[:, self.conditioning_len :, :]
        return conditioning, target

    def _sample_noisy_target(self, target: torch.Tensor, noise_std: float) -> torch.Tensor:
        if noise_std <= 0:
            return target
        noise = torch.randn_like(target) * noise_std
        return target + noise

    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        device = real_samples.device
        alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
        interpolated = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates, _ = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def fit(
        self,
        train_loader,
        optimizers: Dict[str, torch.optim.Optimizer],
        criterions,
        epochs: int,
        verbose: bool = True,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        generator_noise_std: float = 1.0,
        recon_loss_weight: float = 0.0,
        **kwargs,
    ) -> Dict[str, float]:
        del criterions, kwargs
        logger = logging.getLogger(__name__)
        self.train()
        device = next(self.parameters()).device

        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            p_losses = []

            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', disable=not verbose),
                start=1,
            ):
                x_real = batch[0].to(device)
                conditioning, target = self._split_window(x_real)

                # Predictor step (parallel to the discriminator branch per the paper)
                optimizers['Predictor'].zero_grad()
                pred_target, _ = self.predictor(conditioning)
                p_loss = F.mse_loss(pred_target, target)
                p_loss.backward()
                optimizers['Predictor'].step()
                p_losses.append(p_loss.item())

                # Discriminator / critic step
                optimizers['Discriminator'].zero_grad()
                noisy_target = self._sample_noisy_target(target, generator_noise_std)
                fake_target, _ = self.generator(noisy_target)
                real_validity, _ = self.discriminator(target)
                fake_validity, _ = self.discriminator(fake_target.detach())
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                gradient_penalty = self._compute_gradient_penalty(target, fake_target.detach())
                d_loss = d_loss + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizers['Discriminator'].step()
                d_losses.append(d_loss.item())

                # Generator step every n_critic batches
                if batch_idx % max(1, n_critic) == 0:
                    optimizers['Generator'].zero_grad()
                    noisy_target = self._sample_noisy_target(target, generator_noise_std)
                    fake_target, _ = self.generator(noisy_target)
                    fake_validity, _ = self.discriminator(fake_target)
                    g_loss = -torch.mean(fake_validity)
                    if recon_loss_weight > 0:
                        g_loss = g_loss + recon_loss_weight * F.mse_loss(fake_target, target)
                    g_loss.backward()
                    optimizers['Generator'].step()
                    g_losses.append(g_loss.item())

            avg_d = float(sum(d_losses) / max(1, len(d_losses)))
            avg_g = float(sum(g_losses) / max(1, len(g_losses)))
            avg_p = float(sum(p_losses) / max(1, len(p_losses)))
            if verbose:
                logger.info(f'Epoch {epoch + 1} completed. Avg Generator Loss: {avg_g}, Avg Critic Loss: {avg_d}, Avg Predictor Loss: {avg_p}')
                logger.info('-----------------------------------------------------')
                #print(f'Epoch {epoch + 1} completed. Avg Generator Loss: {avg_g}, Avg Critic Loss: {avg_d}, Avg Predictor Loss: {avg_p}')
                #print('-----------------------------------------------------')

        return {'avg_loss_g': avg_g, 'avg_loss_c': avg_d, 'avg_loss_p': avg_p}

    @torch.no_grad()
    def validation_loss(self, val_loader, criterion=None):
        """
        Validation loss for early stopping: generator reconstruction MSE on the
        target window plus predictor forecasting MSE.
        """
        self.eval()
        device = next(self.parameters()).device
        total = 0.0
        for batch in val_loader:
            x = batch[0].to(device)
            conditioning, target = self._split_window(x)
            fake_target, _ = self.generator(target)
            pred_target, _ = self.predictor(conditioning)
            total += (F.mse_loss(fake_target, target) + F.mse_loss(pred_target, target)).item()
        return total / max(1, len(val_loader))

    @torch.no_grad()
    def gen_seq(
        self,
        input_seq: torch.Tensor,
        alpha: float = 0.35,
        beta: float = 0.15,
        gamma: float = 0.50,
        num_samples: int = 1,
        generator_noise_std: float = 1.0,
        blank_fill_value: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        self.eval()
        device = next(self.parameters()).device
        x = input_seq.to(device)
        conditioning, target = self._split_window(x)

        recon_acc = torch.zeros_like(target)
        pred_target, _ = self.predictor(conditioning)
        d_score_acc = torch.zeros(target.size(0), target.size(1), 1, device=device)

        for _ in range(max(1, num_samples)):
            noisy_target = self._sample_noisy_target(target, generator_noise_std)
            recon_target, _ = self.generator(noisy_target)
            #real_validity, _ = self.discriminator(target)
            #d_score_acc = d_score_acc + (1.0 - real_validity)
            fake_validity, _ = self.discriminator(recon_target)
            d_score_acc = d_score_acc + (1.0 - fake_validity)
            recon_acc = recon_acc + recon_target
            
        recon_target = recon_acc / max(1, num_samples)
        d_score = d_score_acc / max(1, num_samples)
        r_score = torch.mean((target - recon_target) ** 2, dim=2, keepdim=True)
        p_score = torch.mean((target - pred_target) ** 2, dim=2, keepdim=True)
        anomaly_target = alpha * r_score + beta * d_score + gamma * p_score

        blank_recon = torch.full(
            (x.size(0), self.conditioning_len, self.num_feat),
            fill_value=blank_fill_value,
            device=device,
            dtype=x.dtype,
        )
        blank_score = torch.full(
            (x.size(0), self.conditioning_len, 1),
            fill_value=blank_fill_value,
            device=device,
            dtype=x.dtype,
        )
        full_reconstruction = torch.cat([blank_recon, recon_target], dim=1)
        full_anomaly = torch.cat([blank_score, anomaly_target], dim=1)
        return full_reconstruction, full_anomaly

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(state)
