import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm


class TimeEmbedding(nn.Module):
    """
    Time embedding using sinusoidal position embeddings.
    """
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        if time_emb_dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even.")
        
        self.lin1 = nn.Linear(time_emb_dim // 4, time_emb_dim)
        self.act1 = nn.SiLU()
        self.lin2 = nn.Linear(time_emb_dim, time_emb_dim)
        self.act2 = nn.SiLU()
        
    def forward(self, time):
        device = time.device
        half_dim = self.time_emb_dim // 8
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        
        embeddings = self.act1(self.lin1(embeddings))
        embeddings = self.act2(self.lin2(embeddings))
        
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block for time series with time embedding injection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        #padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)  
        self.act2 = nn.SiLU()
        
        # Skip connection if channel dimensions don't match
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        # x: [batch_size, in_channels, seq_len]
        # time_emb: [batch_size, time_emb_dim]
        
        # Main branch
        h = self.norm1(x)        
        h = self.act1(x)
        h = self.conv1(h)
        
        # Inject time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None]  # [batch_size, out_channels, 1]
        h = h + time_emb
        
        h = self.norm2(h)
        h = self.act2(h) # NOTE: They also have a dropout here, but I don't think it's necessary for time series
        h = self.conv2(h)
        
        # Skip connection
        return h + self.shortcut(x) # ([batch_size, out_channels, seq_len])


class AttentionBlock(nn.Module):
    
    def __init__(self, num_channels, num_heads=1, d_k=None):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        if d_k is None:
            d_k = num_channels
        
        self.d_k = d_k
        self.num_channels = num_channels
        self.num_heads = num_heads
    
        self.norm = nn.GroupNorm(1, num_channels)
        self.qkv_proj = nn.Linear(num_channels, num_heads * d_k * 3)
        self.out_proj = nn.Linear(num_heads * d_k, num_channels)
        
    def forward(self, x, time_emb=None):
        """
        x: [batch_size, num_channels, seq_len]
        time_emb: Optional time embedding, not used in this block
        """
        _ = time_emb  # Unused, but kept for compatibility
        batch_size, num_channels, seq_len = x.size()
        
        # Normalize input
        x = self.norm(x)
        
        # Project to Q, K, V
        # Change x to [batch_size, seq_len, num_channels]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, num_channels]
        qkv = self.qkv_proj(x) # [batch_size, seq_len, num_heads * d_k * 3]
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.d_k)  # [batch_size, seq_len, num_heads, 3 * d_k]
        q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V
        
        attn = torch.einsum('bshd,bshd->bhs', q, k)  # [batch_size, seq_len, num_heads]
        attn = attn / (self.d_k ** 0.5)  # Scale by sqrt(d_k)
        attn = torch.softmax(attn, dim=-1)  # Softmax over sequence length
        
        res = torch.einsum('bhs,bshd->bshd', attn, v)  # [batch_size, seq_len, num_heads, d_k]
        res = res.reshape(batch_size, seq_len, self.num_heads * self.d_k)  # [batch_size, seq_len, num_heads * d_k]
        res = self.out_proj(res)  # [batch_size, seq_len, num_channels]
        
        # Add residual connection
        res = res + x  # [batch_size, seq_len, num_channels]
        # Change back to [batch_size, num_channels, seq_len]
        res = res.permute(0, 2, 1)  # Change back to [batch_size, num_channels, seq_len]
        return res  # [batch_size, num_channels, seq_len]
        
        
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, has_attention=False):
        super(DownBlock, self).__init__()
       
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)#, kernel_size)
        if has_attention:
            self.attn = AttentionBlock(num_channels=out_channels, num_heads=1, d_k=out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, time_emb):
        """
        x: [batch_size, in_channels, seq_len]
        time_emb: [batch_size, time_emb_dim]
        """
        # Apply residual block
        x = self.res(x, time_emb)
        # Apply attention block if specified
        x = self.attn(x)
        return x
    

class MiddleBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, kernel_size=3):
        super(MiddleBlock, self).__init__()
        
        self.res1 = ResidualBlock(in_channels, in_channels, time_emb_dim)#, kernel_size)
        self.attn = AttentionBlock(num_channels=in_channels, num_heads=1, d_k=in_channels)
        self.res2 = ResidualBlock(in_channels, in_channels, time_emb_dim)#, kernel_size)
        
    def forward(self, x, time_emb):
        """
        x: [batch_size, in_channels, seq_len]
        time_emb: [batch_size, time_emb_dim]
        """
        # Apply residual block
        x = self.res1(x, time_emb)
        # Apply attention block
        x = self.attn(x, time_emb)
        # Apply second residual block
        x = self.res2(x, time_emb)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, has_attention=False):
        super(UpBlock, self).__init__()
        
        # The input channels are the sum of in_channels and out_channels because we concatenate with the first part of U-net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)#, kernel_size)
        if has_attention:
            self.attn = AttentionBlock(num_channels=out_channels, num_heads=1, d_k=out_channels)
        else:
            self.attn = nn.Identity()
        
    def forward(self, x, time_emb):
        """
        x: [batch_size, in_channels, seq_len]
        time_emb: [batch_size, time_emb_dim]
        """
        # Apply residual block
        x = self.res(x, time_emb)
        # Apply attention block if specified
        x = self.attn(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling layer using transposed convolution.
    """
    def __init__(self, num_channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose1d(num_channels, num_channels, kernel_size=4, stride=2, padding=1)#, output_padding=1)
        
    def forward(self, x, time_emb=None):
        """
        x: [batch_size, in_channels, seq_len]
        """
        _ = time_emb  # Unused, but kept for compatibility
        return self.conv(x)  # [batch_size, out_channels, new_seq_len]


class Downsample(nn.Module):
    """
    Downsampling layer using strided convolution.
    """
    def __init__(self, num_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x, time_emb=None):
        """
        x: [batch_size, in_channels, seq_len]
        """
        _ = time_emb  # Unused, but kept for compatibility
        return self.conv(x)  # [batch_size, out_channels, new_seq_len]
    

class UNet(nn.Module):
    """
    U-Net architecture for time series data.
    """
    def __init__(self, num_features, seq_len, num_channels, ch_mults=(1, 2, 2, 4), 
                 is_attn=(False, False, False, False), num_blocks=2, make_divisible_by_2=True):
        super(UNet, self).__init__()
        self.n_resolutions = len(ch_mults)
        self.num_features = num_features
        self.seq_len = seq_len
        self.make_divisible_by_2 = make_divisible_by_2
        
        self.time_emb = TimeEmbedding(time_emb_dim=num_channels*4)  # Time embedding dimension is 4 times the number of channels
        self.proj_window = nn.Conv1d(num_features, num_channels, kernel_size=3, padding=1)  # Project input window to initial channel size

        down = []
        
        out_channels = in_channels = num_channels
        
        for i in range(self.n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(num_blocks):
                down.append(DownBlock(in_channels, out_channels, time_emb_dim=num_channels*4, has_attention=is_attn[i]))
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down) # Encoder
        
        self.middle = MiddleBlock(in_channels, time_emb_dim=num_channels*4) # Encoder (like mean and std of vae)
        
        
        # TODO: CHECK IF CORRECT
        #from copy import deepcopy
        #self.encoder = nn.ModuleList([deepcopy(down), deepcopy(self.middle)])  # For potential use in other contexts
        
        up = []
        
        in_channels = out_channels
        
        for i in reversed(range(self.n_resolutions)):
            out_channels = in_channels
            for _ in range(num_blocks):
                up.append(UpBlock(in_channels, out_channels, time_emb_dim=num_channels*4, has_attention=is_attn[i]))
                
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, time_emb_dim=num_channels*4, has_attention=is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)
        
        self.norm = nn.GroupNorm(1, num_channels)  # Normalization layer for the final output
        self.act = nn.Tanh()  # Activation function for the final output (time series data typically normalized to [-1, 1])
        self.final_conv = nn.Conv1d(num_channels, num_features, kernel_size=3, padding=1)  # Final convolution to project back to original feature size

        # TODO: CHECK IF CORRECT
        #self.decoder = nn.ModuleList([deepcopy(self.up), deepcopy(self.norm), deepcopy(self.final_conv)])  # For potential use in other contexts
        
    def _get_encoder_weights_names(self):
        names = []
        
        for name, _ in self.proj_window.state_dict().items():
            names.append(f"unet.proj_window.{name}")
        
        for name, _ in self.time_emb.state_dict().items():
            names.append(f"unet.time_emb.{name}")
        
        for name, _ in self.down.state_dict().items():
            names.append(f"unet.down.{name}")
            
        for name, _ in self.middle.state_dict().items():
            names.append(f"unet.middle.{name}")
        return names    
            

    def _get_decoder_weights_names(self): # CHECK
        names = []
        for name, _ in self.up.state_dict().items():
            names.append(f"unet.up.{name}")
        
        for name, _ in self.norm.state_dict().items():
            names.append(f"unet.norm.{name}")
            
        for name, _ in self.final_conv.state_dict().items():
            names.append(f"unet.final_conv.{name}")
        return names
        

    def _pad_input(self, x):
        target_length = 2 ** (self.n_resolutions - 1) * 2  # Ensure the length is divisible by 2^n_resolutions
        if self.seq_len < target_length:
            padding_length = target_length - self.seq_len
            x = F.pad(x, (0, padding_length), mode='constant', value=0)
        elif self.seq_len % target_length != 0:
            padding_length = target_length - (self.seq_len % target_length)
            x = F.pad(x, (0, padding_length), mode='constant', value=0)
        return x
    
    def forward(self, x, time):
        """
        x: [batch_size, seq_len, num_features]
        time: [batch_size] (time steps)
        """
        # Reshape x to [batch_size, num_features, seq_len] for 1D convolutions
        x = x.permute(0, 2, 1)  # [batch_size, num_features, seq_len]
        
        # Padding to ensure the input length is divisible by 2^n_resolutions
        if self.make_divisible_by_2:    
            x = self._pad_input(x)
        else:
            assert x.size(2) % (2 ** self.n_resolutions) == 0, f"Input sequence length ({self.seq_len}) must be divisible by 2^n_resolutions ({2 ** self.n_resolutions})"
        
        # Project input window to initial channel size
        x = self.proj_window(x)
        # Get time embeddings
        time_emb = self.time_emb(time)
        h = [x]
        
        for down_block in self.down:    
            x = down_block(x, time_emb)
            h.append(x)
            
        x = self.middle(x, time_emb)
        
        for up_block in self.up:
            if isinstance(up_block, Upsample):
                x = up_block(x, time_emb)
            else:
                s = h.pop()  # Skip connection from downsampling path
                x = torch.cat((x, s), dim=1)
                x = up_block(x, time_emb)
        
        # Normalize and activate the final output
        x = self.norm(x)
        x = self.final_conv(x)
        
        # Return to original shape, brfore padding
        x = x[:, :, :self.seq_len]  # Remove padding if any
        # Reshape back to [batch_size, seq_len, num_features]
        x = x.permute(0, 2, 1)
        return x
  
    
class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for time series generation using UNet.
    """
    def __init__(self, unet_configs, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):#, device=None):
        super().__init__()
        self.unet = UNet(**unet_configs)
       
        self.num_timesteps = num_timesteps # T
       
        # Linear beta schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_timesteps)) # sigma^2 
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0)) # alpha_hat
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / self.alphas))
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.], device=self.betas.device, dtype=self.betas.dtype),
            self.alphas_cumprod[:-1]
        ], dim=0)
        self.register_buffer('posterior_variance', self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def _q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (add noise) at timestep t
        x_start: [batch, seq_len, num_features]
        t: [batch]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def _p_sample(self, x, t):
        """
        Sample from p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * self.unet(x, t))
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_var = self.posterior_variance[t].view(-1, 1, 1)
            return model_mean + torch.sqrt(posterior_var) * noise

    def p_losses(self, x_start, t, noise=None):
        """
        Compute the DDPM loss (MSE between predicted and true noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start, t, noise)
        noise_pred = self.unet(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    
    def get_encoder_weights_names(self):
        return self.unet._get_encoder_weights_names()
    
    def get_decoder_weights_names(self):
        return self.unet._get_decoder_weights_names()
        
    
    def forward(self, x):
        """
        Compute the loss for a batch.
        x: [batch, seq_len, num_features]
        """
        batch_size = x.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t)
    
    def fit(self, train_loader, optimizer, criterion, epochs, verbose=True, **kwargs):
        """
        Training loop for DDPM, similar to LSTM_VAE's fit method.
        Args:
            train_loader: DataLoader yielding batches of [batch, seq_len, num_features]
            optimizer: optimizer for model parameters
            epochs: number of epochs
            verbose: print progress if True
        """
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            #for batch in train_loader:
            # Use tqdm for progress bar
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                optimizer.zero_grad()
                loss = self(x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if verbose:
                logger = logging.getLogger(__name__)
                logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.5f}")
                logger.info("-----------------------------------------------------")
        #avg_loss = avg_loss.detach().cpu().item()
        return {"avg_loss": avg_loss}  # Return the last average loss
    
    @torch.no_grad()
    def validation_loss(self, val_loader, criterion=None):
        """Average denoising loss on a validation loader, used for early stopping."""
        self.eval()
        total_loss = 0.0
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            total_loss += self(x).item()
        return total_loss / max(1, len(val_loader))

    def gen_seq(self, input_seq, num_timesteps, **kwargs):
        """
        Function that reconstructs the window of the input sequence and computes the anomaly score
        
        Arguments:
            model: the trained model
            input_seq: the input sequence to reconstruct
            **kwags: additional arguments (not used)
        Returns:
            decoded_data: the reconstructed data
            rec_error: the reconstruction error
        """
        self.unet.eval()
        # Add noise, them gemerate a sample. Loss is the MSE between the input sequence and the generated sample
        device = next(self.parameters()).device
        shape = input_seq.shape
        #decoded_data = self._q_sample(input_seq, num_timesteps)
        t = torch.full((shape[0],), num_timesteps - 1, device=device, dtype=torch.long)
        decoded_data = self._q_sample(input_seq, t)
        
        # Use tqdm for progress bar
        #pbar = tqdm(range(num_timesteps), desc="Reconstructing sequence", unit="iteration")
        for i in reversed(range(num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            decoded_data = self._p_sample(decoded_data, t)
        #    pbar.update(1)
        
        anomaly_score = torch.mean((decoded_data - input_seq)**2, dim=2).unsqueeze(-1) # [1, seq_len]
        return decoded_data, anomaly_score  # [batch_size, seq_len, num_feat], [batch_size, seq_len, 1]
        
    def save_model(self, path):
        # Create a directory (without the name of the file)
        pth = '/'.join(path.split('/')[:-1])
        os.makedirs(pth, exist_ok=True)
        # Save the generator and discriminator separately
        torch.save({'unet': self.unet.state_dict()}, path)

    def load_model(self, path):
        # Load the weights for the generator and discriminator
        checkpoint = torch.load(path)
        self.unet.load_state_dict(checkpoint['unet'])    