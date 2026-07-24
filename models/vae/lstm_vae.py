import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, num_feat, seq_len, 
                 hidden_dims_encoder, latent_dim
    ):
    
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dims_encoder = hidden_dims_encoder
        
        # Initial hidden states of the encoder (trainable parameters)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dims_encoder[0]), requires_grad=True)
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_dims_encoder[0]), requires_grad=True)
        
        encoder_layers = []
        encoder_in_dim = num_feat
        for encoder_out_dim in hidden_dims_encoder:
            encoder_layers.append(nn.LSTM(encoder_in_dim, encoder_out_dim, batch_first=True))
            encoder_in_dim = encoder_out_dim

        self.encoder_lstm_layers = nn.Sequential(*encoder_layers)
        
        # Latent mean and variance
        self.mean_layer = nn.Linear(encoder_out_dim, latent_dim)
        self.logvar_layer = nn.Linear(encoder_out_dim, latent_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        for i, layer in enumerate(self.encoder_lstm_layers):
            # If LSTM layer
            if isinstance(layer, nn.LSTM):
                if i == 0:
                    h_0 = self.h0.expand(-1, batch_size, -1).contiguous() # (num_layers==1, batch_size, hidden_dims_encoder[0])
                    c_0 = self.c0.expand(-1, batch_size, -1).contiguous() # (num_layers==1, batch_size, hidden_dims_encoder[0])            
                    x, (h, c) = layer(x, (h_0, c_0)) # x: (batch_size, seq_len, hidden_dim), h, c: (1, batch_size, hidden_dim)
                else:
                    x, (h, c) = layer(x) # x: (batch_size, seq_len, hidden_dim), h, c: (1, batch_size, hidden_dim)
        
        z_mean, z_logvar = self.mean_layer(h[-1]), self.logvar_layer(h[-1])     
        return z_mean, z_logvar
        
class Decoder(nn.Module):
    def __init__(self, num_feat, seq_len, 
                 hidden_dims_decoder, latent_dim
    ):
    
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dims_decoder = hidden_dims_decoder
        
        # Project from (batch_size, latent_dim) to (batch_size, hidden_dims) to feed into the hidden state of the first LSTM
        self.lattent_to_hidden_h = nn.Linear(latent_dim, hidden_dims_decoder[0])
        self.lattent_to_hidden_c = nn.Linear(latent_dim, hidden_dims_decoder[0])
        
        # Project from (batch_size, latent_dim) to (batch_size, seq_len*latent_dim) -> then reshape to (batch_size, seq_len, latent_dim)
        self.projection_layer = nn.Linear(latent_dim, latent_dim*seq_len)
        
        decoder_layers = []
        decoder_in_dim = latent_dim
        for decoder_out_dim in hidden_dims_decoder:
            decoder_layers.append(nn.LSTM(decoder_in_dim, decoder_out_dim, batch_first=True))
            decoder_in_dim = decoder_out_dim
        
        self.decoder_lstm_layers = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(decoder_in_dim, num_feat)
        self.tanh = nn.Tanh()
        
        
    def forward(self, z):
        batch_size = z.size(0)
        
        x = self.projection_layer(z) # (batch_size, latent_dim) -> (batch_size, seq_len*latent_dim)
        x = x.reshape(-1, self.seq_len, self.latent_dim) # (batch_size, seq_len, latent_dim)

        for i, layer in enumerate(self.decoder_lstm_layers):
            # If LSTM layer
            if isinstance(layer, nn.LSTM):
                if i == 0:
                    #h_0 = self.lattent_to_hidden(z).unsqueeze(0) # (num_layers==1, batch_size, hidden_dims_decoder[0])
                    #c_0 = torch.zeros(1, batch_size, self.hidden_dims_decoder[0]).to(z.device) # (1, batch_size, hidden_dims_decoder[0])
                    h_0 = self.lattent_to_hidden_h(z).unsqueeze(0) # (num_layers==1, batch_size, hidden_dims_decoder[0])
                    c_0 = self.lattent_to_hidden_c(z).unsqueeze(0) # (num_layers==1, batch_size, hidden_dims_decoder[0])
                    
                    x, (h, c) = layer(x, (h_0, c_0))
                else:
                    x, (h, c) = layer(x)
                    
        x_recon = self.output_layer(x) # (batch_size, seq_len, hidden_dims) -> (batch_size, seq_len, num_feat)
        #x_recon = self.tanh(x_recon) # CHECK WITH REMOVED
        return x_recon # (batch_size, seq_len, num_feat)


# Define the Beta-VAE model
class LSTM_VAE(nn.Module):
    def __init__(self, num_feat, seq_len, 
        hidden_dims_encoder, hidden_dims_decoder, 
        latent_dim
    ):
    
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.hidden_dims_encoder = hidden_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        
        # Encoder
        self.encoder = Encoder(num_feat=num_feat, 
                                seq_len=seq_len, 
                                hidden_dims_encoder=hidden_dims_encoder, 
                                latent_dim=latent_dim)    
        
        # Decoder
        self.decoder = Decoder(num_feat=num_feat,
                                seq_len=seq_len, 
                                hidden_dims_decoder=hidden_dims_decoder, 
                                latent_dim=latent_dim)
           
    
    def _reparameterization(self, mean, logvar):
        batch_size = mean.size(0)
        epsilon = torch.randn(batch_size, self.latent_dim).to(mean.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        return z # (batch_size, latent_dim)
    
    def get_encoder_weights_names(self):
        names = []
        for name, _ in self.encoder.named_parameters():
            names.append(f'encoder.{name}')
        return names

    def get_decoder_weights_names(self):
        names = []
        for name, _ in self.decoder.named_parameters():
            names.append(f'decoder.{name}')
        return names
    
    def forward(self, x):
        # Encode
        z_mean, z_logvar = self.encoder(x)
        z = self._reparameterization(z_mean, z_logvar)
        # Decode
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_logvar


    # Train function
    def fit(self, train_loader, optimizer, criterion, epochs, verbose=True, **kwargs):
        #device = next(self.parameters()).device
        self.train()
        
        for epoch in range(epochs):
            total_loss, total_rec_loss, total_kl_div = 0, 0, 0
            #for batch in train_loader:
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=not verbose):
                x = batch[0]
                x_recon, mu, logvar = self(x)
                loss, recon_loss, kl_div = criterion(x_recon, x, mu, logvar)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss#.item()
                total_rec_loss += recon_loss.item()
                total_kl_div += kl_div.item()
            avg_total_loss = total_loss / len(train_loader)
            avg_rec_loss = total_rec_loss / len(train_loader)
            avg_kl_div = total_kl_div / len(train_loader)
        
            if verbose:
                #print(f"Epoch {epoch + 1} completed. Average Loss: {avg_total_loss:.5f}, Average Recon Loss: {avg_rec_loss:.5f}, Average KL Div: {avg_kl_div:.5f}")
                #print("-----------------------------------------------------")
                
                logger = logging.getLogger(__name__)
                logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_total_loss:.5f}, Average Recon Loss: {avg_rec_loss:.5f}, Average KL Div: {avg_kl_div:.5f}")
                logger.info("-----------------------------------------------------")
        
        # Return a normal float instead of a tensor, and on CPU
        avg_total_loss = avg_total_loss.detach().cpu().item()
        return {"avg_loss": avg_total_loss} # Return the last average total loss  


    @torch.no_grad()
    def validation_loss(self, val_loader, criterion=None):
        """Average ELBO loss on a validation loader, used for early stopping."""
        if criterion is None:
            criterion = VaeLoss()
        self.eval()
        total_loss = 0.0
        for batch in val_loader:
            x = batch[0]
            x_recon, mu, logvar = self(x)
            loss, _, _ = criterion(x_recon, x, mu, logvar)
            total_loss += loss.item()
        return total_loss / max(1, len(val_loader))

    def gen_seq(self, input_seq, **kwargs):
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
        self.encoder.eval() # Set the model to evaluation mode
        self.decoder.eval()
        
        # Get the reconstructed windows
        decoded_data, _, _ = self(input_seq)
        # Get the reconstruction error
        anomaly_score = torch.mean((decoded_data - input_seq)**2, dim=2).unsqueeze(-1) # [1, seq_len]
        return decoded_data, anomaly_score # [batch_size, seq_len, num_feat], [batch_size, seq_len, 1]
    
    
    def save_model(self, path):
        # Create a directory (without the name of the file)
        pth = '/'.join(path.split('/')[:-1])
        os.makedirs(pth, exist_ok=True)
        # Save the model
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        # Load the weights from the path
        self.load_state_dict(torch.load(path))
        
        
class VaeLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        #self.recon_loss = nn.MSELoss()

    def forward(self, input, target, z_mean, z_log_var):
        seq_len = input.shape[1]
        # Reconstruction loss (normalized by the window length)
        recon_loss = torch.mean(
            torch.sum(
                F.mse_loss(input, target, reduction='none'), dim=(1, 2)
            )
        ) / seq_len
        # KL divergence loss (normalized by the window length)
        kl_loss = self.beta * torch.mean(
            torch.sum(
            -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)), dim=1
            )
        ) / seq_len
        # Total loss
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

