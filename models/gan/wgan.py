import os
import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, num_feat, seq_len, 
                 hidden_dims_generator, latent_dim
    ):
        
        super(Generator, self).__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.hidden_dims_generator = hidden_dims_generator
        self.latent_dim = latent_dim
        
        # Project from (batch_size, latent_dim) to (batch_size, hidden_dims) to feed into the hidden state of the first LSTM
        self.lattent_to_hidden_h = nn.Linear(latent_dim, hidden_dims_generator[0])
        self.lattent_to_hidden_c = nn.Linear(latent_dim, hidden_dims_generator[0])
        

        # Project from (batch_size, latent_dim) to (batch_size, seq_len*latent_dim) -> then reshape to (batch_size, seq_len, latent_dim)
        self.projection_layer = nn.Linear(latent_dim, latent_dim*seq_len)
        
        generator_layers = []
        generator_in_dim = latent_dim
        for generator_out_dim in hidden_dims_generator:
            generator_layers.append(nn.LSTM(generator_in_dim, generator_out_dim, batch_first=True))
            generator_in_dim = generator_out_dim

        self.generator_lstm_layers = nn.Sequential(*generator_layers)
        self.output_layer = nn.Linear(generator_in_dim, num_feat)
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        batch_size = z.size(0)
        
        x = self.projection_layer(z) # (batch_size, latent_dim) -> (batch_size, seq_len*latent_dim)
        x = x.reshape(-1, self.seq_len, self.latent_dim) # (batch_size, seq_len, latent_dim)

        for i, layer in enumerate(self.generator_lstm_layers):
            # If LSTM layer
            if isinstance(layer, nn.LSTM):
                if i == 0:
                    h_0 = self.lattent_to_hidden_h(z).unsqueeze(0) # (num_layers==1, batch_size, hidden_dims_decoder[0])
                    c_0 = self.lattent_to_hidden_c(z).unsqueeze(0) # (num_layers==1, batch_size, hidden_dims_decoder[0])
                    x, (h, c) = layer(x, (h_0, c_0))
                else:
                    x, (h, c) = layer(x)
        
        x_gen = self.output_layer(x) # (batch_size, seq_len, hidden_dims_generator) -> (batch_size, seq_len, num_feat)
        #return x_gen, h
        x_gen = self.tanh(x_gen) # Apply tanh activation to the output
        return x_gen, x  # Return the generated sequence and the intermediate features (x) for anomaly scoring
    

class Critic(nn.Module): # Replace Discriminator with Critic
    def __init__(self, num_feat, seq_len, hidden_dims_discriminator):
        super(Critic, self).__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.hidden_dims_discriminator = hidden_dims_discriminator

        # Initial hidden states of the discriminator (trainable parameters)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dims_discriminator[0]), requires_grad=True)
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_dims_discriminator[0]), requires_grad=True)

        discriminator_layers = []
        discriminator_in_dim = num_feat
        for discriminator_out_dim in hidden_dims_discriminator:
            discriminator_layers.append(nn.LSTM(discriminator_in_dim, discriminator_out_dim, batch_first=True))
            discriminator_in_dim = discriminator_out_dim

        self.discriminator_lstm_layers = nn.Sequential(*discriminator_layers)
        
        # Each timestep of the sequence is classified as real or fake
        # The output is a sequence of shape (batch_size, seq_len, 1)
        self.output_layer = nn.Linear(discriminator_in_dim, 1)   
    
    def forward(self, x):
        batch_size = x.size(0)
        for i, layer in enumerate(self.discriminator_lstm_layers):
            # If LSTM layer
            if isinstance(layer, nn.LSTM):
                if i == 0:
                    h_0 = self.h0.expand(-1, batch_size, -1).contiguous() # from shape (1, 1, hidden_dim) to (1, batch_size, hidden_dim)
                    c_0 = self.c0.expand(-1, batch_size, -1).contiguous()
                    x, (h, c) = layer(x, (h_0, c_0)) # x: (batch_size, seq_len, hidden_dim), h, c: (1, batch_size, hidden_dim)
                else:
                    x, (h, c) = layer(x) # x: (batch_size, seq_len, hidden_dim), h: (1, batch_size, hidden_dim)
    
        predictions = self.output_layer(x) # (batch_size, seq_len, hidden_dims_discriminator) -> (batch_size, seq_len, 1)
        return predictions, x #h # Critic does not use sigmoid activation, as it is not a binary classification task


class WGAN(nn.Module):
    """https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/"""
    def __init__(self, num_feat, seq_len, 
        hidden_dims_generator, hidden_dims_discriminator, 
        latent_dim
    ):
    
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.hidden_dims_generator = hidden_dims_generator
        self.hidden_dims_discriminator = hidden_dims_discriminator
        self.latent_dim = latent_dim
        
        # Generator (like DECODER)
        self.generator = Generator(num_feat=num_feat, 
                           seq_len=seq_len, 
                           hidden_dims_generator=hidden_dims_generator, 
                           latent_dim=latent_dim
                           )
        
        # Discriminator (like ENCODER)
        self.discriminator = Critic(num_feat=num_feat, 
                               seq_len=seq_len, 
                               hidden_dims_discriminator=hidden_dims_discriminator 
                               )

    def forward(self, x):
        # Empty
        pass
    
    def get_encoder_weights_names(self):
        names = []
        for name, _ in self.discriminator.named_parameters():
            names.append(f'discriminator.{name}')
        return names

    def get_decoder_weights_names(self):
        names = []
        for name, _ in self.generator.named_parameters():
            names.append(f'generator.{name}')
        return names
    
    def fit(self, train_loader, optimizers, criterions, epochs, verbose=True,
            n_critic=5, lambda_gp=10, **kwargs):

        """
        Args of WGAN:
            criterions: Dictionary with loss functions (not used in WGAN, can be empty)
            n_critic: Number of critic iterations per generator iteration
            lambda_gp: Gradient penalty lambda
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Training WGAN-GP with n_critic: {n_critic}, lambda_gp: {lambda_gp}")
        device = next(self.parameters()).device
        self.generator.to(device)
        self.discriminator.to(device)
        self.generator.train()
        self.discriminator.train()

        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            i = 0
            #for i, batch in enumerate(train_loader):
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=not verbose):
                i += 1
                # Real data
                x_real = batch[0] 
                batch_size = x_real.size(0)
                
                # ---------------------
                #  Train Critic (instead of Discriminator)
                # ---------------------
                optimizers['Discriminator'].zero_grad()
                
                # Sample noise as generator input
                z = torch.randn(batch_size, self.latent_dim).to(device)
                
                # Generate a batch of data
                x_fake, _ = self.generator(z)
                
                # Critic real and fake scores
                real_validity, _ = self.discriminator(x_real)
                fake_validity, _ = self.discriminator(x_fake.detach()) # Detach to avoid training the generator

                # Flatten the validity scores
                #real_validity = real_validity.view(real_validity.size(0), -1)
                #fake_validity = fake_validity.view(fake_validity.size(0), -1)  
                
                # Wasserstein loss with gradient penalty
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                gradient_penalty = self._compute_gradient_penalty(x_real, x_fake.detach(), device)
                d_loss += lambda_gp * gradient_penalty

                d_loss.backward()
                optimizers['Discriminator'].step()

                d_losses.append(d_loss.item())
                
                # Train the generator every n_critic steps
                if i % n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizers['Generator'].zero_grad()
                    
                    # Generate a batch of data
                    gen_data, _ = self.generator(z) # TODO: MB reundant, use x_fake instead
                    
                    # Adversarial loss (minimize critic's loss)
                    fake_validity, _ = self.discriminator(gen_data)
                    
                    # Flatten the validity scores
                    #fake_validity = fake_validity.view(fake_validity.size(0), -1) 
                    
                    g_loss = -torch.mean(fake_validity)
                    
                    g_loss.backward()
                    optimizers['Generator'].step()
                    
                    g_losses.append(g_loss.item())
            
            avg_d_loss = sum(d_losses)/len(d_losses)
            avg_g_loss = sum(g_losses)/len(g_losses)
            # Log training progress
            if verbose:
                logger.info(f"Epoch {epoch+ 1 } completed. Average Generator Loss: {avg_g_loss:.4f}, Average Critic (D) Loss: {avg_d_loss:.4f}")
                logger.info("-----------------------------------------------------")       
        
        #avg_g_loss = avg_g_loss.detach().cpu().item()
        #avg_d_loss = avg_d_loss.detach().cpu().item()
        return { 
                "avg_loss_g": avg_g_loss, #gener
                "avg_loss_c": avg_d_loss #critic
        }
    
    def _compute_gradient_penalty(self, real_samples, fake_samples, device):
        
        """
        Calculates the gradient penalty for WGAN-GP (Source: https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py#L296)
        
        Args:
            real_samples: Real data
            fake_samples: Generated data
            device: Device to use
            
        Returns:
            Gradient penalty
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
        # Get random interpolation between real and fake samples
        #interpolated  = (alpha * real_samples + ((1 - alpha) * fake_samples))#.requires_grad_(True)
        #interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        
        # FIX
        interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        
        # Calculate critic scores on interpolated samples
        #d_interpolates, _ = self.discriminator(interpolated )
        
        # FIX
        # IMPORTANT: disable cudnn only for the critic forward used here so that
        # PyTorch uses the autograd LSTM implementation which supports double backward.
        with torch.backends.cudnn.flags(enabled=False):
            d_interpolates, _ = self.discriminator(interpolated )
        
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated ,
            #grad_outputs=torch.ones_like(d_interpolates).to(device),
            # FIX
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] # (shape: (batch_size, seq_len, num_feat))
        #print(f"Gradients shape: {gradients.shape}")
        # Calculate gradient penalty
        #gradients = gradients.view(gradients.size(0), -1) # Can complain if gradients is not 2D
        #gradients = gradients.reshape(gradients.size(0), -1)  # Flatten the gradients
        # FIX
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    @torch.no_grad()
    def validation_loss(self, val_loader, criterion=None):
        """
        Absolute Wasserstein-1 estimate |E[f(x)] - E[f(G(z))]| on a validation
        loader, used as a training-stability proxy for early stopping.
        """
        self.eval()
        device = next(self.parameters()).device
        total = 0.0
        for batch in val_loader:
            x_real = batch[0]
            z = torch.randn(x_real.size(0), self.latent_dim).to(device)
            x_fake, _ = self.generator(z)
            real_validity, _ = self.discriminator(x_real)
            fake_validity, _ = self.discriminator(x_fake)
            total += torch.abs(torch.mean(real_validity) - torch.mean(fake_validity)).item()
        return total / max(1, len(val_loader))

    def _anomaly_score(self, x, x_gen, lmd=0.1):
        device = next(self.parameters()).device
        
        # x/x_gem: (batch_size, seq_len, num_features)
        residual_loss = torch.mean((x-x_gen)**2, dim=2).unsqueeze(2) # Residual Loss
        
        #residual_loss = torch.mean(
        #    torch.sum(
        #        torch.abs(x-x_gen), dim=1
        #        #torch.square(x-x_gen), dim=1
        #    ) # Take average over the batch and last dimension
        #) # Residual Loss
        
        # x_feature and x_feature_gen are rich intermediate feature representations for real and generated data
        #_, x_feature = self.discriminator(x.to(device)) # x_feature.shape = (num_layers==1, batch_size, 128)
        #_, x_feature_gen = self.discriminator(x_gen.to(device)) 
        # FIX
         # Feature loss — MUST disable gradient flow through discriminator
        with torch.no_grad():
            _, x_feature = self.discriminator(x)
            _, x_feature_gen = self.discriminator(x_gen)
            
        # # x_feature/x_feature_gen: (batch_size, seq_len, hidden_dim)
        discrimination_loss = torch.mean((x_feature-x_feature_gen)**2, dim=2).unsqueeze(2) # Discrimination Loss
        #discrimination_loss = torch.mean(
        #    torch.sum(
        #        torch.abs(x_feature-x_feature_gen), dim=1
        #        # torch.square(x_feature-x_feature_gen), dim=1
        #    ), # Take average over the batch and last dimension
        #) # Discrimination Loss
        
        score = (1-lmd)*residual_loss.to(device) + lmd*discrimination_loss
        return score      

    def gen_seq(self, input_seq, num_steps=1000, lr=1e-2, lmd=0.3, **kwargs):
        """
        Function that reconstructs the window of the input sequence and computes the anomaly score
        
        Arguments:
            model: the trained model
            input_seq: the input sequence to reconstruct
            num_steps: number of steps to reconstruct
            lr: learning rate for the optimizer
            lmd: lambda parameter for the anomaly score
            **kwags: additional arguments (not used)
        Returns:
            decoded_data: the reconstructed data
            rec_error: the reconstruction error
        """
        self.generator.eval()
        self.discriminator.eval()
        device = next(self.parameters()).device
        
        #z = torch.zeros(input_seq.shape[0], self.seq_len, self.latent_dim).to(device)
        z = torch.zeros(input_seq.shape[0], self.latent_dim).to(device)
        z = nn.init.normal_(z, mean=0, std=1)  # Initialize z with normal distribution
        z.requires_grad = True  # Explicitly set requires_grad=True after initialization
        
        z_optimizer = Adam([z], lr=lr)
        
        #anomaly_score = []
        # Create a progress bar with load bar
        #pbar = tqdm(range(num_steps), desc="Reconstructing sequence", unit="iteration")
        for i in range(num_steps):
            z_optimizer.zero_grad()
            # FIX
            with torch.backends.cudnn.flags(enabled=False):   # allow gradients even in eval mode
                x_gen, _ = self.generator(z)
            
            anomaly_score = self._anomaly_score(input_seq, x_gen, lmd)
            loss = torch.mean(anomaly_score) # Mean over the batch and sequence length
            #loss = torch.sum(anomaly_score) # Sum over the batch and sequence length
            loss.backward()
            
            z_optimizer.step()
            #pbar.set_postfix(loss=loss.item())
            # Update the progress bar with the current loss
            #pbar.update(1)
        
        # Append the loss to the list
        #anomaly_score.append(loss_window)
        # Convert the predictions to numpy
        decoded_data, _ = self.generator(z)
        return decoded_data, anomaly_score # [1, seq_len, num_feat], [1, seq_len, 1]
        
    def save_model(self, path):
        # Create a directory (without the name of the file)
        pth = '/'.join(path.split('/')[:-1])
        os.makedirs(pth, exist_ok=True)
        # Save the generator and discriminator separately
        torch.save({'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict()}, path)

    def load_model(self, path):
        # Load the weights for the generator and discriminator
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])