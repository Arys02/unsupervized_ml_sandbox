from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, latent_dim=2, input_size=28 * 28):
        super().__init__()
        self.input_size: int = input_size
        self.latent_dim: int = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.fn_mu = nn.Linear(32, self.latent_dim)
        self.fn_logvar = nn.Linear(32, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    def forward_dec(self, x):
        return self.decoder(x)

    def forward_enc(self, x):

        x = self.encoder(x)
        mu = self.fn_mu(x)
        logvar = self.fn_logvar(x)

        sigma = torch.exp(0.5 * logvar)
        noise = torch.randn_like(logvar, device=logvar.device)

        z = mu + sigma * noise
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.forward_enc(x)

        return z, self.decoder(z), mu, logvar


def VAELoss(x, x_hat, mean, log_var, kl_weight=1, reconstruction_weight=1):
    pixel_mse = ((x - x_hat)**2)

    reconstruction_loss = pixel_mse.sum(axis=-1).mean()
    # reconstruction_loss = pixel_mse.mean()


    kl = (1 + log_var - mean**2 - torch.exp(log_var))

    kl_per_image = -0.5 * torch.sum(kl, dim=-1)


    kl_loss = torch.mean(kl_per_image)
    #print(reconstruction_loss, kl_loss)

    return reconstruction_loss * reconstruction_weight + kl_weight * kl_loss

