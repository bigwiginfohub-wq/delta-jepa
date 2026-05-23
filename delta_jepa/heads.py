import torch
import torch.nn as nn
import torch.nn.functional as F


class H3Head(nn.Module):
    """
    H₃ Warrant Head — outputs falsification condition embedding.
    Solves representation collapse by requiring diverse falsifications.
    """
    def __init__(self, latent_dim: int, warrant_dim: int = 128):
        super().__init__()
        self.falsification_encoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, warrant_dim),
            nn.Tanh()  # Bounded embedding for stable diversity loss
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, latent_dim] -> warrant: [batch, warrant_dim]"""
        return self.falsification_encoder(z)


class BoundaryHead(nn.Module):
    """
    Boundary Statement Head — outputs uncertainty embedding.
    Higher values indicate higher uncertainty (what cannot be predicted).
    """
    def __init__(self, latent_dim: int, boundary_dim: int = 64):
        super().__init__()
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, boundary_dim),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, latent_dim] -> boundary: [batch, boundary_dim]"""
        return self.uncertainty_encoder(z)


class MCLHead(nn.Module):
    """
    MCL Coefficient Head — predicts causal strength (0.0–1.0).
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.confidence_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),  # Concatenates prediction + target
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """z_pred, z_target: [batch, latent_dim] -> mcl: [batch]"""
        combined = torch.cat([z_pred, z_target], dim=-1)
        return self.confidence_encoder(combined).squeeze(-1)