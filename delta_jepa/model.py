import torch
import torch.nn as nn
from .heads import H3Head, BoundaryHead, MCLHead
from .audit import generate_audit_report, AuditReport
from .loss import friction_score


class JEPAEncoder(nn.Module):
    """Simplified JEPA encoder for demonstration."""
    def __init__(self, latent_dim: int, input_dim: int = 2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """Simplified JEPA predictor for demonstration."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
    
    def forward(self, z):
        return self.predictor(z)


class DeltaJEPA(nn.Module):
    """
    Delta-JEPA: Auditable World Model with Governance Layer.
    """
    def __init__(
        self,
        latent_dim: int = 512,
        input_dim: int = 2048,
        warrant_dim: int = 128,
        boundary_dim: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # JEPA components
        self.encoder = JEPAEncoder(latent_dim, input_dim)
        self.predictor = JEPAPredictor(latent_dim)
        self.target_encoder = JEPAEncoder(latent_dim, input_dim)
        
        # Initialize target encoder with same weights
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Delta audit heads
        self.h3_head = H3Head(latent_dim, warrant_dim)
        self.boundary_head_low = BoundaryHead(latent_dim, boundary_dim)
        self.boundary_head_high = BoundaryHead(latent_dim, boundary_dim)
        self.mcl_head = MCLHead(latent_dim)
        
        # Momentum for target encoder
        self.momentum = 0.996
    
    @torch.no_grad()
    def update_target_encoder(self):
        """Momentum update for target encoder."""
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * param.data
    
    def forward(self, x, future_steps: int = 10):
        """
        Forward pass with audit.
        Returns: (prediction, audit_report)
        """
        # Encode current state
        z_current = self.encoder(x)
        
        # Generate predictions
        predictions = []
        z_t = z_current
        for _ in range(future_steps):
            z_t = self.predictor(z_t)
            predictions.append(z_t)
        
        # Final prediction
        z_pred = predictions[-1] if predictions else z_current
        
        # Target encoding (for training, use target encoder)
        z_target = self.target_encoder(x)
        
        # Delta audit
        warrant = self.h3_head(z_current)
        boundary_low = self.boundary_head_low(z_current)
        boundary_high = self.boundary_head_high(z_target)
        mcl = self.mcl_head(z_pred, z_target)
        
        # Compute friction
        friction = friction_score(predictions)
        
        # Generate audit report
        audit = generate_audit_report(
            prediction=z_pred,
            h3_warrant=warrant,
            boundary=boundary_low,
            friction=friction.item() if torch.is_tensor(friction) else friction,
            mcl=mcl.mean().item(),
            horizon=future_steps,
            friction_threshold=0.3
        )
        
        return {
            'prediction': z_pred,
            'target': z_target,
            'warrant': warrant,
            'boundary_low': boundary_low,
            'boundary_high': boundary_high,
            'mcl': mcl,
            'predictions': predictions,
            'audit': audit
        }