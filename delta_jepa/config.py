from dataclasses import dataclass


@dataclass
class DeltaJEPAConfig:
    # Architecture
    latent_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    patch_size: int = 16
    
    # JEPA
    momentum: float = 0.996
    prediction_horizon: int = 10
    
    # Delta coefficients
    lambda_h3: float = 0.1
    lambda_bs: float = 0.05
    lambda_f: float = 0.2
    lambda_mcl: float = 0.15
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    num_workers: int = 4
    
    # Audit
    friction_window: int = 10
    boundary_consistency_weight: float = 1.0
    
    def __post_init__(self):
        assert 0 <= self.lambda_h3 <= 1
        assert 0 <= self.lambda_bs <= 1
        assert 0 <= self.lambda_f <= 1
        assert 0 <= self.lambda_mcl <= 1