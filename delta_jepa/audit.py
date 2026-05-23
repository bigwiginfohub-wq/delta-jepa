from dataclasses import dataclass
from typing import List, Dict, Any
import torch


@dataclass
class AuditReport:
    """Audit report for a Delta-JEPA prediction."""
    prediction: torch.Tensor
    h3_warrant: torch.Tensor
    boundary_statement: torch.Tensor
    friction_score: float
    mcl_coefficient: float
    prediction_horizon: int
    drift_detected: bool
    confidence_calibration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_shape': list(self.prediction.shape),
            'h3_warrant_norm': float(self.h3_warrant.norm().item()),
            'boundary_uncertainty': float(self.boundary_statement.mean().item()),
            'friction_score': self.friction_score,
            'mcl_coefficient': self.mcl_coefficient,
            'prediction_horizon': self.prediction_horizon,
            'drift_detected': self.drift_detected,
            'confidence_calibration': self.confidence_calibration
        }
    
    def __str__(self) -> str:
        return (f"AuditReport(\n"
                f"  friction={self.friction_score:.3f},\n"
                f"  mcl={self.mcl_coefficient:.3f},\n"
                f"  uncertainty={float(self.boundary_statement.mean().item()):.3f},\n"
                f"  drift={self.drift_detected}\n)")


def generate_audit_report(
    prediction: torch.Tensor,
    h3_warrant: torch.Tensor,
    boundary: torch.Tensor,
    friction: float,
    mcl: float,
    horizon: int,
    friction_threshold: float = 0.3
) -> AuditReport:
    """Generate an audit report for a prediction."""
    drift_detected = friction > friction_threshold
    calibration = 1.0 - abs(mcl - 0.5) * 2  # Simple calibration metric
    
    return AuditReport(
        prediction=prediction,
        h3_warrant=h3_warrant,
        boundary_statement=boundary,
        friction_score=friction,
        mcl_coefficient=mcl,
        prediction_horizon=horizon,
        drift_detected=drift_detected,
        confidence_calibration=calibration
    )