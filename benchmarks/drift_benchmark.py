"""
Benchmark for invisible autoregressive error.
Expected: Friction score < 0.3, no catastrophic failure at 100 steps.
"""

import torch
from delta_jepa.loss import friction_score


def run_drift_benchmark(model, dataloader, max_horizon=100):
    model.eval()
    friction_scores = []
    failures = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch, future_steps=max_horizon)
            friction = outputs['audit'].friction_score
            friction_scores.append(friction)
            
            # Check for catastrophic failure (extremely high friction)
            if friction > 0.8:
                failures.append(True)
            else:
                failures.append(False)
    
    avg_friction = sum(friction_scores) / len(friction_scores)
    failure_rate = sum(failures) / len(failures)
    
    passed = (avg_friction < 0.3) and (failure_rate < 0.05)
    
    return {
        'benchmark': 'invisible_drift',
        'avg_friction': avg_friction,
        'failure_rate': failure_rate,
        'threshold_friction': 0.3,
        'threshold_failure': 0.05,
        'passed': passed,
        'interpretation': f'Average friction {avg_friction:.3f} across {max_horizon} steps'
    }