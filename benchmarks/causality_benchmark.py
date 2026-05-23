"""
Benchmark for correlation vs. causality.
Expected: MCL correlation with ground truth > 0.8.
"""

import torch
import numpy as np


def run_causality_benchmark(model, counterfactual_dataset):
    """
    counterfactual_dataset: dataset with (input, counterfactual, causal_strength)
    """
    mcl_predictions = []
    ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for item in counterfactual_dataset:
            x = item['input']
            causal_strength = item['causal_strength']  # Ground truth 0-1
            
            outputs = model(x)
            mcl = outputs['mcl'].mean().item()
            
            mcl_predictions.append(mcl)
            ground_truths.append(causal_strength)
    
    correlation = np.corrcoef(mcl_predictions, ground_truths)[0, 1]
    passed = correlation > 0.8
    
    return {
        'benchmark': 'causality_correlation',
        'correlation': correlation,
        'threshold': 0.8,
        'passed': passed,
        'interpretation': f'Correlation {correlation:.3f} between MCL and ground truth causality'
    }