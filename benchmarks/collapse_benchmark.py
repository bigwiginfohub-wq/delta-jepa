"""
Benchmark for representation collapse.
Expected: Delta-JEPA diversity score > 0.8
"""

import torch
from delta_jepa import DeltaJEPA


def compute_diversity_score(model, dataloader):
    """Higher = more diverse representations = less collapse."""
    warrants = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            warrants.append(outputs['warrant'])
    
    all_warrants = torch.cat(warrants, dim=0)
    
    # Compute pairwise distances
    distances = torch.cdist(all_warrants, all_warrants, p=2)
    
    # Diversity score: normalized average distance
    max_distance = all_warrants.shape[-1] ** 0.5  # Maximum possible Euclidean distance
    diversity = distances.mean().item() / max_distance
    
    return diversity


def run_collapse_benchmark(model, dataloader):
    diversity = compute_diversity_score(model, dataloader)
    passed = diversity > 0.8
    
    return {
        'benchmark': 'representation_collapse',
        'diversity_score': diversity,
        'threshold': 0.8,
        'passed': passed,
        'interpretation': 'Low diversity (<0.3) = collapse likely. High diversity (>0.8) = healthy.'
    }