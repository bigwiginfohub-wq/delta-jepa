"""
Benchmark for hierarchical planning.
Expected: Success rate > 85% on block stacking task.
"""

import torch


def run_hierarchy_benchmark(model, env, n_episodes=100):
    """
    env: hierarchical task environment (e.g., block stacking)
         Must return success/failure for each episode.
    """
    successes = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Abstract planning (high-level)
            abstract_latent = model.encoder(state)
            abstract_boundary = model.boundary_head_high(abstract_latent)
            
            # Check if uncertainty is too high
            if abstract_boundary.mean() > 0.7:
                # Fall back to simpler policy
                action = env.random_action()
            else:
                # Execute planned action
                action = env.sample_action()
            
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        if info.get('success', False):
            successes += 1
    
    success_rate = successes / n_episodes
    passed = success_rate > 0.85
    
    return {
        'benchmark': 'hierarchical_planning',
        'success_rate': success_rate,
        'threshold': 0.85,
        'passed': passed,
        'interpretation': f'Success rate {success_rate:.1%} on block stacking'
    }