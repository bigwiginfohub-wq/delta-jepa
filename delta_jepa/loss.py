import torch
import torch.nn.functional as F


def h3_warrant_loss(warrant_embeddings: torch.Tensor, input_similarity: torch.Tensor = None) -> torch.Tensor:
    """
    L_h3 = -log(diversity) + similarity_penalty
    
    Encourages diverse falsification conditions.
    """
    # Diversity loss: maximize pairwise distance
    pairwise_distances = torch.cdist(warrant_embeddings, warrant_embeddings, p=2)
    diversity_loss = -torch.log(torch.mean(pairwise_distances) + 1e-8)
    
    # Similarity penalty: similar inputs should have similar warrants
    if input_similarity is not None:
        warrant_similarity = torch.cosine_similarity(
            warrant_embeddings.unsqueeze(1),
            warrant_embeddings.unsqueeze(0),
            dim=2
        )
        similarity_penalty = F.mse_loss(warrant_similarity, input_similarity)
    else:
        similarity_penalty = 0.0
    
    return diversity_loss + 0.1 * similarity_penalty


def boundary_statement_loss(boundary_low: torch.Tensor, boundary_high: torch.Tensor) -> torch.Tensor:
    """
    L_bs = consistency_loss(boundary_low, boundary_high) + entropy_penalty
    
    For hierarchical JEPA: low-level uncertainty must be >= high-level uncertainty.
    """
    # Consistency: low-level uncertainty cannot be less than high-level
    consistency_loss = F.relu(boundary_high - boundary_low).mean()
    
    # Entropy penalty: encourage meaningful uncertainty (not always 0 or 1)
    entropy = -(boundary_low * torch.log(boundary_low + 1e-8) + 
                (1 - boundary_low) * torch.log(1 - boundary_low + 1e-8)).mean()
    entropy_penalty = -0.01 * entropy
    
    return consistency_loss + entropy_penalty


def friction_score(predictions: list, actuals: list = None) -> torch.Tensor:
    """
    Compute friction score (0.0 = no divergence, 1.0 = complete divergence)
    """
    if not predictions:
        return torch.tensor(0.0)
    
    pred_tensor = torch.stack(predictions)  # [horizon, batch, latent_dim]
    
    if actuals is not None:
        actual_tensor = torch.stack(actuals)
        divergence = F.mse_loss(pred_tensor, actual_tensor, reduction='none').mean(dim=-1)
    else:
        # Temporal variance as proxy for divergence
        divergence = pred_tensor.var(dim=0).mean(dim=-1)
    
    friction = torch.sigmoid(divergence.mean())
    return friction


def friction_loss(predictions: list, actuals: list, horizon: int) -> torch.Tensor:
    """
    L_f = mean(friction) + drift_penalty
    """
    friction_scores = []
    for t in range(1, min(horizon, len(predictions)) + 1):
        pred_t = predictions[:t]
        actual_t = actuals[:t] if actuals else None
        friction = friction_score(pred_t, actual_t)
        friction_scores.append(friction)
    
    if not friction_scores:
        return torch.tensor(0.0)
    
    mean_friction = torch.mean(torch.stack(friction_scores))
    
    # Drift penalty: friction should not increase with prediction length
    if len(friction_scores) > 1:
        gradient = torch.tensor([float(i) for i in range(len(friction_scores))])
        drift = torch.cat(torch.gradient(torch.stack(friction_scores)))[0]
        drift_penalty = F.relu(drift).mean()
    else:
        drift_penalty = 0.0
    
    return mean_friction + 0.5 * drift_penalty


def mcl_loss(mcl_pred: torch.Tensor, mcl_ground_truth: torch.Tensor) -> torch.Tensor:
    """
    L_mcl = BCE + calibration_penalty
    """
    bce_loss = F.binary_cross_entropy(mcl_pred, mcl_ground_truth)
    
    # Simple expected calibration error
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (mcl_pred >= bin_boundaries[i]) & (mcl_pred < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = mcl_pred[in_bin].mean()
            avg_accuracy = mcl_ground_truth[in_bin].mean()
            ece += (in_bin.sum() / len(mcl_pred)) * torch.abs(avg_confidence - avg_accuracy)
    
    return bce_loss + 0.1 * ece


def delta_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    warrant_embeddings: torch.Tensor,
    boundary_low: torch.Tensor,
    boundary_high: torch.Tensor,
    predictions: list,
    actuals: list,
    mcl_pred: torch.Tensor,
    mcl_gt: torch.Tensor,
    lambda_h3: float = 0.1,
    lambda_bs: float = 0.05,
    lambda_f: float = 0.2,
    lambda_mcl: float = 0.15,
    horizon: int = 10
) -> dict:
    """
    Complete Delta-JEPA loss.
    """
    # JEPA prediction loss
    L_jepa = F.mse_loss(z_pred, z_target.detach())
    
    # Delta losses
    L_h3 = h3_warrant_loss(warrant_embeddings)
    L_bs = boundary_statement_loss(boundary_low, boundary_high)
    L_f = friction_loss(predictions, actuals, horizon)
    L_mcl = mcl_loss(mcl_pred, mcl_gt)
    
    # Combined
    total = (L_jepa 
             + lambda_h3 * L_h3
             + lambda_bs * L_bs
             + lambda_f * L_f
             + lambda_mcl * L_mcl)
    
    return {
        'total': total,
        'L_jepa': L_jepa,
        'L_h3': L_h3,
        'L_bs': L_bs,
        'L_f': L_f,
        'L_mcl': L_mcl
    }