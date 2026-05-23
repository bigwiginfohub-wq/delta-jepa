import torch
import torch.optim as optim
from tqdm import tqdm
from .loss import delta_jepa_loss
from .config import DeltaJEPAConfig


def train_delta_jepa(
    model,
    train_loader,
    val_loader=None,
    config: DeltaJEPAConfig = None
):
    if config is None:
        config = DeltaJEPAConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'audit_metrics': []}
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for batch in pbar:
            # Move batch to device
            x = batch.to(device)
            
            # Forward pass
            outputs = model(x, future_steps=config.prediction_horizon)
            
            # Compute Delta-JEPA loss
            losses = delta_jepa_loss(
                z_pred=outputs['prediction'],
                z_target=outputs['target'],
                warrant_embeddings=outputs['warrant'],
                boundary_low=outputs['boundary_low'],
                boundary_high=outputs['boundary_high'],
                predictions=outputs['predictions'],
                actuals=None,  # Would need ground truth future states
                mcl_pred=outputs['mcl'],
                mcl_gt=torch.ones_like(outputs['mcl']) * 0.5,  # Placeholder
                lambda_h3=config.lambda_h3,
                lambda_bs=config.lambda_bs,
                lambda_f=config.lambda_f,
                lambda_mcl=config.lambda_mcl,
                horizon=config.prediction_horizon
            )
            
            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            epoch_loss += losses['total'].item()
            
            # Update target encoder
            model.update_target_encoder()
            
            # Update progress bar
            pbar.set_postfix({'loss': losses['total'].item()})
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    outputs = model(x)
                    losses = delta_jepa_loss(...)  # Same as above
                    val_loss += losses['total'].item()
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
        
        # Record audit metrics
        history['audit_metrics'].append({
            'epoch': epoch,
            'friction': outputs['audit'].friction_score,
            'mcl': outputs['audit'].mcl_coefficient,
            'drift': outputs['audit'].drift_detected
        })
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, "
              f"Friction = {outputs['audit'].friction_score:.3f}, "
              f"MCL = {outputs['audit'].mcl_coefficient:.3f}")
    
    return model, history