import torch
from typing import Optional

def lr_auto(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_steps: int = 1000,
    verbose: bool = False
) -> float:
    """
    Linear warmup + linear decay learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        step: Current training step
        total_steps: Total training steps
        lr_max: Peak learning rate
        lr_min: Minimum learning rate
        warmup_steps: Steps for warmup
        verbose: Print current LR
    
    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        lr = lr_min + (lr_max - lr_min) * step / warmup_steps
    else:
        decay_ratio = max(0.0, (step - warmup_steps) / (total_steps - warmup_steps))
        lr = lr_max * (1 - decay_ratio)
    
    lr = max(lr, lr_min)
    
    for group in optimizer.param_groups:
        group['lr'] = lr
    
    if verbose:
        print(f"[LR-AUTO] Step {step}/{total_steps} | LR: {lr:.2e}")
    
    return lr

def lr_auto_cosine(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_steps: int = 1000
) -> float:
    """
    Cosine decay with linear warmup.
    """
    if step < warmup_steps:
        lr = lr_min + (lr_max - lr_min) * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        lr = lr_min + (lr_max - lr_min) * cosine
    
    for group in optimizer.param_groups:
        group['lr'] = lr
    
    return lr
