from typing import Optional, Tuple, Union

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR, ConstantLR


def make_optimizer(
    parameters,
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cosine_decay_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    return_lr_schedule: bool = False,
) -> Union[optim.Optimizer, Tuple[optim.Optimizer, LambdaLR]]:
    """
    Create an optimizer with optional learning rate scheduling and gradient clipping.
    
    Args:
        parameters: Model parameters to optimize
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        cosine_decay_steps: Number of steps for cosine decay (None for constant)
        weight_decay: Weight decay coefficient
        clip_grad_norm: Maximum gradient norm for clipping
        return_lr_schedule: Whether to return scheduler alongside optimizer
        
    Returns:
        Optimizer (and optionally scheduler)
    """
    # Create base optimizer
    if weight_decay is not None and weight_decay > 0:
        optimizer = optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.Adam(
            parameters,
            lr=learning_rate,
        )
    
    # Store clip_grad_norm for use during training
    optimizer.clip_grad_norm = clip_grad_norm
    
    # Create learning rate scheduler
    if cosine_decay_steps is not None:
        # Warmup + Cosine decay schedule
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0 to 1
                return step / max(1, warmup_steps)
            else:
                # Cosine decay from 1 to 0
                progress = (step - warmup_steps) / max(1, cosine_decay_steps - warmup_steps)
                progress = min(progress, 1.0)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif warmup_steps > 0:
        # Warmup + Constant schedule
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        # Constant schedule
        scheduler = None
    
    if return_lr_schedule:
        return optimizer, scheduler
    else:
        return optimizer


def clip_gradients(optimizer: optim.Optimizer, max_norm: Optional[float] = None):
    """
    Clip gradients by global norm if max_norm is set on the optimizer.
    
    Args:
        optimizer: The optimizer whose parameters' gradients to clip
        max_norm: Maximum gradient norm (default: use optimizer.clip_grad_norm)
    """
    if max_norm is None:
        max_norm = getattr(optimizer, 'clip_grad_norm', None)
    
    if max_norm is not None and max_norm > 0:
        # Get all parameters with gradients
        parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if len(parameters) > 0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class WarmupCosineScheduler(LambdaLR):
    """
    Learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(progress, 1.0)
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        super().__init__(optimizer, lr_lambda)
