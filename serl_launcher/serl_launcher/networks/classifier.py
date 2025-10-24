import torch
import torch.nn as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    """
    Binary classifier with optional pretrained encoder.
    
    Args:
        pretrained_encoder: Pretrained encoder module
        encoder: Main encoder module
        network: Classification network
        enable_stacking: Whether to enable frame stacking
    """
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        encoder: nn.Module,
        network: nn.Module,
        enable_stacking: bool = False,
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.encoder = encoder
        self.network = network
        self.enable_stacking = enable_stacking
        
        # Output layer will be created lazily
        self.output_layer = None
    
    def forward(
        self,
        x: torch.Tensor,
        train: bool = False,
        return_encoded: bool = False,
        classify_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            train: Whether in training mode
            return_encoded: If True, return encoded features only
            classify_encoded: If True, x is already encoded
            
        Returns:
            Encoded features (if return_encoded=True) or classification logits
        """
        if return_encoded:
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(x.shape) == 4:  # (T, H, W, C)
                    x = rearrange(x, "T H W C -> H W (T C)")
                if len(x.shape) == 5:  # (B, T, H, W, C)
                    x = rearrange(x, "B T H W C -> B H W (T C)")
            x = self.pretrained_encoder(x, train=train)
            return x
        
        x = self.encoder(x, train=train, is_encoded=classify_encoded)
        x = self.network(x, train=train)
        
        # Lazy initialization of output layer
        if self.output_layer is None:
            self.output_layer = nn.Linear(x.shape[-1], 1).to(x.device)
        
        x = self.output_layer(x).squeeze(-1)
        return x
