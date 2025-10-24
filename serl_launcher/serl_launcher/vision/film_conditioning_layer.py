"""
FiLM (Feature-wise Linear Modulation) conditioning layer for PyTorch.

Adapted from: https://github.com/google-research/robotics_transformer/
"""

import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    """
    Applies FiLM conditioning to a convolutional feature map.
    
    FiLM modulates conv features using: output = features * (1 + gamma) + beta
    where gamma and beta are computed from the conditioning vector.
    """
    
    def __init__(self):
        super().__init__()
        # Layers will be created lazily on first forward pass
        self.add_projection = None
        self.mult_projection = None
    
    def forward(
        self,
        conv_filters: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            conv_filters: Convolutional features of shape [batch_size, height, width, channels]
                         or [batch_size, channels, height, width] (will be auto-detected)
            conditioning: Conditioning vector of shape [batch_size, conditioning_size]
            
        Returns:
            Modulated features of same shape as conv_filters
        """
        # Detect if input is channels-first or channels-last
        if conv_filters.ndim == 4:
            if conv_filters.shape[1] < conv_filters.shape[3]:
                # Likely channels-first (B, C, H, W)
                channels_first = True
                num_channels = conv_filters.shape[1]
            else:
                # Likely channels-last (B, H, W, C)
                channels_first = False
                num_channels = conv_filters.shape[-1]
        else:
            raise ValueError(f"Expected 4D tensor, got shape {conv_filters.shape}")
        
        # Lazy initialization of projection layers
        if self.add_projection is None:
            self.add_projection = nn.Linear(
                conditioning.shape[-1],
                num_channels,
            ).to(conv_filters.device)
            nn.init.zeros_(self.add_projection.weight)
            nn.init.zeros_(self.add_projection.bias)
            
            self.mult_projection = nn.Linear(
                conditioning.shape[-1],
                num_channels,
            ).to(conv_filters.device)
            nn.init.zeros_(self.mult_projection.weight)
            nn.init.zeros_(self.mult_projection.bias)
        
        # Project conditioning
        projected_cond_add = self.add_projection(conditioning)  # (B, C)
        projected_cond_mult = self.mult_projection(conditioning)  # (B, C)
        
        # Reshape for broadcasting
        if channels_first:
            # (B, C) -> (B, C, 1, 1)
            projected_cond_add = projected_cond_add[..., :, None, None]
            projected_cond_mult = projected_cond_mult[..., :, None, None]
        else:
            # (B, C) -> (B, 1, 1, C)
            projected_cond_add = projected_cond_add[..., None, None, :]
            projected_cond_mult = projected_cond_mult[..., None, None, :]
        
        # Apply FiLM modulation
        return conv_filters * (1 + projected_cond_add) + projected_cond_mult


if __name__ == "__main__":
    # Test the module
    import torch
    
    # Create test inputs
    x = torch.randn(2, 32, 32, 64)  # (B, H, W, C)
    z = torch.ones(2, 128)  # (B, conditioning_size)
    
    # Create and apply FiLM layer
    film = FilmConditioning()
    y = film(x, z)
    
    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {z.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with channels-first format
    x_cf = torch.randn(2, 64, 32, 32)  # (B, C, H, W)
    film_cf = FilmConditioning()
    y_cf = film_cf(x_cf, z)
    print(f"Channels-first output shape: {y_cf.shape}")
