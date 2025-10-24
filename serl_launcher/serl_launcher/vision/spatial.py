"""Spatial operations for vision networks in PyTorch."""

from typing import Sequence, Callable
import torch
import torch.nn as nn


class SpatialLearnedEmbeddings(nn.Module):
    """
    Learned spatial embeddings for feature maps.
    
    Args:
        height: Height of the feature map
        width: Width of the feature map
        channel: Number of channels in the feature map
        num_features: Number of output features
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        num_features: int = 5,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        
        # Initialize kernel with LeCun normal initialization
        self.kernel = nn.Parameter(
            torch.empty(height, width, channel, num_features)
        )
        nn.init.normal_(self.kernel, std=(1.0 / channel) ** 0.5)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape [B, H, W, C] or [H, W, C]
            
        Returns:
            Embedded features of shape [B, num_features * channel] or [num_features * channel]
        """
        squeeze = False
        if len(features.shape) == 3:
            features = features.unsqueeze(0)
            squeeze = True
        
        batch_size = features.shape[0]
        
        # Compute weighted sum: (B, H, W, C, 1) * (1, H, W, C, num_features)
        features_expanded = features.unsqueeze(-1)  # (B, H, W, C, 1)
        kernel_expanded = self.kernel.unsqueeze(0)  # (1, H, W, C, num_features)
        
        # Element-wise multiplication and sum over H, W, C
        result = (features_expanded * kernel_expanded).sum(dim=[1, 2, 3])  # (B, num_features)
        
        # Flatten
        result = result.reshape(batch_size, -1)
        
        if squeeze:
            result = result.squeeze(0)
        
        return result


class SpatialSoftmax(nn.Module):
    """
    Spatial softmax layer for extracting keypoints from feature maps.
    
    Args:
        height: Height of the feature map
        width: Width of the feature map
        channel: Number of channels
        temperature: Temperature parameter for softmax (None, float, or -1 for learnable)
        log_heatmap: Whether to log the heatmap
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        temperature: float = None,
        log_heatmap: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.log_heatmap = log_heatmap
        
        # Create position grids
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        pos_x = pos_x.reshape(1, height * width)
        pos_y = pos_y.reshape(1, height * width)
        
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        
        # Temperature parameter
        if temperature == -1:
            # Learnable temperature
            self.temperature = nn.Parameter(torch.ones(1))
        elif temperature is None:
            self.temperature = 1.0
        else:
            self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape [B, H, W, C] or [H, W, C]
            
        Returns:
            Expected positions of shape [B, 2*C] or [2*C]
        """
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features.unsqueeze(0)
        
        assert len(features.shape) == 4
        batch_size = features.shape[0]
        num_featuremaps = features.shape[3]
        
        # Reshape: (B, H, W, C) -> (B, C, H*W)
        features = features.permute(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )
        
        # Apply softmax with temperature
        if isinstance(self.temperature, nn.Parameter):
            temperature = self.temperature
        else:
            temperature = self.temperature
        
        softmax_attention = torch.softmax(features / temperature, dim=2)
        
        # Compute expected positions
        expected_x = (self.pos_x * softmax_attention).sum(dim=2, keepdim=True)
        expected_x = expected_x.reshape(batch_size, num_featuremaps)
        
        expected_y = (self.pos_y * softmax_attention).sum(dim=2, keepdim=True)
        expected_y = expected_y.reshape(batch_size, num_featuremaps)
        
        expected_xy = torch.cat([expected_x, expected_y], dim=1)
        expected_xy = expected_xy.reshape(batch_size, 2 * num_featuremaps)
        
        if no_batch_dim:
            expected_xy = expected_xy[0]
        
        return expected_xy
