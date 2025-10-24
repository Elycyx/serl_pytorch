"""MobileNet encoder wrapper for PyTorch."""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


class MobileNetEncoder(nn.Module):
    """
    Wrapper for pretrained MobileNet encoder.
    
    This serves as a wrapper for an ImageNet pretrained MobileNet encoder,
    providing pooling and optional bottleneck layers.
    
    Args:
        pretrained: Whether to use pretrained weights
        pool_method: Pooling method ('max', 'avg', 'spatial_learned_embeddings')
        bottleneck_dim: Optional bottleneck dimension
        spatial_block_size: Number of spatial blocks for learned embeddings
        freeze_backbone: Whether to freeze the backbone weights
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        pool_method: str = "spatial_learned_embeddings",
        bottleneck_dim: Optional[int] = None,
        spatial_block_size: Optional[int] = 8,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.pool_method = pool_method
        self.bottleneck_dim = bottleneck_dim
        self.spatial_block_size = spatial_block_size
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Use the features part (without the classifier)
        self.encoder = mobilenet.features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Pooling layer (created lazily if using spatial learned embeddings)
        self.spatial_pool = None
        
        # Bottleneck layers (created lazily)
        self.bottleneck_dense = None
        self.bottleneck_norm = None
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """
        Encode an image using the MobileNet encoder.
        
        Args:
            x: Input images of shape [B, H, W, C] or [B, C, H, W]
            train: Whether in training mode (affects dropout)
            
        Returns:
            Encoded features
        """
        # Detect format and convert to channels-first if needed
        reshape = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            reshape = True
        
        # Check if channels-last and convert
        if x.shape[-1] == 3 and x.shape[1] != 3:
            # Channels-last (B, H, W, C) -> (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # Normalize using ImageNet mean and std
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # Forward through encoder (frozen, so no grad)
        with torch.set_grad_enabled(not self.freeze_backbone):
            x = self.encoder(x)
        
        # Stop gradient if backbone is frozen
        if self.freeze_backbone:
            x = x.detach()
        
        # Apply pooling
        if self.pool_method == "max":
            # Global max pooling
            x = F.adaptive_max_pool2d(x, (1, 1))
            x = x.squeeze(-1).squeeze(-1)
        elif self.pool_method == "avg":
            # Global average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.squeeze(-1).squeeze(-1)
        elif self.pool_method == "spatial_learned_embeddings":
            if self.spatial_block_size is None:
                raise ValueError(
                    "spatial_block_size must be set when using spatial_learned_embeddings"
                )
            
            # Convert to channels-last for spatial embeddings
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            
            # Lazy initialization of spatial pool
            if self.spatial_pool is None:
                _, h, w, c = x.shape
                self.spatial_pool = SpatialLearnedEmbeddings(
                    height=h,
                    width=w,
                    channel=c,
                    num_features=self.spatial_block_size,
                ).to(x.device)
            
            x = self.spatial_pool(x)
            
            if train:
                x = self.dropout(x)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        # Remove batch dim if it was added
        if reshape:
            x = x.squeeze(0)
        
        # Apply bottleneck if specified
        if self.bottleneck_dim is not None:
            if self.bottleneck_dense is None:
                self.bottleneck_dense = nn.Linear(x.shape[-1], self.bottleneck_dim).to(x.device)
                self.bottleneck_norm = nn.LayerNorm(self.bottleneck_dim).to(x.device)
            
            x = self.bottleneck_dense(x)
            x = self.bottleneck_norm(x)
            x = torch.tanh(x)
        
        return x


def create_mobilenet_encoder(
    pretrained: bool = True,
    **kwargs
) -> MobileNetEncoder:
    """
    Create a MobileNet encoder.
    
    Args:
        pretrained: Whether to use pretrained ImageNet weights
        **kwargs: Additional arguments for MobileNetEncoder
        
    Returns:
        MobileNetEncoder instance
    """
    return MobileNetEncoder(pretrained=pretrained, **kwargs)
