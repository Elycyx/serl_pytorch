"""Small convolutional encoders for vision tasks in PyTorch."""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings


class SmallEncoder(nn.Module):
    """
    Small convolutional encoder for vision tasks.
    
    Args:
        features: Number of features for each conv layer
        kernel_sizes: Kernel sizes for each conv layer
        strides: Strides for each conv layer
        padding: Padding for each conv layer (can be int sequence or 'same'/'valid')
        pool_method: Pooling method ('max', 'avg', 'spatial_learned_embeddings')
        bottleneck_dim: Optional bottleneck dimension
        spatial_block_size: Number of spatial blocks for learned embeddings
    """
    
    def __init__(
        self,
        features: Sequence[int] = (16, 16, 16),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        padding: Union[Sequence[int], str] = (1, 1, 1),
        pool_method: str = "spatial_learned_embeddings",
        bottleneck_dim: Optional[int] = None,
        spatial_block_size: Optional[int] = 8,
    ):
        super().__init__()
        assert len(features) == len(strides), \
            f"features and strides must have same length: {len(features)} vs {len(strides)}"
        
        self.features = features
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.pool_method = pool_method
        self.bottleneck_dim = bottleneck_dim
        self.spatial_block_size = spatial_block_size
        
        # Build conv layers
        self.conv_layers = nn.ModuleList()
        for i in range(len(features)):
            # Determine padding for this layer
            if isinstance(padding, str):
                pad = padding
            else:
                pad = padding[i]
            
            # Note: input channels will be determined lazily
            conv = nn.Conv2d(
                in_channels=-1,  # Will be set lazily
                out_channels=features[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=pad if isinstance(pad, str) else pad,
            )
            self.conv_layers.append(conv)
        
        # Pooling layer (created lazily if using spatial learned embeddings)
        self.spatial_pool = None
        
        # Bottleneck layers (created lazily)
        self.bottleneck_dense = None
        self.bottleneck_norm = None
        self.dropout = nn.Dropout(p=0.1)
    
    def _initialize_layers(self, input_channels: int):
        """Initialize conv layers with proper input channels."""
        prev_channels = input_channels
        
        for i, conv in enumerate(self.conv_layers):
            # Recreate conv layer with correct input channels
            if isinstance(self.padding, str):
                pad = self.padding
            else:
                pad = self.padding[i]
            
            new_conv = nn.Conv2d(
                in_channels=prev_channels,
                out_channels=self.features[i],
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                padding=pad if isinstance(pad, str) else pad,
            )
            self.conv_layers[i] = new_conv.to(next(self.parameters()).device)
            prev_channels = self.features[i]
        
        self._initialized = True
    
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: Input images of shape [B, H, W, C] or [B, C, H, W]
            train: Whether in training mode
            
        Returns:
            Encoded features
        """
        # Detect format and convert to channels-first if needed
        if observations.shape[-1] <= 4 and observations.shape[1] > 4:
            # Likely channels-last (B, H, W, C)
            x = observations.permute(0, 3, 1, 2)
        else:
            x = observations
        
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Lazy initialization
        if not hasattr(self, '_initialized'):
            self._initialize_layers(x.shape[1])
        
        # Apply conv layers with ReLU
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
        
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
        
        # Apply bottleneck if specified
        if self.bottleneck_dim is not None:
            if self.bottleneck_dense is None:
                self.bottleneck_dense = nn.Linear(x.shape[-1], self.bottleneck_dim).to(x.device)
                self.bottleneck_norm = nn.LayerNorm(self.bottleneck_dim).to(x.device)
            
            x = self.bottleneck_dense(x)
            x = self.bottleneck_norm(x)
            x = torch.tanh(x)
        
        return x


# Configuration dictionary
small_configs = {
    "small": SmallEncoder,
}


def create_small_encoder(config_name: str = "small", **kwargs):
    """
    Create a small encoder from config.
    
    Args:
        config_name: Name of the config ('small')
        **kwargs: Additional arguments to pass to encoder
        
    Returns:
        SmallEncoder instance
    """
    if config_name not in small_configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(small_configs.keys())}")
    
    encoder_class = small_configs[config_name]
    return encoder_class(**kwargs)
