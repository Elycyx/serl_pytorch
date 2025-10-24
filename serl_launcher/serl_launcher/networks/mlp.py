from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.common.common import default_init


class MLP(nn.Module):
    """Multi-Layer Perceptron network."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Union[Callable[[torch.Tensor], torch.Tensor], str] = "silu",
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        
        # Get activation function
        if isinstance(activations, str):
            self.activation = getattr(F, activations)
        else:
            self.activation = activations
        
        # Build layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate is not None and dropout_rate > 0 else None
        
        for i, size in enumerate(hidden_dims):
            layer = nn.Linear(
                hidden_dims[i-1] if i > 0 else -1,  # -1 will be inferred from input
                size
            )
            # Note: We'll handle initialization differently since we don't know input size yet
            self.layers.append(layer)
            
            if use_layer_norm and (i + 1 < len(hidden_dims) or activate_final):
                self.layer_norms.append(nn.LayerNorm(size))
            
            if dropout_rate is not None and dropout_rate > 0 and (i + 1 < len(hidden_dims) or activate_final):
                self.dropouts.append(nn.Dropout(p=dropout_rate))
    
    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Lazy initialization on first forward pass
        if not hasattr(self, '_initialized'):
            self._initialize_layers(x.shape[-1])
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropouts is not None and i < len(self.dropouts):
                    x = self.dropouts[i](x) if train else x
                if self.layer_norms is not None and i < len(self.layer_norms):
                    x = self.layer_norms[i](x)
                x = self.activation(x)
        
        return x
    
    def _initialize_layers(self, input_dim: int):
        """Initialize layers with proper input dimensions."""
        prev_dim = input_dim
        
        for i, layer in enumerate(self.layers):
            # Recreate layer with correct input dimension
            new_layer = nn.Linear(prev_dim, self.hidden_dims[i])
            default_init()(new_layer)
            self.layers[i] = new_layer
            prev_dim = self.hidden_dims[i]
        
        self._initialized = True


class MLPResNetBlock(nn.Module):
    """Residual block for MLPResNet."""
    
    def __init__(
        self,
        features: int,
        activation: Callable = F.silu,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        self.residual_proj = None  # Will be created if needed
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize weights
        default_init()(self.dense1)
        default_init()(self.dense2)
    
    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        residual = x
        
        if hasattr(self, 'dropout') and self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x) if train else x
        
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        
        if residual.shape != x.shape:
            if self.residual_proj is None:
                self.residual_proj = nn.Linear(residual.shape[-1], self.features)
                default_init()(self.residual_proj)
                self.residual_proj = self.residual_proj.to(x.device)
            residual = self.residual_proj(residual)
        
        return residual + x


class MLPResNet(nn.Module):
    """MLP with residual connections."""
    
    def __init__(
        self,
        num_blocks: int,
        out_dim: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = F.silu,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activations = activations
        
        self.input_layer = None  # Will be created lazily
        
        self.blocks = nn.ModuleList([
            MLPResNetBlock(
                hidden_dim,
                activation=activations,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        default_init()(self.output_layer)
    
    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        # Lazy initialization
        if self.input_layer is None:
            self.input_layer = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
            default_init()(self.input_layer)
        
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x, train=train)
        
        x = self.activations(x)
        x = self.output_layer(x)
        
        return x


class Scalar(nn.Module):
    """A learnable scalar parameter."""
    
    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self) -> torch.Tensor:
        return self.value
