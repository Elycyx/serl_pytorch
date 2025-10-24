from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from serl_launcher.common.common import default_init
from serl_launcher.networks.mlp import MLP


class ValueCritic(nn.Module):
    """Value function critic (V(s))."""
    
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        
        # Output layer will be created lazily
        self.output_layer = None
    
    def forward(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        if self.encoder is not None:
            observations = self.encoder(observations, train=train)
        
        outputs = self.network(observations, train=train)
        
        # Lazy initialization of output layer
        if self.output_layer is None:
            self.output_layer = nn.Linear(outputs.shape[-1], 1).to(outputs.device)
            if self.init_final is not None:
                nn.init.uniform_(self.output_layer.weight, -self.init_final, self.init_final)
                if self.output_layer.bias is not None:
                    nn.init.zeros_(self.output_layer.bias)
            else:
                default_init()(self.output_layer)
        
        value = self.output_layer(outputs)
        return value.squeeze(-1)


class Critic(nn.Module):
    """Q-function critic (Q(s, a))."""
    
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.init_final = init_final
        
        # Output layer will be created lazily
        self.output_layer = None
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Q-function.
        
        Args:
            observations: observations tensor [batch_size, obs_dim] or [batch_size, ...]
            actions: actions tensor [batch_size, action_dim] or [batch_size, num_actions, action_dim]
            train: whether in training mode
            
        Returns:
            Q-values: [batch_size] or [batch_size, num_actions]
        """
        # Handle multiple actions per state (for ensembling)
        if actions.ndim == 3:
            # actions shape: [batch_size, num_actions, action_dim]
            batch_size, num_actions, action_dim = actions.shape
            
            # Expand observations to match
            if self.encoder is None:
                obs_enc = observations.unsqueeze(1).expand(-1, num_actions, -1)
            else:
                obs_enc = self.encoder(observations, train=train)
                obs_enc = obs_enc.unsqueeze(1).expand(-1, num_actions, -1)
            
            # Reshape for batch processing
            obs_enc = obs_enc.reshape(batch_size * num_actions, -1)
            actions_flat = actions.reshape(batch_size * num_actions, -1)
            
            inputs = torch.cat([obs_enc, actions_flat], dim=-1)
            outputs = self.network(inputs, train=train)
            
            # Lazy initialization of output layer
            if self.output_layer is None:
                self.output_layer = nn.Linear(outputs.shape[-1], 1).to(outputs.device)
                if self.init_final is not None:
                    nn.init.uniform_(self.output_layer.weight, -self.init_final, self.init_final)
                    if self.output_layer.bias is not None:
                        nn.init.zeros_(self.output_layer.bias)
                else:
                    default_init()(self.output_layer)
            
            value = self.output_layer(outputs)
            value = value.reshape(batch_size, num_actions)
            return value.squeeze(-1) if num_actions == 1 else value
        else:
            # Standard case: actions shape [batch_size, action_dim]
            if self.encoder is None:
                obs_enc = observations
            else:
                obs_enc = self.encoder(observations, train=train)
            
            inputs = torch.cat([obs_enc, actions], dim=-1)
            outputs = self.network(inputs, train=train)
            
            # Lazy initialization of output layer
            if self.output_layer is None:
                self.output_layer = nn.Linear(outputs.shape[-1], 1).to(outputs.device)
                if self.init_final is not None:
                    nn.init.uniform_(self.output_layer.weight, -self.init_final, self.init_final)
                    if self.output_layer.bias is not None:
                        nn.init.zeros_(self.output_layer.bias)
                else:
                    default_init()(self.output_layer)
            
            value = self.output_layer(outputs)
            return value.squeeze(-1)


class DistributionalCritic(nn.Module):
    """Distributional Q-function critic."""
    
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        q_low: float,
        q_high: float,
        num_atoms: int = 51,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.q_low = q_low
        self.q_high = q_high
        self.num_atoms = num_atoms
        self.init_final = init_final
        
        # Output layer will be created lazily
        self.output_layer = None
        
        # Register atoms as buffer
        atoms = torch.linspace(q_low, q_high, num_atoms)
        self.register_buffer('atoms', atoms)
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool = False,
    ):
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train)
        
        inputs = torch.cat([obs_enc, actions], dim=-1)
        outputs = self.network(inputs, train=train)
        
        # Lazy initialization of output layer
        if self.output_layer is None:
            self.output_layer = nn.Linear(outputs.shape[-1], self.num_atoms).to(outputs.device)
            if self.init_final is not None:
                nn.init.uniform_(self.output_layer.weight, -self.init_final, self.init_final)
                if self.output_layer.bias is not None:
                    nn.init.zeros_(self.output_layer.bias)
            else:
                default_init()(self.output_layer)
        
        logits = self.output_layer(outputs)
        atoms = self.atoms.expand_as(logits)
        
        return logits, atoms


class ContrastiveCritic(nn.Module):
    """Contrastive critic for goal-conditioned RL."""
    
    def __init__(
        self,
        encoder: nn.Module,
        sa_net: nn.Module,
        g_net: nn.Module,
        repr_dim: int = 16,
        twin_q: bool = True,
        sa_net2: Optional[nn.Module] = None,
        g_net2: Optional[nn.Module] = None,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.sa_net = sa_net
        self.g_net = g_net
        self.repr_dim = repr_dim
        self.twin_q = twin_q
        self.sa_net2 = sa_net2
        self.g_net2 = g_net2
        self.init_final = init_final
        
        # Output layers will be created lazily
        self.sa_output = None
        self.g_output = None
        self.sa_output2 = None
        self.g_output2 = None
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        obs_goal_encoding = self.encoder(observations, train=train)
        encoding_dim = obs_goal_encoding.shape[-1] // 2
        obs_encoding = obs_goal_encoding[..., :encoding_dim]
        goal_encoding = obs_goal_encoding[..., encoding_dim:]
        
        if self.init_final is not None:
            init_fn = lambda m: nn.init.uniform_(m.weight, -self.init_final, self.init_final)
        else:
            init_fn = default_init()
        
        sa_inputs = torch.cat([obs_encoding, actions], dim=-1)
        sa_repr = self.sa_net(sa_inputs, train=train)
        
        # Lazy initialization
        if self.sa_output is None:
            self.sa_output = nn.Linear(sa_repr.shape[-1], self.repr_dim).to(sa_repr.device)
            init_fn(self.sa_output)
        
        sa_repr = self.sa_output(sa_repr)
        
        g_repr = self.g_net(goal_encoding, train=train)
        
        if self.g_output is None:
            self.g_output = nn.Linear(g_repr.shape[-1], self.repr_dim).to(g_repr.device)
            init_fn(self.g_output)
        
        g_repr = self.g_output(g_repr)
        
        # Compute outer product: (batch_i, k) x (batch_j, k) -> (batch_i, batch_j)
        outer = torch.einsum('ik,jk->ij', sa_repr, g_repr)
        
        if self.twin_q and self.sa_net2 is not None and self.g_net2 is not None:
            sa_repr2 = self.sa_net2(sa_inputs, train=train)
            
            if self.sa_output2 is None:
                self.sa_output2 = nn.Linear(sa_repr2.shape[-1], self.repr_dim).to(sa_repr2.device)
                init_fn(self.sa_output2)
            
            sa_repr2 = self.sa_output2(sa_repr2)
            
            g_repr2 = self.g_net2(goal_encoding, train=train)
            
            if self.g_output2 is None:
                self.g_output2 = nn.Linear(g_repr2.shape[-1], self.repr_dim).to(g_repr2.device)
                init_fn(self.g_output2)
            
            g_repr2 = self.g_output2(g_repr2)
            
            outer2 = torch.einsum('ik,jk->ij', sa_repr2, g_repr2)
            outer = torch.stack([outer, outer2], dim=-1)
        
        return outer


def create_ensemble(module_class, num_qs: int, *args, **kwargs):
    """
    Create an ensemble of modules.
    
    Args:
        module_class: The module class to instantiate
        num_qs: Number of ensemble members
        *args, **kwargs: Arguments to pass to module_class
    
    Returns:
        nn.ModuleList of ensemble members
    """
    return nn.ModuleList([module_class(*args, **kwargs) for _ in range(num_qs)])


class Policy(nn.Module):
    """Stochastic policy network."""
    
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        action_dim: int,
        init_final: Optional[float] = None,
        std_parameterization: str = "exp",  # "exp", "softplus", "fixed", or "uniform"
        std_min: Optional[float] = 1e-5,
        std_max: Optional[float] = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.init_final = init_final
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        
        if fixed_std is not None:
            self.register_buffer('fixed_std', fixed_std)
        else:
            self.fixed_std = None
        
        # Output layers will be created lazily
        self.mean_layer = None
        self.std_layer = None
        self.log_std_param = None
    
    def forward(
        self,
        observations: torch.Tensor,
        temperature: float = 1.0,
        train: bool = False,
    ):
        """
        Forward pass to get action distribution.
        
        Args:
            observations: Observation tensor
            temperature: Temperature for exploration
            train: Whether in training mode
            
        Returns:
            Distribution object
        """
        if self.encoder is None:
            obs_enc = observations
        else:
            # Note: stop_gradient in JAX means we don't backprop through encoder
            # In PyTorch, we can use .detach() for the same effect
            with torch.no_grad() if not train else torch.enable_grad():
                obs_enc = self.encoder(observations, train=train)
                if not train:
                    obs_enc = obs_enc.detach()
        
        outputs = self.network(obs_enc, train=train)
        
        # Lazy initialization of output layers
        if self.mean_layer is None:
            self.mean_layer = nn.Linear(outputs.shape[-1], self.action_dim).to(outputs.device)
            default_init()(self.mean_layer)
        
        means = self.mean_layer(outputs)
        
        # Compute standard deviations
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                if self.std_layer is None:
                    self.std_layer = nn.Linear(outputs.shape[-1], self.action_dim).to(outputs.device)
                    default_init()(self.std_layer)
                log_stds = self.std_layer(outputs)
                stds = torch.exp(log_stds)
            elif self.std_parameterization == "softplus":
                if self.std_layer is None:
                    self.std_layer = nn.Linear(outputs.shape[-1], self.action_dim).to(outputs.device)
                    default_init()(self.std_layer)
                stds = F.softplus(self.std_layer(outputs))
            elif self.std_parameterization == "uniform":
                if self.log_std_param is None:
                    self.log_std_param = nn.Parameter(torch.zeros(self.action_dim, device=outputs.device))
                stds = torch.exp(self.log_std_param).expand(means.shape[0], -1)
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = self.fixed_std.expand(means.shape[0], -1)
        
        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = torch.clamp(stds, self.std_min, self.std_max) * torch.sqrt(torch.tensor(temperature))
        
        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
        else:
            # Use independent normal distributions
            distribution = Normal(means, stds)
            distribution = torch.distributions.Independent(distribution, 1)
        
        return distribution


class TanhMultivariateNormalDiag(TransformedDistribution):
    """
    Multivariate normal distribution with diagonal covariance, transformed by tanh.
    Similar to distrax.Transformed in JAX.
    """
    
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        # Create base distribution
        base_dist = Normal(loc, scale_diag)
        base_dist = torch.distributions.Independent(base_dist, 1)
        
        # Create tanh transform
        transforms = [TanhTransform(cache_size=1)]
        
        # Add rescaling if bounds are provided
        if low is not None and high is not None:
            # After tanh, values are in (-1, 1), rescale to (low, high)
            # This is done via: x_rescaled = (x + 1) / 2 * (high - low) + low
            transforms.append(RescaleTransform(low, high))
        
        if len(transforms) == 1:
            transform = transforms[0]
        else:
            transform = torch.distributions.ComposeTransform(transforms)
        
        super().__init__(base_dist, transform)
    
    def mode(self) -> torch.Tensor:
        """Return the mode of the distribution (deterministic action)."""
        mode = self.base_dist.mean
        for transform in self.transforms:
            mode = transform(mode)
        return mode
    
    def stddev(self) -> torch.Tensor:
        """Return the standard deviation (approximate)."""
        return self.base_dist.stddev


class RescaleTransform(torch.distributions.Transform):
    """Transform to rescale values from (-1, 1) to (low, high)."""
    
    bijective = True
    
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.low = low
        self.high = high
    
    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)
        return (x + 1) / 2 * (self.high - self.low) + self.low
    
    def _inverse(self, y):
        # Rescale from (low, high) to (-1, 1)
        return 2 * (y - self.low) / (self.high - self.low) - 1
    
    def log_abs_det_jacobian(self, x, y):
        # Log determinant of Jacobian
        return torch.sum(torch.log(0.5 * (self.high - self.low)), dim=-1)
