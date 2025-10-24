import copy
from functools import partial
from typing import Optional, Tuple, FrozenSet, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from serl_launcher.common.common import TrainState, ModuleDict
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer, clip_gradients
from serl_launcher.common.typing import Batch, Data, Params
from serl_launcher.networks.actor_critic_nets import Critic, Policy, create_ensemble
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP


@dataclass
class SACAgent:
    """
    Soft Actor-Critic (SAC) agent implementation in PyTorch.
    
    Supports several variants:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    """
    
    state: TrainState
    config: Dict[str, Any]
    device: torch.device = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward_critic(
        self,
        observations: Data,
        actions: torch.Tensor,
        *,
        model: Optional[nn.Module] = None,
        train: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for critic network.
        
        Args:
            observations: Observation data
            actions: Action tensor
            model: Model to use (default: self.state.model)
            train: Whether in training mode
            
        Returns:
            Q-values from critic ensemble
        """
        if model is None:
            model = self.state.model
        
        # Get critic from model
        if isinstance(model, ModuleDict):
            critic = model.modules_dict['critic']
        else:
            critic = model.critic
        
        # Forward through each critic in ensemble
        if isinstance(critic, nn.ModuleList):
            q_values = torch.stack([c(observations, actions, train=train) for c in critic])
        else:
            q_values = critic(observations, actions, train=train)
            if q_values.ndim == 1:
                q_values = q_values.unsqueeze(0)
        
        return q_values
    
    def forward_target_critic(
        self,
        observations: Data,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for target critic network.
        
        Args:
            observations: Observation data
            actions: Action tensor
            
        Returns:
            Q-values from target critic ensemble
        """
        return self.forward_critic(
            observations, actions, model=self.state.target_model, train=False
        )
    
    def forward_policy(
        self,
        observations: Data,
        *,
        model: Optional[nn.Module] = None,
        train: bool = True,
        temperature: float = 1.0,
    ):
        """
        Forward pass for policy network.
        
        Args:
            observations: Observation data
            model: Model to use (default: self.state.model)
            train: Whether in training mode
            temperature: Temperature for exploration
            
        Returns:
            Action distribution
        """
        if model is None:
            model = self.state.model
        
        # Get policy from model
        if isinstance(model, ModuleDict):
            policy = model.modules_dict['actor']
        else:
            policy = model.actor
        
        return policy(observations, temperature=temperature, train=train)
    
    def forward_temperature(self, *, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Forward pass for temperature Lagrange multiplier.
        
        Args:
            model: Model to use (default: self.state.model)
            
        Returns:
            Temperature value
        """
        if model is None:
            model = self.state.model
        
        # Get temperature from model
        if isinstance(model, ModuleDict):
            temperature_module = model.modules_dict['temperature']
        else:
            temperature_module = model.temperature
        
        return temperature_module()
    
    def temperature_lagrange_penalty(
        self,
        entropy: torch.Tensor,
        *,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute Lagrange penalty for temperature constraint.
        
        Args:
            entropy: Current entropy value
            model: Model to use (default: self.state.model)
            
        Returns:
            Temperature Lagrange penalty
        """
        if model is None:
            model = self.state.model
        
        # Get temperature module
        if isinstance(model, ModuleDict):
            temperature_module = model.modules_dict['temperature']
        else:
            temperature_module = model.temperature
        
        target_entropy = self.config['target_entropy']
        return temperature_module(lhs=entropy, rhs=torch.tensor(target_entropy, device=entropy.device))
    
    def _compute_next_actions(self, batch: Batch):
        """
        Compute next actions and log probabilities.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (next_actions, next_log_probs)
        """
        next_action_distribution = self.forward_policy(
            batch['next_observations'],
            train=False,
        )
        next_actions = next_action_distribution.rsample()
        next_log_probs = next_action_distribution.log_prob(next_actions)
        
        return next_actions, next_log_probs
    
    def critic_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute critic loss.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (loss, info_dict)
        """
        batch_size = batch['rewards'].shape[0]
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self._compute_next_actions(batch)
            
            target_qs = self.forward_target_critic(
                batch['next_observations'],
                next_actions,
            )
            
            # Subsample critics if specified
            if self.config['critic_subsample_size'] is not None:
                subsample_size = self.config['critic_subsample_size']
                indices = torch.randperm(target_qs.shape[0], device=self.device)[:subsample_size]
                target_qs = target_qs[indices]
            
            # Take minimum over ensemble
            target_q = target_qs.min(dim=0).values
            
            # Add entropy bonus if requested
            if self.config.get('backup_entropy', False):
                temperature = self.forward_temperature()
                target_q = target_q + temperature * next_log_probs
            
            # Compute TD target
            discount = self.config['discount']
            target_q = batch['rewards'] + discount * batch['masks'] * target_q
        
        # Compute predicted Q-values
        predicted_qs = self.forward_critic(
            batch['observations'],
            batch['actions'],
            train=True,
        )
        
        # Compute TD error for each critic
        td_errors = predicted_qs - target_q.unsqueeze(0)
        critic_loss = (td_errors ** 2).mean()
        
        info = {
            'critic_loss': critic_loss.item(),
            'q_values': predicted_qs.mean().item(),
            'q_std': predicted_qs.std().item(),
        }
        
        return critic_loss, info
    
    def policy_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute policy loss.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (loss, info_dict)
        """
        batch_size = batch['rewards'].shape[0]
        temperature = self.forward_temperature().detach()
        
        # Sample actions from policy
        action_distribution = self.forward_policy(
            batch['observations'],
            train=True,
        )
        actions = action_distribution.rsample()
        log_probs = action_distribution.log_prob(actions)
        
        # Compute Q-values for sampled actions
        with torch.no_grad():
            predicted_qs = self.forward_critic(
                batch['observations'],
                actions,
                train=False,
            )
            predicted_q = predicted_qs.mean(dim=0)
        
        # Compute policy loss (maximize Q - temperature * log_prob)
        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -actor_objective.mean()
        
        info = {
            'actor_loss': actor_loss.item(),
            'temperature': temperature.item(),
            'entropy': -log_probs.mean().item(),
        }
        
        return actor_loss, info
    
    def temperature_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute temperature loss.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Tuple of (loss, info_dict)
        """
        with torch.no_grad():
            _, next_log_probs = self._compute_next_actions(batch)
            entropy = -next_log_probs.mean()
        
        temperature_loss = self.temperature_lagrange_penalty(entropy)
        
        return temperature_loss.mean(), {'temperature_loss': temperature_loss.mean().item()}
    
    def update(
        self,
        batch: Batch,
        networks_to_update: FrozenSet[str] = frozenset({'actor', 'critic', 'temperature'}),
    ) -> Tuple["SACAgent", Dict[str, float]]:
        """
        Take one gradient step on all (or a subset) of the networks.
        
        Args:
            batch: Batch of transitions
            networks_to_update: Names of networks to update
            
        Returns:
            Tuple of (updated agent, info dict)
        """
        info = {}
        
        # Update critic
        if 'critic' in networks_to_update:
            self.state.zero_grad()
            critic_loss, critic_info = self.critic_loss_fn(batch)
            critic_loss.backward()
            
            if 'critic' in self.state.optimizers:
                clip_gradients(self.state.optimizers['critic'])
                self.state.optimizers['critic'].step()
            
            info.update(critic_info)
            
            # Update target network
            self.state.target_update(self.config['soft_target_update_rate'])
        
        # Update actor
        if 'actor' in networks_to_update:
            self.state.zero_grad()
            actor_loss, actor_info = self.policy_loss_fn(batch)
            actor_loss.backward()
            
            if 'actor' in self.state.optimizers:
                clip_gradients(self.state.optimizers['actor'])
                self.state.optimizers['actor'].step()
            
            info.update(actor_info)
        
        # Update temperature
        if 'temperature' in networks_to_update:
            self.state.zero_grad()
            temperature_loss, temp_info = self.temperature_loss_fn(batch)
            temperature_loss.backward()
            
            if 'temperature' in self.state.optimizers:
                clip_gradients(self.state.optimizers['temperature'])
                self.state.optimizers['temperature'].step()
            
            info.update(temp_info)
        
        # Update learning rate schedulers if present
        for name, optimizer in self.state.optimizers.items():
            if hasattr(optimizer, 'scheduler') and optimizer.scheduler is not None:
                optimizer.scheduler.step()
                info[f'{name}_lr'] = optimizer.param_groups[0]['lr']
        
        self.state.step += 1
        
        return self, info
    
    def sample_actions(
        self,
        observations: Data,
        *,
        argmax: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Sample actions from the policy network.
        
        Args:
            observations: Observation data
            argmax: Whether to return the mode (deterministic)
            
        Returns:
            Sampled actions
        """
        with torch.no_grad():
            dist = self.forward_policy(observations, train=False)
            if argmax:
                actions = dist.mode()
            else:
                actions = dist.sample()
        
        return actions.cpu().numpy()
    
    @classmethod
    def create(
        cls,
        observations: Data,
        actions: np.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs: Dict = None,
        critic_optimizer_kwargs: Dict = None,
        temperature_optimizer_kwargs: Dict = None,
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        device: torch.device = None,
    ) -> "SACAgent":
        """
        Create a new SAC agent.
        
        Args:
            observations: Example observations
            actions: Example actions
            actor_def: Actor network definition
            critic_def: Critic network definition
            temperature_def: Temperature Lagrange multiplier
            actor_optimizer_kwargs: Actor optimizer configuration
            critic_optimizer_kwargs: Critic optimizer configuration
            temperature_optimizer_kwargs: Temperature optimizer configuration
            discount: Discount factor
            soft_target_update_rate: Target network update rate
            target_entropy: Target entropy (default: -action_dim / 2)
            backup_entropy: Whether to include entropy in TD backup
            critic_ensemble_size: Number of critics in ensemble
            critic_subsample_size: Number of critics to subsample for target
            device: Device to place models on
            
        Returns:
            New SAC agent instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default optimizer kwargs
        if actor_optimizer_kwargs is None:
            actor_optimizer_kwargs = {'learning_rate': 3e-4, 'warmup_steps': 2000}
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {'learning_rate': 3e-4, 'warmup_steps': 2000}
        if temperature_optimizer_kwargs is None:
            temperature_optimizer_kwargs = {'learning_rate': 3e-4}
        
        # Create model
        networks = {
            'actor': actor_def,
            'critic': critic_def,
            'temperature': temperature_def,
        }
        model = ModuleDict(networks).to(device)
        
        # Create target model
        target_model = copy.deepcopy(model)
        for param in target_model.parameters():
            param.requires_grad = False
        
        # Create optimizers
        optimizers = {
            'actor': make_optimizer(model.modules_dict['actor'].parameters(), **actor_optimizer_kwargs),
            'critic': make_optimizer(model.modules_dict['critic'].parameters(), **critic_optimizer_kwargs),
            'temperature': make_optimizer(model.modules_dict['temperature'].parameters(), **temperature_optimizer_kwargs),
        }
        
        # Create train state
        state = TrainState.create(
            model=model,
            target_model=target_model,
            optimizers=optimizers,
        )
        
        # Config
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2
        
        config = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
        )
        
        return cls(state=state, config=config, device=device)
    
    @classmethod
    def create_pixels(
        cls,
        observations: Data,
        actions: np.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: Dict = None,
        policy_network_kwargs: Dict = None,
        policy_kwargs: Dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        **kwargs,
    ) -> "SACAgent":
        """
        Create a new pixel-based SAC agent.
        
        Args:
            observations: Example observations
            actions: Example actions
            encoder_def: Image encoder network
            shared_encoder: Whether to share encoder between actor and critic
            use_proprio: Whether to use proprioception
            critic_network_kwargs: Critic MLP configuration
            policy_network_kwargs: Policy MLP configuration
            policy_kwargs: Policy-specific configuration
            critic_ensemble_size: Number of critics in ensemble
            critic_subsample_size: Number of critics to subsample
            temperature_init: Initial temperature value
            **kwargs: Additional arguments for create()
            
        Returns:
            New SAC agent instance
        """
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {'hidden_dims': [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {'hidden_dims': [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                'tanh_squash_distribution': True,
                'std_parameterization': 'uniform',
            }
        
        policy_network_kwargs['activate_final'] = True
        critic_network_kwargs['activate_final'] = True
        
        # Create encoding wrapper
        encoder_def = EncodingWrapper(
            encoder=encoder_def,
            use_proprio=use_proprio,
            enable_stacking=True,
        )
        
        if shared_encoder:
            encoders = {
                'actor': encoder_def,
                'critic': encoder_def,
            }
        else:
            encoders = {
                'actor': encoder_def,
                'critic': copy.deepcopy(encoder_def),
            }
        
        # Define networks
        policy_def = Policy(
            encoder=encoders['actor'],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        )
        
        # Create critic ensemble
        critic_def = create_ensemble(
            lambda: Critic(encoder=encoders['critic'], network=MLP(**critic_network_kwargs)),
            critic_ensemble_size,
        )
        
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )
        
        return cls.create(
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            **kwargs,
        )
    
    @classmethod
    def create_states(
        cls,
        observations: Data,
        actions: np.ndarray,
        # Model architecture
        critic_network_kwargs: Dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        policy_network_kwargs: Dict = None,
        policy_kwargs: Dict = None,
        temperature_init: float = 1.0,
        **kwargs,
    ) -> "SACAgent":
        """
        Create a new state-based SAC agent (no vision encoder).
        
        Args:
            observations: Example observations
            actions: Example actions
            critic_network_kwargs: Critic MLP configuration
            critic_ensemble_size: Number of critics in ensemble
            critic_subsample_size: Number of critics to subsample
            policy_network_kwargs: Policy MLP configuration
            policy_kwargs: Policy-specific configuration
            temperature_init: Initial temperature value
            **kwargs: Additional arguments for create()
            
        Returns:
            New SAC agent instance
        """
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {'hidden_dims': [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {'hidden_dims': [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                'tanh_squash_distribution': True,
                'std_parameterization': 'uniform',
            }
        
        policy_network_kwargs['activate_final'] = True
        critic_network_kwargs['activate_final'] = True
        
        # Define networks (no encoders)
        policy_def = Policy(
            encoder=None,
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        )
        
        # Create critic ensemble
        critic_def = create_ensemble(
            lambda: Critic(encoder=None, network=MLP(**critic_network_kwargs)),
            critic_ensemble_size,
        )
        
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )
        
        return cls.create(
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            **kwargs,
        )
    
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
    ) -> Tuple["SACAgent", Dict[str, float]]:
        """
        High-UTD (Update-To-Data ratio) version of update.
        
        Splits the batch into minibatches, performs `utd_ratio` critic
        updates, and then one actor/temperature update.
        
        Args:
            batch: Batch of transitions
            utd_ratio: Number of critic updates per actor update
            
        Returns:
            Tuple of (updated agent, info dict)
        """
        batch_size = batch['rewards'].shape[0]
        assert batch_size % utd_ratio == 0, \
            f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        
        minibatch_size = batch_size // utd_ratio
        
        # Split batch into minibatches
        def split_batch(data):
            if isinstance(data, dict):
                return {k: split_batch(v) for k, v in data.items()}
            else:
                return data.reshape(utd_ratio, minibatch_size, *data.shape[1:])
        
        minibatches = split_batch(batch)
        
        # Update critic multiple times
        critic_infos = []
        for i in range(utd_ratio):
            minibatch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in minibatches.items()}
            _, info = self.update(minibatch, networks_to_update=frozenset({'critic'}))
            critic_infos.append(info)
        
        # Average critic info
        critic_info = {k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0].keys()}
        
        # Update actor and temperature once
        _, actor_temp_info = self.update(batch, networks_to_update=frozenset({'actor', 'temperature'}))
        
        # Combine info
        info = {**critic_info, **actor_temp_info}
        
        return self, info
