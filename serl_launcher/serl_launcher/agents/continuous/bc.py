"""
Behavioral Cloning (BC) agent implementation in PyTorch.

BC is a simple imitation learning algorithm that trains a policy
to match expert demonstrations using supervised learning.
"""

from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.common.common import TorchRLTrainState, ModuleDict
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, PRNGKey, Data
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.torch_utils import next_rng


class BCAgent:
    """
    Behavioral Cloning agent.
    
    Trains a policy to match expert demonstrations using supervised learning
    (negative log-likelihood loss).
    """
    
    def __init__(self, state: TorchRLTrainState, config: dict):
        """
        Initialize BC agent.
        
        Args:
            state: Training state
            config: Configuration dict
        """
        self.state = state
        self.config = config
    
    def data_augmentation_fn(
        self,
        observations: Data,
        generator: Optional[torch.Generator] = None,
    ) -> Data:
        """
        Apply data augmentation (random crop) to observations.
        
        Args:
            observations: Observations dict
            generator: Random generator
            
        Returns:
            Augmented observations
        """
        augmented_obs = {}
        for key, value in observations.items():
            if key in self.config["image_keys"]:
                # Apply random crop augmentation
                augmented_obs[key] = batched_random_crop(
                    value,
                    padding=4,
                    generator=generator,
                )
            else:
                augmented_obs[key] = value
        
        return augmented_obs
    
    def update(
        self,
        batch: Batch,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["BCAgent", Dict[str, Any]]:
        """
        Update the policy using behavioral cloning.
        
        Args:
            batch: Training batch with observations and expert actions
            pmap_axis: Parallel axis (not used in single-GPU PyTorch)
            
        Returns:
            Updated agent and info dict
        """
        # Unpack batch if necessary
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        
        # Optional: Apply data augmentation (currently commented out in original)
        # obs_generator = next_rng()
        # obs = self.data_augmentation_fn(batch["observations"], obs_generator)
        # batch = {**batch, "observations": obs}
        
        # Zero gradients
        self.state.optimizers["actor"].zero_grad()
        
        # Forward pass through policy
        dist = self.state.models["actor"](
            batch["observations"],
            temperature=1.0,
            train=True,
        )
        
        # Policy actions (mode)
        pi_actions = dist.mean  # For Gaussian, mean is the mode
        
        # Log probabilities of expert actions
        log_probs = dist.log_prob(batch["actions"])
        
        # MSE loss (for monitoring)
        mse = ((pi_actions - batch["actions"]) ** 2).sum(dim=-1)
        
        # Behavioral cloning loss: negative log-likelihood
        actor_loss = -log_probs.mean()
        
        # Backward pass
        actor_loss.backward()
        
        # Update parameters
        self.state.optimizers["actor"].step()
        
        # Increment step
        self.state.step += 1
        
        # Collect info
        info = {
            "actor_loss": actor_loss.item(),
            "mse": mse.mean().item(),
        }
        
        return self, info
    
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> torch.Tensor:
        """
        Sample actions from the policy.
        
        Args:
            observations: Observations
            seed: Random seed (generator)
            temperature: Temperature for sampling
            argmax: If True, return mode (deterministic)
            
        Returns:
            Sampled actions
        """
        with torch.no_grad():
            dist = self.state.models["actor"](
                observations,
                temperature=temperature,
                train=False,
            )
            
            if argmax:
                actions = dist.mean  # Deterministic mode
            else:
                if seed is not None and isinstance(seed, torch.Generator):
                    # Sample with specific generator (for reproducibility)
                    actions = dist.sample()
                else:
                    actions = dist.sample()
        
        return actions
    
    def get_debug_metrics(self, batch: Batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get debug metrics for monitoring.
        
        Args:
            batch: Training batch
            **kwargs: Additional arguments
            
        Returns:
            Dict of debug metrics
        """
        with torch.no_grad():
            dist = self.state.models["actor"](
                batch["observations"],
                temperature=1.0,
                train=False,
            )
            pi_actions = dist.mean
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(dim=-1)
        
        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: torch.Tensor,
        # Model architecture
        encoder_type: str = "small",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: Dict = None,
        policy_kwargs: Dict = None,
        # Optimizer
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ):
        """
        Create a new BC agent.
        
        Args:
            rng: Random generator
            observations: Sample observations
            actions: Sample actions
            encoder_type: Type of encoder ('small', 'resnet', 'resnet-pretrained')
            image_keys: Keys for image observations
            use_proprio: Whether to use proprioception
            network_kwargs: MLP config for policy
            policy_kwargs: Policy config
            learning_rate: Learning rate
            device: Device to use
            
        Returns:
            BCAgent instance
        """
        # Default kwargs
        if network_kwargs is None:
            network_kwargs = {"hidden_dims": [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {"tanh_squash_distribution": False}
        
        # Create encoder
        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder
            
            encoders = nn.ModuleDict({
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="valid",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                )
                for image_key in image_keys
            })
        elif encoder_type == "resnet":
            # ResNet encoder placeholder
            raise NotImplementedError("ResNet encoder not yet implemented for PyTorch")
        elif encoder_type == "resnet-pretrained":
            # Pretrained ResNet encoder placeholder
            raise NotImplementedError("Pretrained ResNet encoder not yet implemented for PyTorch")
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")
        
        # Create encoding wrapper
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        ).to(device)
        
        # Move observations to device
        observations = torch.tree_map(
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            observations
        )
        actions = actions.to(device)
        
        # Create policy network
        network_kwargs["activate_final"] = True
        actor_def = Policy(
            encoder=encoder_def,
            network=MLP(**network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        ).to(device)
        
        networks = {"actor": actor_def}
        model_dict = ModuleDict(networks)
        
        # Create optimizer
        optimizer = torch.optim.Adam(actor_def.parameters(), lr=learning_rate)
        optimizers = {"actor": optimizer}
        
        # Create train state
        state = TorchRLTrainState.create(
            apply_fn=model_dict.forward,
            models=networks,
            optimizers=optimizers,
            rng=rng,
            target_models=None,  # BC doesn't use target networks
        )
        
        config = dict(
            image_keys=image_keys,
        )
        
        agent = cls(state, config)
        
        if encoder_type == "resnet-pretrained":
            # Load pretrained weights for ResNet-10
            # from serl_launcher.utils.train_utils import load_resnet10_params
            # agent = load_resnet10_params(agent, image_keys)
            raise NotImplementedError("Pretrained ResNet loading not yet implemented")
        
        return agent
