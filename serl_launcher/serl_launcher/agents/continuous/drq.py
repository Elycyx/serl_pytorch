"""
DrQ (Data-Regularized Q) agent implementation in PyTorch.

DrQ is an extension of SAC that adds data augmentation (random crop)
for image-based RL.
"""

import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import TorchRLTrainState, ModuleDict
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.torch_utils import next_rng


class DrQAgent(SACAgent):
    """
    DrQ agent: SAC with data augmentation.
    
    Adds random crop data augmentation on top of SAC for image-based RL.
    """
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: torch.Tensor,
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
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
    ):
        """
        Create DrQAgent instance.
        
        Args:
            rng: Random generator
            observations: Sample observations
            actions: Sample actions
            actor_def: Actor network
            critic_def: Critic network
            temperature_def: Temperature network
            actor_optimizer_kwargs: Actor optimizer config
            critic_optimizer_kwargs: Critic optimizer config
            temperature_optimizer_kwargs: Temperature optimizer config
            discount: Discount factor
            soft_target_update_rate: Target network update rate
            target_entropy: Target entropy for temperature
            entropy_per_dim: Whether to use per-dimension entropy
            backup_entropy: Whether to back up entropy in target
            critic_ensemble_size: Number of critics in ensemble
            critic_subsample_size: Number of critics to subsample
            image_keys: Keys for image observations
            device: Device to use
            
        Returns:
            DrQAgent instance
        """
        # Default optimizer kwargs
        if actor_optimizer_kwargs is None:
            actor_optimizer_kwargs = {"learning_rate": 3e-4}
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {"learning_rate": 3e-4}
        if temperature_optimizer_kwargs is None:
            temperature_optimizer_kwargs = {"learning_rate": 3e-4}
        
        networks = {
            "actor": actor_def.to(device),
            "critic": critic_def.to(device),
            "temperature": temperature_def.to(device),
        }
        
        model_dict = ModuleDict(networks)
        
        # Move observations to device
        observations = torch.tree_map(
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            observations
        )
        actions = actions.to(device)
        
        # Define optimizers
        optimizers = {
            "actor": make_optimizer(networks["actor"].parameters(), **actor_optimizer_kwargs),
            "critic": make_optimizer(networks["critic"].parameters(), **critic_optimizer_kwargs),
            "temperature": make_optimizer(networks["temperature"].parameters(), **temperature_optimizer_kwargs),
        }
        
        # Create target networks
        target_networks = {
            "critic": copy.deepcopy(networks["critic"]),
            "temperature": copy.deepcopy(networks["temperature"]),
        }
        
        # Create train state
        state = TorchRLTrainState.create(
            apply_fn=model_dict.forward,
            models=networks,
            optimizers=optimizers,
            rng=rng,
            target_models=target_networks,
        )
        
        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2
        
        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
            ),
        )
    
    @classmethod
    def create_drq(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: torch.Tensor,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: Dict = None,
        policy_network_kwargs: Dict = None,
        policy_kwargs: Dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
        **kwargs,
    ):
        """
        Create a new pixel-based DrQ agent with specified encoder.
        
        Args:
            rng: Random generator
            observations: Sample observations
            actions: Sample actions
            encoder_type: Type of encoder ('small', 'resnet', 'resnet-pretrained')
            shared_encoder: Whether to share encoder between actor and critic
            use_proprio: Whether to use proprioception
            critic_network_kwargs: Critic MLP config
            policy_network_kwargs: Policy MLP config
            policy_kwargs: Policy config
            critic_ensemble_size: Number of critics
            critic_subsample_size: Number of critics to subsample
            temperature_init: Initial temperature value
            image_keys: Keys for image observations
            device: Device to use
            **kwargs: Additional arguments
            
        Returns:
            DrQAgent instance
        """
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": True,
                "std_parameterization": "uniform",
            }
        
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        
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
            # Would need to implement resnetv1_configs for PyTorch
            raise NotImplementedError("ResNet encoder not yet implemented for PyTorch")
        elif encoder_type == "resnet-pretrained":
            # Pretrained ResNet encoder placeholder
            raise NotImplementedError("Pretrained ResNet encoder not yet implemented for PyTorch")
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")
        
        # Calculate encoder output dim
        # This needs to be done by running a forward pass or manually calculated
        sample_image = observations[list(image_keys)[0]]
        if isinstance(sample_image, np.ndarray):
            sample_image = torch.from_numpy(sample_image).to(device)
        
        # Assuming encoder output dim is known or can be inferred
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        ).to(device)
        
        # For encoder output dimension, we need to do a forward pass
        with torch.no_grad():
            encoder_output = encoder_def(observations)
            encoder_out_dim = encoder_output.shape[-1]
        
        encoders_dict = {
            "critic": encoder_def,
            "actor": encoder_def if shared_encoder else copy.deepcopy(encoder_def),
        }
        
        # Define networks
        critic_backbone_fn = lambda: MLP(**critic_network_kwargs)
        critic_backbones = nn.ModuleList([
            critic_backbone_fn() for _ in range(critic_ensemble_size)
        ])
        
        critic_def = Critic(
            encoder=encoders_dict["critic"],
            network=critic_backbones,
        ).to(device)
        
        policy_def = Policy(
            encoder=encoders_dict["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
        ).to(device)
        
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        ).to(device)
        
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            device=device,
            **kwargs,
        )
        
        if encoder_type == "resnet-pretrained":
            # Load pretrained weights for ResNet-10
            # from serl_launcher.utils.train_utils import load_resnet10_params
            # agent = load_resnet10_params(agent, image_keys)
            raise NotImplementedError("Pretrained ResNet loading not yet implemented")
        
        return agent
    
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
    
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Fast high-UTD version of update.
        
        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.
        
        Batch dimension must be divisible by `utd_ratio`.
        
        It also performs data augmentation on the observations and next_observations
        before updating the network.
        
        Args:
            batch: Training batch
            utd_ratio: Update-to-data ratio
            pmap_axis: Parallel axis (not used in single-GPU PyTorch)
            
        Returns:
            Updated agent and info dict
        """
        new_agent = self
        
        # Unpack batch if necessary
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        
        # Apply data augmentation
        obs_generator = next_rng()
        next_obs_generator = next_rng()
        
        obs = self.data_augmentation_fn(batch["observations"], obs_generator)
        next_obs = self.data_augmentation_fn(batch["next_observations"], next_obs_generator)
        
        # Update batch with augmented observations
        augmented_batch = {**batch}
        augmented_batch["observations"] = obs
        augmented_batch["next_observations"] = next_obs
        
        # Call parent class's update_high_utd
        return SACAgent.update_high_utd(
            new_agent, augmented_batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )
    
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Update only the critics (not actor or temperature).
        
        Useful for high UTD training where critics are updated more frequently
        than the actor.
        
        Args:
            batch: Training batch
            pmap_axis: Parallel axis (not used in single-GPU PyTorch)
            
        Returns:
            Updated agent and critic info dict
        """
        new_agent = self
        
        # Unpack batch if necessary
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        
        # Apply data augmentation
        obs_generator = next_rng()
        next_obs_generator = next_rng()
        
        obs = self.data_augmentation_fn(batch["observations"], obs_generator)
        next_obs = self.data_augmentation_fn(batch["next_observations"], next_obs_generator)
        
        # Update batch with augmented observations
        augmented_batch = {**batch}
        augmented_batch["observations"] = obs
        augmented_batch["next_observations"] = next_obs
        
        # Update only critics
        new_agent, critic_infos = new_agent.update(
            augmented_batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        
        # Remove unused keys
        critic_infos.pop("actor", None)
        critic_infos.pop("temperature", None)
        
        return new_agent, critic_infos
