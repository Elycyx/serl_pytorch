"""
VICE (Variational Inverse Control with Events) agent implementation in PyTorch.

VICE extends DrQ with a learned reward classifier that discriminates between
successful and unsuccessful transitions.
"""

import copy
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.common import TorchRLTrainState, ModuleDict
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.classifier import BinaryClassifier
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.utils.torch_utils import next_rng


class VICEAgent(DrQAgent):
    """
    VICE agent: DrQ with learned reward function.
    
    Uses a binary classifier to distinguish between successful and unsuccessful
    transitions, which provides the reward signal for RL.
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
        vice_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs: Dict = None,
        critic_optimizer_kwargs: Dict = None,
        temperature_optimizer_kwargs: Dict = None,
        vice_optimizer_kwargs: Dict = None,
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
        Create VICE Agent instance.
        
        Args:
            rng: Random generator
            observations: Sample observations
            actions: Sample actions
            actor_def: Actor network
            critic_def: Critic network
            temperature_def: Temperature network
            vice_def: VICE reward classifier
            actor_optimizer_kwargs: Actor optimizer config
            critic_optimizer_kwargs: Critic optimizer config
            temperature_optimizer_kwargs: Temperature optimizer config
            vice_optimizer_kwargs: VICE optimizer config
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
            VICEAgent instance
        """
        # Default optimizer kwargs
        if actor_optimizer_kwargs is None:
            actor_optimizer_kwargs = {"learning_rate": 3e-4}
        if critic_optimizer_kwargs is None:
            critic_optimizer_kwargs = {"learning_rate": 3e-4}
        if temperature_optimizer_kwargs is None:
            temperature_optimizer_kwargs = {"learning_rate": 3e-4}
        if vice_optimizer_kwargs is None:
            vice_optimizer_kwargs = {"learning_rate": 3e-4}
        
        networks = {
            "actor": actor_def.to(device),
            "critic": critic_def.to(device),
            "temperature": temperature_def.to(device),
            "vice": vice_def.to(device),
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
            "vice": make_optimizer(networks["vice"].parameters(), **vice_optimizer_kwargs),
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
    def create_vice(
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
        vice_network_kwargs: Dict = None,
        policy_kwargs: Dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        device: str = "cpu",
        **kwargs,
    ):
        """
        Create a new pixel-based VICE agent with specified encoder.
        
        Args:
            rng: Random generator
            observations: Sample observations
            actions: Sample actions
            encoder_type: Type of encoder ('small', 'resnet', 'resnet-pretrained')
            shared_encoder: Whether to share encoder between actor and critic
            use_proprio: Whether to use proprioception
            critic_network_kwargs: Critic MLP config
            policy_network_kwargs: Policy MLP config
            vice_network_kwargs: VICE classifier MLP config
            policy_kwargs: Policy config
            critic_ensemble_size: Number of critics
            critic_subsample_size: Number of critics to subsample
            temperature_init: Initial temperature value
            image_keys: Keys for image observations
            device: Device to use
            **kwargs: Additional arguments
            
        Returns:
            VICEAgent instance
        """
        # Default kwargs
        if critic_network_kwargs is None:
            critic_network_kwargs = {"hidden_dims": [256, 256]}
        if policy_network_kwargs is None:
            policy_network_kwargs = {"hidden_dims": [256, 256]}
        if vice_network_kwargs is None:
            vice_network_kwargs = {
                "hidden_dims": [256],
                "activations": F.leaky_relu,
                "use_layer_norm": True,
                "dropout_rate": 0.1,
            }
        if policy_kwargs is None:
            policy_kwargs = {
                "tanh_squash_distribution": True,
                "std_parameterization": "uniform",
            }
        
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        vice_network_kwargs["activate_final"] = True
        
        # Create encoders
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
            
            vice_encoders = nn.ModuleDict({
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
        
        # Create encoding wrappers
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        ).to(device)
        
        vice_encoder_def = EncodingWrapper(
            encoder=vice_encoders,
            use_proprio=False,
            enable_stacking=True,
            image_keys=image_keys,
        ).to(device)
        
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
        
        # VICE classifier
        # Pretrained encoder is None for now (would need to load for resnet-pretrained)
        vice_def = BinaryClassifier(
            pretrained_encoder=None,  # Placeholder
            encoder=vice_encoder_def,
            network=MLP(**vice_network_kwargs),
            enable_stacking=True,
        ).to(device)
        
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            vice_def=vice_def,
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
    
    def encode_images(
        self,
        images: Data,
        train: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Forward pass for encoder network to get embeddings.
        
        Args:
            images: Input images
            train: Whether in training mode
            generator: Random generator for dropout
            
        Returns:
            Encoded embeddings
        """
        if train:
            self.state.models["vice"].train()
        else:
            self.state.models["vice"].eval()
        
        with torch.set_grad_enabled(train):
            encoded = self.state.models["vice"](
                images,
                train=train,
                return_encoded=True,
            )
        
        return encoded
    
    def loss_fns(self, batch: Batch) -> Dict:
        """
        Override loss functions to include VICE (set to zero for normal updates).
        
        Args:
            batch: Training batch
            
        Returns:
            Dict of loss functions
        """
        loss_fns_dict = super().loss_fns(batch)
        # Add VICE loss (set to zero for normal SAC updates)
        loss_fns_dict["vice"] = lambda: (torch.tensor(0.0), {})
        return loss_fns_dict
    
    def update_vice(
        self,
        batch: Batch,
        pmap_axis: Optional[str] = None,
        mixup_alpha: float = 1.0,
        label_smoothing: float = 0.2,
        gp_weight: float = 10.0,
    ) -> Tuple["VICEAgent", Dict]:
        """
        Update the VICE reward classifier using BCE loss with mixup and gradient penalty.
        
        NOTE: Assumes that the second half of the batch contains goal/success images (labels = 1).
        
        Args:
            batch: Training batch
            pmap_axis: Parallel axis (not used in single-GPU PyTorch)
            mixup_alpha: Alpha parameter for mixup regularization
            label_smoothing: Label smoothing factor
            gp_weight: Weight for gradient penalty regularization
            
        Returns:
            Updated agent and info dict
        """
        new_agent = self
        
        # Unpack batch if necessary
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        
        observations = batch["next_observations"]
        
        # Apply data augmentation
        aug_generator = next_rng()
        aug_observations = self.data_augmentation_fn(observations, aug_generator)
        
        all_info = {}
        
        for image_key in self.config["image_keys"]:
            pixels = observations[image_key]
            batch_size = pixels.shape[0]
            
            # Split into observations and goals
            pixels = observations[image_key][: batch_size // 2]
            aug_pixels = aug_observations[image_key][: batch_size // 2]
            goal_pixels = observations[image_key][batch_size // 2 :]
            aug_goal_pixels = aug_observations[image_key][batch_size // 2 :]
            
            # Concatenate all images
            all_obs_pixels = torch.cat([pixels, aug_pixels], dim=0)
            all_goal_pixels = torch.cat([goal_pixels, aug_goal_pixels], dim=0)
            all_pixels = torch.cat([all_goal_pixels, all_obs_pixels], dim=0)
            
            # Create labels (goals=1, observations=0)
            ones = torch.ones((batch_size, 1), device=pixels.device)
            zeros = torch.zeros((batch_size, 1), device=pixels.device)
            y_batch = torch.cat([ones, zeros], dim=0).squeeze(-1)
            
            # Label smoothing
            y_batch = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing
            
            # Encode images
            encoded = self.encode_images({image_key: all_pixels}, train=True)
            
            # Mixup
            def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
                """Perform mixup regularization."""
                if alpha > 0:
                    lam = np.random.beta(alpha, alpha)
                else:
                    lam = 1.0
                
                batch_size_mixup = x.shape[0]
                index = torch.randperm(batch_size_mixup, device=x.device)
                
                mixed_x = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]
                
                return mixed_x, y_a, y_b, lam
            
            mix_encoded, y_a, y_b, lam = mixup_data(encoded, y_batch, mixup_alpha)
            
            # Interpolate for Gradient Penalty
            epsilon = torch.rand(
                (len(mix_encoded) // 2, *([1] * (len(mix_encoded.shape) - 1))),
                device=mix_encoded.device
            )
            gp_encoded = (
                epsilon * mix_encoded[: len(mix_encoded) // 2]
                + (1 - epsilon) * mix_encoded[len(mix_encoded) // 2 :]
            )
            
            # Enable gradient for GP
            gp_encoded.requires_grad_(True)
            
            # Zero gradients
            self.state.optimizers["vice"].zero_grad()
            
            # Mixup loss
            y_hat_mix = self.state.models["vice"](
                {image_key: mix_encoded},
                train=True,
                classify_encoded=True,
            )
            
            bce_loss_a = F.binary_cross_entropy_with_logits(y_hat_mix, y_a)
            bce_loss_b = F.binary_cross_entropy_with_logits(y_hat_mix, y_b)
            bce_loss = lam * bce_loss_a + (1 - lam) * bce_loss_b
            
            # Gradient penalty
            y_hat_gp = self.state.models["vice"](
                {image_key: gp_encoded},
                train=True,
                classify_encoded=True,
            )
            
            # Compute gradients w.r.t. input
            gradients = torch.autograd.grad(
                outputs=y_hat_gp,
                inputs=gp_encoded,
                grad_outputs=torch.ones_like(y_hat_gp),
                create_graph=True,
                retain_graph=True,
            )[0]
            
            # Flatten and compute norms
            gradients = gradients.reshape(gradients.shape[0], -1)
            grad_norms = torch.sqrt(torch.sum(gradients ** 2 + 1e-6, dim=1))
            grad_penalty = torch.mean((grad_norms - 1) ** 2)
            
            # Total loss
            total_loss = bce_loss + gp_weight * grad_penalty
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            self.state.optimizers["vice"].step()
            
            # Collect info
            all_info.update({
                "bce_loss": bce_loss.item(),
                "grad_norm": grad_norms.mean().item(),
            })
        
        self.state.step += 1
        
        return new_agent, all_info
    
    def vice_reward(self, observation: Data) -> torch.Tensor:
        """
        Compute VICE reward from observations.
        
        Args:
            observation: Observations
            
        Returns:
            Predicted rewards (sigmoid of classifier output)
        """
        with torch.no_grad():
            logits = self.state.models["vice"](observation, train=False)
            rewards = torch.sigmoid(logits)
        
        return rewards
    
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["VICEAgent", Dict]:
        """
        Update only the critics using VICE rewards.
        
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
        
        # Compute VICE rewards (binary: >= 0.5 -> 1.0, < 0.5 -> 0.0)
        rewards = (self.vice_reward(next_obs) >= 0.5).float()
        
        # Update batch with augmented observations and VICE rewards
        augmented_batch = {**batch}
        augmented_batch["observations"] = obs
        augmented_batch["next_observations"] = next_obs
        augmented_batch["rewards"] = rewards
        
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
    
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["VICEAgent", Dict]:
        """
        Fast high-UTD version of update using VICE rewards.
        
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
        
        # Compute VICE rewards
        rewards = (self.vice_reward(next_obs) >= 0.5).float()
        
        # Update batch with augmented observations and VICE rewards
        augmented_batch = {**batch}
        augmented_batch["observations"] = obs
        augmented_batch["next_observations"] = next_obs
        augmented_batch["rewards"] = rewards
        
        # Call parent class's update_high_utd
        new_agent, info = SACAgent.update_high_utd(
            new_agent, augmented_batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )
        
        info["vice_rewards"] = rewards.mean().item()
        
        return new_agent, info
