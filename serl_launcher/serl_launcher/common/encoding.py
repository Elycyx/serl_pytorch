from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network (can be a ModuleDict for multiple image keys).
        use_proprio: Whether to concatenate proprioception (after encoding).
        proprio_latent_dim: Dimension for proprioception encoding.
        enable_stacking: Whether to enable frame stacking.
        image_keys: Keys for image observations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        use_proprio: bool,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        image_keys: Iterable[str] = ("image",),
    ):
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.ModuleDict) else nn.ModuleDict({'image': encoder})
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = tuple(image_keys)
        
        # Proprioception projection layers (created lazily)
        self.proprio_dense = None
        self.proprio_norm = None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Encode observations.
        
        Args:
            observations: Dictionary of observations
            train: Whether in training mode
            stop_gradient: Whether to stop gradient flow
            is_encoded: Whether images are already encoded
            
        Returns:
            Encoded observation tensor
        """
        # Encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:  # (T, H, W, C)
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:  # (B, T, H, W, C)
                        image = rearrange(image, "B T H W C -> B H W (T C)")
            
            # Encode image
            if hasattr(self.encoder[image_key], 'forward'):
                # Standard PyTorch module
                if 'encode' in self.encoder[image_key].forward.__code__.co_varnames:
                    image = self.encoder[image_key](image, train=train, encode=not is_encoded)
                else:
                    image = self.encoder[image_key](image, train=train)
            else:
                image = self.encoder[image_key](image)
            
            if stop_gradient:
                image = image.detach()
            
            encoded.append(image)
        
        encoded = torch.cat(encoded, dim=-1)
        
        if self.use_proprio and "state" in observations:
            # Project state to embeddings as well
            state = observations["state"]
            
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:  # (T, C)
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:  # (B, T, C)
                    state = rearrange(state, "B T C -> B (T C)")
            
            # Lazy initialization of proprioception layers
            if self.proprio_dense is None:
                self.proprio_dense = nn.Linear(state.shape[-1], self.proprio_latent_dim).to(state.device)
                nn.init.xavier_uniform_(self.proprio_dense.weight)
                self.proprio_norm = nn.LayerNorm(self.proprio_latent_dim).to(state.device)
            
            state = self.proprio_dense(state)
            state = self.proprio_norm(state)
            state = torch.tanh(state)
            encoded = torch.cat([encoded, state], dim=-1)
        
        return encoded


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        goal_encoder: Optional[nn.Module],
        use_proprio: bool,
        stop_gradient: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.goal_encoder = goal_encoder
        self.use_proprio = use_proprio
        self.stop_gradient_flag = stop_gradient

    def forward(
        self,
        observations_and_goals: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Encode observations and goals.
        
        Args:
            observations_and_goals: Tuple of (observations, goals)
            
        Returns:
            Encoded tensor
        """
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat goals so that there's a goal for each frame
            goal_image = repeat(
                goals["image"], "B H W C -> (B repeat) H W C", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            # early goal concat
            encoder_inputs = torch.cat([obs_image, goal_image], dim=-1)
            encoding = self.encoder(encoder_inputs)
        else:
            # late fusion
            encoding = self.encoder(obs_image)
            goal_encoding = self.goal_encoder(goals["image"])
            encoding = torch.cat([encoding, goal_encoding], dim=-1)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio and "proprio" in observations:
            encoding = torch.cat([encoding, observations["proprio"]], dim=-1)

        if self.stop_gradient_flag:
            encoding = encoding.detach()

        return encoding


class LCEncodingWrapper(nn.Module):
    """
    Encodes observations and language instructions into a single flat encoding.

    Takes a tuple (observations, goals) as input, where goals contains the language instruction.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        use_proprio: bool,
        stop_gradient: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_proprio = use_proprio
        self.stop_gradient_flag = stop_gradient

    def forward(
        self,
        observations_and_goals: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Encode observations and language instructions.
        
        Args:
            observations_and_goals: Tuple of (observations, goals)
            
        Returns:
            Encoded tensor
        """
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat language so that there's an instruction for each frame
            language = repeat(
                goals["language"], "B E -> (B repeat) E", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            language = goals["language"]

        # Encode with conditioning on language
        encoding = self.encoder(obs_image, cond_var=language)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio and "proprio" in observations:
            encoding = torch.cat([encoding, observations["proprio"]], dim=-1)

        if self.stop_gradient_flag:
            encoding = encoding.detach()

        return encoding
