import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from serl_launcher.common.typing import Params, PRNGKey


def default_init(gain: float = 1.0):
    """Default initialization using Xavier uniform."""
    def init_fn(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return init_fn


class ModuleDict(nn.Module):
    """
    Utility class for wrapping a dictionary of modules. This is useful when you have multiple modules that you want to
    initialize all at once (creating a single `params` dictionary), but you want to be able to call them separately
    later. As a bonus, the modules may have sub-modules nested inside them that share parameters (e.g. an image encoder)
    and PyTorch will automatically handle this without duplicating the parameters.

    To call the modules, pass the name of the module as the `name` kwarg, and then pass the arguments
    to the module as additional args or kwargs.

    Example usage:
    ```
    shared_encoder = Encoder()
    actor = Actor(encoder=shared_encoder)
    critic = Critic(encoder=shared_encoder)

    model_def = ModuleDict({"actor": actor, "critic": critic})

    actor_output = model_def(example_obs, name="actor")
    critic_output = model_def(example_obs, action=example_action, name="critic")
    ```
    """

    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__()
        self.modules_dict = nn.ModuleDict(modules)

    def forward(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules_dict.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules_dict.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules_dict[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules_dict[key](*value)
                else:
                    out[key] = self.modules_dict[key](value)
            return out

        return self.modules_dict[name](*args, **kwargs)


@dataclass
class TrainState:
    """
    Custom TrainState class to replace Flax's TrainState.

    Adds support for holding target params and updating them via polyak
    averaging. Adds the ability to hold an rng key for dropout.

    Attributes:
        step: The current training step.
        model: The PyTorch model.
        target_model: The target model (for Q-learning).
        optimizers: Dictionary of optimizers.
        rng: The internal rng state.
    """
    step: int = 0
    model: nn.Module = field(default=None)
    target_model: nn.Module = field(default=None)
    optimizers: Dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    rng: PRNGKey = field(default=None)

    def target_update(self, tau: float) -> "TrainState":
        """
        Performs an update of the target params via polyak averaging. The new
        target params are given by:

            new_target_params = tau * params + (1 - tau) * target_params
        """
        if self.target_model is not None:
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_model.parameters(), self.model.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
        return self

    def apply_gradients(self) -> "TrainState":
        """
        Apply gradients using the optimizers.
        Note: In PyTorch, gradients are computed and applied separately.
        """
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        self.step += 1
        return self

    def zero_grad(self):
        """Zero out all gradients."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        target_model: nn.Module = None,
        rng: PRNGKey = None,
    ) -> "TrainState":
        """
        Initializes a new train state.

        Args:
            model: The PyTorch model.
            optimizers: Dictionary of optimizers.
            target_model: The target model (optional).
            rng: The rng generator used for dropout etc.
        """
        if rng is None:
            rng = torch.Generator()
            rng.manual_seed(0)
        
        return cls(
            step=0,
            model=model,
            target_model=target_model,
            optimizers=optimizers,
            rng=rng,
        )


def tree_map(fn: Callable, *trees):
    """
    Simple tree_map implementation for PyTorch.
    Applies fn to corresponding leaves of the trees.
    """
    if len(trees) == 0:
        return None
    
    if isinstance(trees[0], dict):
        return {k: tree_map(fn, *[t[k] for t in trees]) for k in trees[0].keys()}
    elif isinstance(trees[0], (list, tuple)):
        tree_type = type(trees[0])
        return tree_type(tree_map(fn, *[t[i] for t in trees]) for i in range(len(trees[0])))
    else:
        return fn(*trees)
