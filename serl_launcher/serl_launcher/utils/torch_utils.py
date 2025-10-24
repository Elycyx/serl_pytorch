"""PyTorch utility functions to replace JAX utilities."""

import torch
from typing import Optional


def batch_to_device(batch, device: Optional[torch.device] = None):
    """Move a batch (nested dict/list) to device."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


class TorchRNG:
    """A convenient stateful PyTorch RNG wrapper. Can be used to wrap RNG inside functions."""

    @classmethod
    def from_seed(cls, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return cls(generator)

    def __init__(self, generator: torch.Generator):
        self.generator = generator

    def __call__(self, keys=None):
        """
        Generate random numbers.
        
        Args:
            keys: If None, returns a new generator.
                  If int, returns a tuple of generators.
                  If list, returns a dict of generators.
        """
        if keys is None:
            # Create a new generator with a random seed from current generator
            new_gen = torch.Generator(device=self.generator.device)
            new_gen.manual_seed(torch.randint(0, 2**32, (1,), generator=self.generator).item())
            return new_gen
        elif isinstance(keys, int):
            # Return tuple of generators
            generators = []
            for _ in range(keys):
                new_gen = torch.Generator(device=self.generator.device)
                new_gen.manual_seed(torch.randint(0, 2**32, (1,), generator=self.generator).item())
                generators.append(new_gen)
            return tuple(generators)
        else:
            # Return dict of generators
            generators = {}
            for key in keys:
                new_gen = torch.Generator(device=self.generator.device)
                new_gen.manual_seed(torch.randint(0, 2**32, (1,), generator=self.generator).item())
                generators[key] = new_gen
            return generators


def wrap_function_with_rng(generator: torch.Generator):
    """To be used as decorator, automatically bookkeep a RNG for the wrapped function."""

    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal generator
            # Create a new generator for this call
            new_gen = torch.Generator(device=generator.device)
            new_gen.manual_seed(torch.randint(0, 2**32, (1,), generator=generator).item())
            return function(new_gen, *args, **kwargs)

        return wrapped

    return wrap_function


# Global RNG for convenience
_torch_utils_rng = None


def init_rng(seed: int):
    """Initialize global RNG with seed."""
    global _torch_utils_rng
    _torch_utils_rng = TorchRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    """Get next RNG from global RNG."""
    global _torch_utils_rng
    if _torch_utils_rng is None:
        init_rng(0)
    return _torch_utils_rng(*args, **kwargs)

