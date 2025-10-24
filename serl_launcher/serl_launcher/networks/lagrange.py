from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagrangeMultiplier(nn.Module):
    """
    Lagrange multiplier for constrained optimization.
    
    Args:
        init_value: Initial value of the multiplier
        constraint_shape: Shape of the constraint
        constraint_type: Type of constraint ("eq", "leq", "geq")
        parameterization: Parameterization method ("softplus", "exp"), or None for equality
    """
    
    def __init__(
        self,
        init_value: float = 1.0,
        constraint_shape: Sequence[int] = (),
        constraint_type: str = "eq",  # One of ("eq", "leq", "geq")
        parameterization: Optional[str] = None,  # One of ("softplus", "exp"), or None
    ):
        super().__init__()
        self.constraint_type = constraint_type
        self.parameterization = parameterization
        self.constraint_shape = constraint_shape
        
        # Validate inputs
        if constraint_type != "eq":
            assert init_value > 0, \
                "Inequality constraints must have non-negative initial multiplier values"
            
            if parameterization == "softplus":
                # Inverse softplus: softplus^{-1}(x) = log(exp(x) - 1)
                init_value = torch.log(torch.exp(torch.tensor(init_value)) - 1).item()
            elif parameterization == "exp":
                init_value = torch.log(torch.tensor(init_value)).item()
            else:
                raise ValueError(f"Invalid multiplier parameterization {parameterization}")
        else:
            assert parameterization is None, \
                "Equality constraints must have no parameterization"
        
        # Create parameter
        if constraint_shape == () or len(constraint_shape) == 0:
            shape = (1,)
        else:
            shape = constraint_shape
        
        self.multiplier = nn.Parameter(
            torch.full(shape, init_value, dtype=torch.float32)
        )
    
    def forward(
        self,
        lhs: Optional[torch.Tensor] = None,
        rhs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Lagrange penalty.
        
        Args:
            lhs: Left-hand side of constraint
            rhs: Right-hand side of constraint (default 0)
            
        Returns:
            If lhs is None, returns the raw multiplier.
            Otherwise, returns the Lagrange penalty.
        """
        multiplier = self.multiplier
        
        # Apply parameterization if needed
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = F.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = torch.exp(multiplier)
            else:
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")
        
        # Return raw multiplier if no constraint provided
        if lhs is None:
            return multiplier
        
        # Compute Lagrange penalty
        if rhs is None:
            rhs = torch.zeros_like(lhs)
        
        diff = lhs - rhs
        
        # Check shapes match
        assert diff.shape == multiplier.shape or multiplier.numel() == 1, \
            f"Shape mismatch: diff {diff.shape} vs multiplier {multiplier.shape}"
        
        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff
        else:
            raise ValueError(f"Invalid constraint type {self.constraint_type}")


# Convenience functions for common constraint types
def GeqLagrangeMultiplier(init_value: float = 1.0, constraint_shape: Sequence[int] = ()):
    """Greater-or-equal constraint Lagrange multiplier."""
    return LagrangeMultiplier(
        init_value=init_value,
        constraint_shape=constraint_shape,
        constraint_type="geq",
        parameterization="softplus"
    )


def LeqLagrangeMultiplier(init_value: float = 1.0, constraint_shape: Sequence[int] = ()):
    """Less-or-equal constraint Lagrange multiplier."""
    return LagrangeMultiplier(
        init_value=init_value,
        constraint_shape=constraint_shape,
        constraint_type="leq",
        parameterization="softplus"
    )
