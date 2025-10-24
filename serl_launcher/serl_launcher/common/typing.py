from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import torch

PRNGKey = torch.Generator
Params = Dict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, torch.Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
