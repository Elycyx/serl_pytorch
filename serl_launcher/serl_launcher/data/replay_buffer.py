import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gym
import torch
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    """Initialize replay buffer storage for a given observation space."""
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError(f"Unsupported space type: {type(obs_space)}")


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    """Insert data recursively into nested dictionary structure."""
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys(), (
            f"Key mismatch: {dataset_dict.keys()} vs {data_dict.keys()}"
        )
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError(f"Unsupported type: {type(dataset_dict)}")


class ReplayBuffer(Dataset):
    """
    Replay buffer for storing and sampling transitions.
    
    Args:
        observation_space: Observation space
        action_space: Action space
        capacity: Maximum buffer size
        next_observation_space: Next observation space (default: same as observation_space)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        """
        Insert a transition into the replay buffer.
        
        Args:
            data_dict: Dictionary containing transition data
        """
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(
        self,
        queue_size: int = 2,
        sample_args: dict = {},
        device: torch.device = None,
    ) -> Iterator:
        """
        Get an iterator that yields batches from the replay buffer.
        
        Args:
            queue_size: Number of batches to prefetch
            sample_args: Arguments to pass to sample()
            device: Device to place batches on
            
        Yields:
            Batches of transitions
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                # Convert to torch tensors and move to device
                data_torch = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        data_torch[k] = {
                            sub_k: torch.as_tensor(sub_v, device=device)
                            for sub_k, sub_v in v.items()
                        }
                    else:
                        data_torch[k] = torch.as_tensor(v, device=device)
                queue.append(data_torch)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int) -> Tuple[int, DatasetDict]:
        """
        Download a range of transitions from the buffer.
        
        Args:
            from_idx: Starting index
            to_idx: Ending index (exclusive)
            
        Returns:
            Tuple of (to_idx, data_dict)
        """
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self) -> Iterator[DatasetDict]:
        """
        Get an iterator that downloads all data from the buffer.
        
        Yields:
            Batches of transitions
        """
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise StopIteration(f"Reached end of buffer: {last_idx} >= {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
    
    @property
    def size(self) -> int:
        """Current size of the replay buffer."""
        return self._size
    
    @property
    def capacity(self) -> int:
        """Maximum capacity of the replay buffer."""
        return self._capacity
    
    def is_full(self) -> bool:
        """Check if the replay buffer is full."""
        return self._size >= self._capacity
