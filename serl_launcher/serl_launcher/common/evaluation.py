import math
from collections import defaultdict
from typing import Dict, Callable

import gym
import torch
import numpy as np


def supply_rng(f, generator: torch.Generator = None):
    """
    Decorator to supply RNG to a function.
    
    Args:
        f: Function to wrap
        generator: PyTorch generator for random number generation
        
    Returns:
        Wrapped function that automatically provides fresh RNG
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)
    
    def wrapped(*args, **kwargs):
        nonlocal generator
        # Create new generator for this call
        new_gen = torch.Generator()
        seed = torch.randint(0, 2**32, (1,), generator=generator).item()
        new_gen.manual_seed(seed)
        return f(*args, seed=new_gen, **kwargs)
    
    return wrapped


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_info(info: dict) -> dict:
    """
    Filter out non-scalar information from info dict.
    
    Args:
        info: Info dictionary from environment
        
    Returns:
        Filtered info dictionary
    """
    filter_keys = [
        "object_names",
        "target_object",
        "initial_positions",
        "target_position",
        "goal",
    ]
    filtered = info.copy()
    for k in filter_keys:
        if k in filtered:
            del filtered[k]
    return filtered


def add_to(dict_of_lists: dict, single_dict: dict):
    """
    Add values from single_dict to corresponding lists in dict_of_lists.
    
    Args:
        dict_of_lists: Dictionary of lists
        single_dict: Dictionary of values to add
    """
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    policy_fn: Callable,
    env: gym.Env,
    num_episodes: int,
    return_trajectories: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a policy on an environment.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        return_trajectories: Whether to return full trajectories
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    trajectories = [] if return_trajectories else None
    
    for episode_idx in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(filter_info(info)))
        
        done = False
        episode_trajectory = [] if return_trajectories else None
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from policy
            action = policy_fn(observation)
            
            # Convert action to numpy if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            if return_trajectories:
                episode_trajectory.append({
                    'observation': observation.copy() if isinstance(observation, np.ndarray) else observation,
                    'action': action.copy() if isinstance(action, np.ndarray) else action,
                })
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            add_to(stats, flatten(filter_info(info)))
        
        # Add final info
        add_to(stats, flatten(filter_info(info), parent_key="final"))
        
        # Add episode statistics
        stats['episode_reward'].append(episode_reward)
        stats['episode_length'].append(episode_length)
        
        if return_trajectories:
            trajectories.append(episode_trajectory)
    
    # Compute summary statistics
    summary_stats = {}
    for k, v in stats.items():
        if len(v) > 0 and isinstance(v[0], (int, float, np.number)):
            summary_stats[f"{k}_mean"] = np.mean(v)
            summary_stats[f"{k}_std"] = np.std(v)
            summary_stats[f"{k}_min"] = np.min(v)
            summary_stats[f"{k}_max"] = np.max(v)
    
    if return_trajectories:
        return summary_stats, trajectories
    else:
        return summary_stats


def evaluate_with_success_rate(
    policy_fn: Callable,
    env: gym.Env,
    num_episodes: int,
    success_key: str = "success",
) -> Dict[str, float]:
    """
    Evaluate a policy and compute success rate.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        success_key: Key in info dict that indicates success
        
    Returns:
        Dictionary of evaluation statistics including success rate
    """
    stats = evaluate(policy_fn, env, num_episodes)
    
    # Extract success rate if available
    if f"final.{success_key}_mean" in stats:
        stats['success_rate'] = stats[f"final.{success_key}_mean"]
    elif f"{success_key}_mean" in stats:
        stats['success_rate'] = stats[f"{success_key}_mean"]
    
    return stats
