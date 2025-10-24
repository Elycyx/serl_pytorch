"""
Simple example of using PyTorch SAC with SERL.

This demonstrates:
1. Creating a SAC agent
2. Training loop with replay buffer
3. Evaluation

Usage:
    python pytorch_sac_example.py
"""

import torch
import numpy as np
import gym
from typing import Dict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.common.evaluation import evaluate


def create_simple_env():
    """Create a simple gym environment for testing."""
    # You can replace this with any gym environment
    env = gym.make('Pendulum-v1')
    return env


def collect_episode(env, agent, replay_buffer, epsilon=0.1):
    """
    Collect one episode of data.
    
    Args:
        env: Gym environment
        agent: SAC agent
        replay_buffer: Replay buffer to store transitions
        epsilon: Random action probability
        
    Returns:
        Episode return
    """
    observation, _ = env.reset()
    done = False
    episode_return = 0
    episode_length = 0
    
    while not done:
        # Convert observation to torch tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
        
        # Select action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(obs_tensor, argmax=False)
            action = action[0]  # Remove batch dimension
        
        # Step environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        transition = {
            'observations': observation,
            'actions': action,
            'next_observations': next_observation,
            'rewards': np.array([reward], dtype=np.float32),
            'masks': np.array([1.0 - float(done)], dtype=np.float32),
            'dones': np.array([done], dtype=bool),
        }
        replay_buffer.insert(transition)
        
        observation = next_observation
        episode_return += reward
        episode_length += 1
    
    return episode_return, episode_length


def train_sac(
    env_name: str = 'Pendulum-v1',
    num_epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    start_training: int = 1000,
    eval_episodes: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train SAC agent.
    
    Args:
        env_name: Name of gym environment
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        batch_size: Batch size for training
        start_training: Steps before starting training
        eval_episodes: Number of evaluation episodes
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"Training SAC on {env_name}")
    print(f"Device: {device}")
    
    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=100000,
    )
    
    # Create example batch for initialization
    example_obs = torch.FloatTensor(env.observation_space.sample()).unsqueeze(0)
    example_action = env.action_space.sample().reshape(1, -1)
    
    # Create SAC agent
    agent = SACAgent.create_states(
        observations=example_obs,
        actions=example_action,
        critic_network_kwargs={
            'hidden_dims': [256, 256],
        },
        policy_network_kwargs={
            'hidden_dims': [256, 256],
        },
        critic_ensemble_size=2,
        discount=0.99,
        soft_target_update_rate=0.005,
        target_entropy=-action_dim,
        device=torch.device(device),
    )
    
    print("Agent created successfully!")
    
    # Training loop
    total_steps = 0
    for epoch in range(num_epochs):
        # Collect data
        epoch_return = 0
        epoch_length = 0
        num_episodes = 0
        
        while epoch_length < steps_per_epoch:
            episode_return, episode_len = collect_episode(
                env, agent, replay_buffer, epsilon=0.1
            )
            epoch_return += episode_return
            epoch_length += episode_len
            num_episodes += 1
            total_steps += episode_len
        
        avg_return = epoch_return / num_episodes
        
        # Training
        if total_steps >= start_training:
            train_info = {}
            num_updates = steps_per_epoch
            
            for _ in range(num_updates):
                # Sample batch
                batch = replay_buffer.sample(batch_size)
                
                # Convert to torch tensors
                batch_torch = {
                    'observations': torch.FloatTensor(batch['observations']).to(device),
                    'actions': torch.FloatTensor(batch['actions']).to(device),
                    'next_observations': torch.FloatTensor(batch['next_observations']).to(device),
                    'rewards': torch.FloatTensor(batch['rewards']).to(device),
                    'masks': torch.FloatTensor(batch['masks']).to(device),
                    'dones': torch.BoolTensor(batch['dones']).to(device),
                }
                
                # Update agent
                agent, info = agent.update(batch_torch)
                
                # Accumulate info
                for k, v in info.items():
                    if k not in train_info:
                        train_info[k] = []
                    train_info[k].append(v)
            
            # Average training info
            train_info = {k: np.mean(v) for k, v in train_info.items()}
        else:
            train_info = {}
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            def policy_fn(obs):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                return agent.sample_actions(obs_tensor, argmax=True)
            
            eval_info = evaluate(policy_fn, eval_env, eval_episodes)
            eval_return = eval_info.get('episode_reward_mean', 0)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Return: {avg_return:.2f}")
            print(f"  Eval Return: {eval_return:.2f}")
            if train_info:
                print(f"  Critic Loss: {train_info.get('critic_loss', 0):.4f}")
                print(f"  Actor Loss: {train_info.get('actor_loss', 0):.4f}")
                print(f"  Temperature: {train_info.get('temperature', 0):.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Return: {avg_return:.2f}")
    
    env.close()
    eval_env.close()
    
    print("\nTraining completed!")
    return agent


def main():
    """Main function."""
    # Train SAC on Pendulum
    agent = train_sac(
        env_name='Pendulum-v1',
        num_epochs=100,
        steps_per_epoch=1000,
        batch_size=256,
    )
    
    print("Training finished!")
    
    # Save agent (optional)
    # torch.save(agent.state.model.state_dict(), 'sac_model.pt')


if __name__ == '__main__':
    main()

