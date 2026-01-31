"""
Training script for Picar-X robot using PPO algorithm.
Optimized for RTX 3090 GPU.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import json
from datetime import datetime

# Import environment
from picarx_env import PicarXEnv, PicarXEnvVec


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.

    Architecture:
    - Shared CNN encoder for image processing
    - Separate actor (policy) and critic (value) heads
    """

    def __init__(self, obs_dim, action_dim, img_size=96):
        super(ActorCritic, self).__init__()

        self.img_size = img_size
        self.img_pixels = img_size * img_size * 3

        # CNN encoder for image (first part of observation)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 96x96 -> 48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48x48 -> 24x24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 24x24 -> 12x12
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 12x12 -> 6x6
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        cnn_output_size = 64 * 6 * 6  # 2304

        # Joint info encoder (positions + velocities = 10 features)
        self.joint_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combined feature size
        combined_size = cnn_output_size + 64

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch_size, obs_dim]

        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: State value estimate
        """
        batch_size = obs.shape[0]

        # Split observation into image and joint info
        img_flat = obs[:, : self.img_pixels]
        joint_info = obs[:, self.img_pixels :]

        # Reshape image to [batch, channels, height, width]
        img = img_flat.view(batch_size, 3, self.img_size, self.img_size)

        # Encode image
        img_features = self.cnn(img)

        # Encode joint info
        joint_features = self.joint_encoder(joint_info)

        # Combine features
        combined = torch.cat([img_features, joint_features], dim=1)

        # Shared processing
        shared_features = self.shared_fc(combined)

        # Actor output (scale to [-100, 100])
        action_mean = self.actor_mean(shared_features) * 100.0
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        # Critic output
        value = self.critic(shared_features)

        return action_mean, action_std, value

    def get_action_and_value(self, obs, action=None):
        """Get action, log probability, and value for PPO update."""
        action_mean, action_std, value = self.forward(obs)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)


class PPOTrainer:
    """PPO trainer with support for vectorized environments."""

    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=10,
        batch_size=64,
        device="cuda",
    ):
        self.env = env
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Get observation and action dimensions
        obs, _ = env.reset()
        self.obs_dim = obs.shape[0]
        self.action_dim = 2  # Left and right motors

        # Initialize network and optimizer
        self.policy = ActorCritic(self.obs_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def collect_rollouts(self, num_steps):
        """Collect rollout data for PPO update."""
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        log_probs_list = []
        values_list = []

        obs, _ = self.env.reset()

        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(
                    obs_tensor
                )

            action_np = action.cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            # Store data
            obs_list.append(obs)
            actions_list.append(action_np)
            rewards_list.append(reward)
            dones_list.append(terminated or truncated)
            log_probs_list.append(log_prob.cpu().numpy())
            values_list.append(value.cpu().numpy())

            obs = next_obs

            # Track episode stats
            if terminated or truncated:
                self.episode_rewards.append(info.get("episode_reward", reward))
                self.episode_lengths.append(info.get("episode_length", step))

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(actions_list)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards_list)).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(dones_list)).to(self.device)
        log_probs_tensor = torch.FloatTensor(np.array(log_probs_list)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(values_list)).to(self.device)

        return (
            obs_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            log_probs_tensor,
            values_tensor,
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, rewards, dones, old_log_probs, old_values):
        """Perform PPO update."""
        # Compute advantages and returns
        with torch.no_grad():
            _, _, _, next_value = self.policy.get_action_and_value(obs[-1:])

        advantages, returns = self.compute_gae(
            rewards, old_values, dones, next_value.item()
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(self.num_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(batch_obs, batch_actions)
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        num_updates = self.num_epochs * (len(obs) // self.batch_size + 1)

        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def train(self, total_timesteps=1000000, rollout_length=2048, save_interval=10000):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Total timesteps: {total_timesteps}")

        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"checkpoints/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        # Save hyperparameters
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(
                {
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_epsilon": self.clip_epsilon,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "total_timesteps": total_timesteps,
                    "rollout_length": rollout_length,
                },
                f,
                indent=2,
            )

        timesteps = 0
        iteration = 0

        while timesteps < total_timesteps:
            iteration += 1

            # Collect rollouts
            print(f"\nIteration {iteration} - Collecting rollouts...")
            obs, actions, rewards, dones, log_probs, values = self.collect_rollouts(
                rollout_length
            )

            # Update policy
            print("Updating policy...")
            update_info = self.update(obs, actions, rewards, dones, log_probs, values)

            timesteps += rollout_length

            # Print stats
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

            print(f"Timesteps: {timesteps}/{total_timesteps}")
            print(f"Avg Episode Reward: {avg_reward:.2f}")
            print(f"Avg Episode Length: {avg_length:.2f}")
            print(f"Loss: {update_info['loss']:.4f}")
            print(f"Policy Loss: {update_info['policy_loss']:.4f}")
            print(f"Value Loss: {update_info['value_loss']:.4f}")
            print(f"Entropy: {update_info['entropy']:.4f}")

            # Save checkpoint
            if timesteps % save_interval < rollout_length:
                checkpoint_path = f"{save_dir}/checkpoint_{timesteps}.pt"
                torch.save(
                    {
                        "timesteps": timesteps,
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "avg_reward": avg_reward,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = f"{save_dir}/final_model.pt"
        torch.save(self.policy.state_dict(), final_path)
        print(f"\nTraining complete! Final model saved: {final_path}")

        return self.policy


def main():
    """Main entry point."""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Create environment
    print("\nInitializing environment...")
    env = PicarXEnv(
        num_envs=1,
        render_mode=None,  # Set to "human" for visualization (slower)
        max_steps=1000,
        img_width=96,
        img_height=96,
    )

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=10,
        batch_size=64,
        device=device,
    )

    # Train
    print("\nStarting training...")
    try:
        policy = trainer.train(
            total_timesteps=100000,  # Start with 100k for testing
            rollout_length=2048,
            save_interval=10000,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
