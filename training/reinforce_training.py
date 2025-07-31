import os
import sys
import csv
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.custom_env import AngazaEnv
from stable_baselines3.common.monitor import Monitor

import csv
import os
from datetime import datetime

#Global Hyperparameters
lr = 0.0005
gamma = 0.99
hidden_dim = 128
max_grad_norm = 1.0
reward_normalization = True
success_threshold = 7
log_interval = 50
eval_episodes = 50
entropy_coef = 0.001
total_episodes = 1500


def log_experiment_to_csv(log_path, config, results):
    """Append hyperparams + results to a CSV log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fieldnames = list(config.keys()) + list(results.keys()) + ["timestamp"]

    # Check if file exists to decide whether to write headers
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        log_entry = {**config, **results, "timestamp": datetime.now().isoformat()}
        writer.writerow(log_entry)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=hidden_dim, output_dim=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def discount_rewards(rewards):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted

def evaluate_policy(policy, env, episodes=eval_episodes):
    policy.eval()
    success_count = 0
    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)
        if ep_reward >= 7:  # success threshold
            success_count += 1

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / episodes
    print(f"\nEvaluation -> Avg reward: {avg_reward:.2f} | Success rate: {success_rate:.2%}")
    policy.train()

def train_reinforce(total_episodes=total_episodes, save_path="models/pg/angaza_reinforce.pth"):
    env = Monitor(AngazaEnv(), filename="models/pg/monitor_log")
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    #success_counter = 0
    recent_rewards = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        assert isinstance(obs, (list, np.ndarray)), f"obs is not a list/array: {obs}"
        assert len(obs) == 2, f"obs has wrong shape: {obs}"
        log_probs = []
        rewards = []
        done = False
        total_reward = 0

        while not done:
            #obs_tensor = torch.tensor(obs, dtype=torch.float32)
            obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            #obs, reward, done, _ = env.step(action.item())
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            total_reward += reward

        if total_reward < -15:
            continue  # skip this update

        #clipping rewards before discounting
        rewards = np.clip(rewards, -10, 10)
        discounted = discount_rewards(rewards)

        # Reward normalisation
        discounted = discount_rewards(rewards)
        discounted = torch.tensor(discounted, dtype=torch.float32)
        if reward_normalization and len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

        # Compute entropy bonus
        obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
        logits = policy(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()

        log_probs = torch.stack(log_probs)
        loss = -(log_probs * discounted).sum() - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        episode_rewards.append(total_reward)

        #Moving average for recent rewards
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        mean_recent_reward = np.mean(recent_rewards)

        #Periodic Evaluation
        if ep % log_interval == 0:
            print(f"Ep {ep} | Mean reward (last 50): {mean_recent_reward:.2f} | Loss: {loss.item():.2f}")
            evaluate_policy(policy, AngazaEnv(), episodes=eval_episodes)

        #Early stopping if success threshold is met
        if mean_recent_reward >= success_threshold:
            print(f"Consistent success threshold reached at episode {ep}")
            break

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    np.save("models/pg/reinforce_rewards.npy", episode_rewards)
    print(f"REINFORCE model saved to {save_path}")

        # Log results + hyperparams
    final_mean_reward = float(np.mean(episode_rewards[-50:]))
    results = {
        "final_mean_reward": final_mean_reward,
        "total_episodes": total_episodes,
        "final_loss": loss.item() if 'loss' in locals() else 0.0,
    }

    config = {
        "learning_rate": lr,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "max_grad_norm": max_grad_norm,
        "reward_normalization": reward_normalization,
        "success_threshold": success_threshold,
        "log_interval": log_interval,
        "eval_episodes": eval_episodes,
        "env_name": "AngazaEnv"
    }

    log_path = "logs/pg/reinforce_logs.csv"
    log_experiment_to_csv(log_path, config, results)
    print(f"Experiment logged to {log_path}")

     # Run final evaluation
    evaluate_policy(policy, AngazaEnv(), episodes=eval_episodes)



if __name__ == "__main__":
    train_reinforce()
