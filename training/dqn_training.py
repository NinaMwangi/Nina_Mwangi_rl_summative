import os
import gym
import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import AngazaEnv
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Optional: for reproducibility
SEED = 123
learning_rate = 5e-4
buffer_size = 50000
learning_starts = 1000
batch_size = 64
gamma = 0.99
exploration_fraction = 0.1
exploration_final_eps = 0.02
target_update_interval = 250
train_freq = 4
total_timesteps = 300000
eval_episodes = 30
success_threshold = 7

torch.manual_seed(SEED)

# Create log and model dirs
log_dir = "logs/dqn"
model_dir = "models/dqn_angaza"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
#env = AngazaEnv()
env = AngazaEnv(render_mode="human")
env = Monitor(env, log_dir)

# Reward shaping stub 
def shape_reward(reward, done, info):
    # Example: small bonus for success
    if done and reward >= success_threshold:
        reward += 1.0
    return reward

# Define model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_final_eps,
    target_update_interval=target_update_interval,
    train_freq=train_freq,
    verbose=1,
    tensorboard_log=log_dir,
    seed=SEED
)

# Train
model.learn(total_timesteps=total_timesteps, log_interval=10)

# Save model
model_path = (f"{model_dir}/dqn_angaza_model")
model.save(model_path)

# Evaluate
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
episode_rewards = []
reward_threshold = 7
successes = 0
for _ in range(eval_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    episode_rewards.append(total_reward)
    if total_reward >= reward_threshold:
        successes += 1

mean_reward = sum(episode_rewards) / eval_episodes
std_reward = torch.tensor(episode_rewards).std().item()
success_rate = (successes / eval_episodes) * 100

print(f"Evaluation -> Avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Success rate: {success_rate:.2f}%")

print(f"Evaluation -> Avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save results to CSV
csv_path = os.path.join(log_dir, "dqn_logs.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "timesteps", "mean_reward", "std_reward", "learning_rate", "gamma",
        "buffer_size", "batch_size", "exploration_final_eps", "train_freq"
    ])
    writer.writerow([
        total_timesteps, mean_reward, std_reward, learning_rate, gamma,
        buffer_size, batch_size, exploration_final_eps, train_freq
    ])
print(f"Experiment logged to {csv_path}")