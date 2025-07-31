import os
import sys
import csv
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AngazaEnv

#Global PPO Hyperparameters
PPO_PARAMS = {
    "learning_rate": lambda f: 0.00025 * f,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.015,
    "total_timesteps": 75000,
    "env_name": "AngazaEnv",
}

LOG_PATH = "logs/pg/ppo_logs.csv"
MODEL_SAVE_PATH = "models/pg/angaza_ppo"

#Logging Function
def log_experiment_to_csv(log_path, config, results):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fieldnames = list(config.keys()) + list(results.keys()) + ["timestamp"]
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        log_entry = {**config, **results, "timestamp": datetime.now().isoformat()}
        writer.writerow(log_entry)

def train_ppo_model(params=PPO_PARAMS, save_path=MODEL_SAVE_PATH, log_path=LOG_PATH):
    # Create vectorized environment
    def make_env():
        env = AngazaEnv()
        return Monitor(env, filename="logs/pg/angaza_ppo_monitor.csv")

    env = DummyVecEnv([make_env])

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        verbose=1,
        tensorboard_log="./logs/pg"
    )

    # Train
    model.learn(total_timesteps=params['total_timesteps'])

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"PPO model saved to: {save_path}")

    # Evaluate
    #eval_env = Monitor(AngazaEnv())
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"PPO Evaluation: Mean reward = {mean_reward:.2f}, Std = {std_reward:.2f}")

    # Log to CSV
    results = {
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "n_eval_episodes": 10,
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    log_experiment_to_csv(log_path, params, results)
    print(f"Logged to: {log_path}")

if __name__ == "__main__":
    train_ppo_model()