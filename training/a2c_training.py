import os
import sys
import csv
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.custom_env import AngazaEnv
from stable_baselines3.common.monitor import Monitor

#Global Hyperparameters
A2C_PARAMS = {
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "n_steps": 5,
    "ent_coef": 0.005,
    "vf_coef": 0.04,
    "max_grad_norm": 1.0,
    "total_timesteps": 75000,
    "seed": 123,
    "env_name": "AngazaEnv"
}

LOG_PATH = "logs/pg/a2c_logs.csv"
MODEL_SAVE_PATH = "models/pg/angaza_a2c"

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

def train_a2c_model(params=A2C_PARAMS, save_path=MODEL_SAVE_PATH, log_path=LOG_PATH):
    # Wrap custom env for SB3
    def make_env():
        env = AngazaEnv()
        return Monitor(env, filename="logs/pg/angaza_a2c_monitor.csv")

    env = DummyVecEnv([make_env])

    # Define the A2C model
    model = A2C(
        "MlpPolicy",
        env,
        seed=params["seed"],
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        verbose=1,
        tensorboard_log="./logs/pg"
    )

    # Train the model
    model.learn(total_timesteps=params["total_timesteps"])

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"A2C model saved to: {save_path}")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"A2C Evaluation: Mean reward = {mean_reward:.2f}, Std = {std_reward:.2f}")

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
    train_a2c_model()
