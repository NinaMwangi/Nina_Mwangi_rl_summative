import pandas as pd
import matplotlib.pyplot as plt

# Paths to your CSV files
paths = {
    "PPO": "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/ppo_logs.csv",
    "A2C": "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/a2c_logs.csv",
    "REINFORCE": "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/reinforce_logs.csv"
}

plt.figure(figsize=(10, 6))

# Loop through each algorithm and plot its rewards
for label, path in paths.items():
    df = pd.read_csv(path)
    if 'episode' not in df.columns or 'reward' not in df.columns:
        raise ValueError(f"{path} must have 'episode' and 'reward' columns.")
    
    # Plot reward per episode (you could also do a rolling average for smoothness)
    plt.plot(df['episode'], df['reward'], label=label)

plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_methods_cumulative_rewards.png")
plt.show()
