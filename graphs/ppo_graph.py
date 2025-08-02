import pandas as pd
import matplotlib.pyplot as plt

# Load PPO results CSV
path = "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/ppo_logs.csv" 
df = pd.read_csv(path)

# Sort by timesteps (optional but helpful for consistent plotting)
df = df.sort_values(by='total_timesteps')

# Plot mean reward with standard deviation band
plt.figure(figsize=(10, 6))
plt.plot(df['total_timesteps'], df['mean_reward'], marker='o', label='PPO Mean Reward')
plt.fill_between(df['total_timesteps'],
                 df['mean_reward'] - df['std_reward'],
                 df['mean_reward'] + df['std_reward'],
                 alpha=0.2, label='±1 Std Dev')

plt.xlabel('Total Timesteps')
plt.ylabel('Mean Reward')
plt.title('PPO: Average Reward vs Total Timesteps')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show plot
plt.savefig("ppo_avg_reward_plot.png")
plt.show()
