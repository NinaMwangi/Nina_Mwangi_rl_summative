import pandas as pd
import matplotlib.pyplot as plt

# Load A2C results
path = "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/a2c_logs.csv"
df = pd.read_csv(path, on_bad_lines='skip')

# Keep only needed columns and drop any rows with missing data
df = df[['total_timesteps', 'mean_reward', 'std_reward']].dropna()

# Sort for clean plotting
df = df.sort_values(by='total_timesteps')

# Plot mean reward with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(df['total_timesteps'], df['mean_reward'], marker='o', label='A2C Mean Reward', color='orange')
plt.fill_between(df['total_timesteps'],
                 df['mean_reward'] - df['std_reward'],
                 df['mean_reward'] + df['std_reward'],
                 alpha=0.2, color='orange', label='±1 Std Dev')

plt.xlabel('Total Timesteps')
plt.ylabel('Mean Reward')
plt.title('A2C: Average Reward vs Total Timesteps')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show plot
plt.savefig("a2c_avg_reward_plot.png")
plt.show()
