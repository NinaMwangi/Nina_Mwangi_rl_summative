import pandas as pd
import matplotlib.pyplot as plt

# Load the DQN CSV
path = "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/DQN_experiment_results.csv"
df = pd.read_csv(path)

# Plot Avg Reward vs Total Timesteps
plt.figure(figsize=(10, 6))
plt.plot(df['total_timesteps'], df['Avg Reward'], marker='o', label='DQN Avg Reward')
plt.fill_between(df['total_timesteps'],
                 df['Avg Reward'] - df['Std Dev'],
                 df['Avg Reward'] + df['Std Dev'],
                 alpha=0.2, label='±1 Std Dev')

plt.xlabel('Total Timesteps')
plt.ylabel('Average Reward')
plt.title('DQN: Average Reward vs Total Timesteps')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("dqn_avg_reward_plot.png")
plt.show()