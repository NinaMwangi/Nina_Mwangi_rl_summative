import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set your CSV path
path = "/Users/nina/Nina_Mwangi_rl_summative/logs/pg/reinforce_logs.csv"

# Load the data
df = pd.read_csv(path)

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='total_episodes', y='final_mean_reward', marker='o')

plt.title("REINFORCE: Final Mean Reward vs Total Episodes")
plt.xlabel("Total Episodes")
plt.ylabel("Final Mean Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("reinforce_reward_plot.png")
plt.show()