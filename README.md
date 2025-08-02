# Angaza 

## Project Overview
This project introduces a custom reinforcement learning environment named AngazaEnv, simulating a safety navigation task. The goal is for an agent to navigate a 5x5 grid while avoiding a moving threat and reaching a designated safe zone. I implemented and evaluated multiple RL algorithms—DQN, PPO, A2C, and REINFORCE, using stable-baselines3 and custom evaluation scripts. Through careful tuning, visualization, and comparison, this project demonstrates the impact of architecture and hyperparameters on agent performance and convergence.

## Environment Description
- 2.1 Agent(s)
The agent represents a civilian trying to navigate to a safe zone while avoiding a threat. It can move in four directions. It cannot pass through the threat or exit the grid. The agent starts at a random location for each episode.
- 2.2 Action Space
Type: Discrete, Actions: Move up, move down, move left and move right.

- 2.3 State Space
  - Type: Box (Flat Array)
  - Observations consist of:
  - Agent position (x, y)
  - Threat position (x, y)
  - Safe zone position (x, y)
  - Encoded as a flattened NumPy array: [ax, ay, tx, ty, sx, sy]
- 2.4 Reward Structure
  - +10 for reaching the safe zone
  - -10 for collision with a threat
  - -0.1 per timestep to encourage faster convergence
  - Episode ends on success or collision

- 2.5 Environment Visualization
I implemented a visualizer using Pygame, with animated movement and custom icons:
  - 🧍 Agent
  - ☠️ Threat
  - 🛡️ Safe Zone
  - 👣 Trail (recent path of the agent)
  - Flashing effect on collision

