import gym
from gym import spaces
import numpy as np

class AngazaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 5, seed: int = 42):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(6)  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Hide, 5=Send Alert
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2,),
            dtype=np.int32
        )

        self.rng = np.random.default_rng(seed)
        self.agent_pos = None
        self.threat_pos = (2, 2)
        self.safe_zone = (4, 4)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos

    def step(self, action: int):
        x, y = self.agent_pos

        if action == 0 and y > 0:                      # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:   # Down
            y += 1
        elif action == 2 and x > 0:                    # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:   # Right
            x += 1
        elif action == 4:
            pass  # Hide (no move) — reward shaping later
        elif action == 5:
            pass  # Send alert — reward shaping later

        self.agent_pos = np.array([x, y], dtype=np.int32)

        reward = -1
        done = False

        if tuple(self.agent_pos) == self.threat_pos:
            reward = -10
            done = True
        elif tuple(self.agent_pos) == self.safe_zone:
            reward = 10
            done = True

        info = {}
        return self.agent_pos, reward, done, info

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.threat_pos] = "T"
        grid[self.safe_zone] = "S"
        grid[tuple(self.agent_pos)] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
