#import gym
#from gym import spaces
import numpy as np
from gymnasium import Env
from gymnasium import spaces

#class AngazaEnv(gym.Env):
class AngazaEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 5, seed: int = 42, render_mode=None):
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

        self.render_mode = render_mode
        self.max_steps = self.grid_size * 4
        self.current_step = 0


    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Randomize agent's position (excluding threat and safe zone)
        while True:
            x = self.rng.integers(0, self.grid_size)
            y = self.rng.integers(0, self.grid_size)
            if (x, y) != self.threat_pos and (x, y) != self.safe_zone:
                break


        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.current_step = 0
        info = {}

        return self.agent_pos, info

    def step(self, action: int):
        self.current_step += 1
        x, y = self.agent_pos

        if action == 0 and y > 0:                      # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:   # Down
            y += 1
        elif action == 2 and x > 0:                    # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:   # Right
            x += 1
        '''elif action == 4:
            pass  # Hide 
        elif action == 5:
            pass  # Send alert '''
        if action in [1, 1, 2, 3]:
            self.agent_pos = np.array([x, y], dtype=np.int32)

     # Threat Moves randomly (1 step in any direction)
        tx, ty = self.threat_pos
        direction = self.rng.integers(0, 4)  # 0=up,1=down,2=left,3=right
        if direction == 0 and ty > 0:
            ty -= 1
        elif direction == 1 and ty < self.grid_size - 1:
            ty += 1
        elif direction == 2 and tx > 0:
            tx -= 1
        elif direction == 3 and tx < self.grid_size - 1:
            tx += 1
        self.threat_pos = (tx, ty)


        reward = -0.1
        terminated = False
        truncated = False
        info = {}

        distance_to_threat = np.linalg.norm(self.agent_pos - np.array(self.threat_pos), ord=1)

        if action == 4:  # Hide
            if distance_to_threat <= 1:
                reward = 5  # Reward for hiding near danger
            else:
                reward = -1  # Penalty for unnecessary hiding

        elif action == 5:  # Send Alert
            if distance_to_threat <= 1:
                reward = 7  # Alerting nearby threat = success
                terminated = True
            else:
                reward = -2  # False alert = penalty

        #terminal conditions
        if tuple(self.agent_pos) == self.threat_pos:
            reward = -10
            terminated = True
        elif tuple(self.agent_pos) == self.safe_zone:
            reward = 10
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        obs = np.array(self.agent_pos, dtype=np.int32)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        grid[self.threat_pos] = "T"
        grid[self.safe_zone] = "S"
        grid[tuple(self.agent_pos)] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()