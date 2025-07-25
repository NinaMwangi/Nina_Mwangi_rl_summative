from environment.custom_env import AngazaEnv

def run_random_rollout(episodes=3, max_steps=50):
    env = AngazaEnv()
    # Basic sanity checks
    assert env.action_space.n == 6
    assert env.observation_space.shape == (2,)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        print(f"\n=== Episode {ep} ===")
        env.render()

        while not done and step < max_steps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            print(f"Step {step} | Action: {action} | Obs: {obs} | Reward: {reward} | Done: {done}")
            env.render()

        print(f"Episode {ep} finished. Total reward: {total_reward}")

if __name__ == "__main__":
    run_random_rollout()
