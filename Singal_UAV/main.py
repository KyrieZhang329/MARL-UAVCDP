import time
import numpy as np
from env.env import UAVEnv

def main():
    env = UAVEnv(render_mode="human")
    obs,info = env.reset()

    total_reward = 0

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step%10 == 0:
            print(f"Step: {step}, Pos: {obs[:2]}, Coverage: {obs[2] * 100:.1f}%, Reward: {reward:.2f}")

        env.render()

        if terminated or truncated:
            print("Ends")
            print(f"Total coverage: {truncated * 100:.1f}% and reward: {total_reward:.2f}")
            obs, info = env.reset()
            continue

    env.close()


if __name__ == '__main__':
    main()
