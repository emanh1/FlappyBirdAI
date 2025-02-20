import gymnasium as gym
from stable_baselines3 import DQN
import FlappyBirdEnv

model = DQN.load("best_model/best_model.zip")

env = gym.make('FlappyBirdEnv/FlappyBird-v0', render_mode="human")


obs, info = env.reset()
total = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        obs, info = env.reset()
        print(total)
        total = 0

env.close()
