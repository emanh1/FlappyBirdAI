import pygame
import sys
from env.FlappyBirdEnv.envs.flappy_bird import FlappyBirdEnv
import gymnasium as gym
def play_flappy_bird():
    env = gym.make('FlappyBirdEnv/FlappyBird-v0', render_mode='human')
    observation, info = env.reset()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Flap on mouse click
                observation, reward, terminated, truncated, info = env.step(1)
                
                if terminated:
                    observation, info = env.reset()
            
        # Apply gravity when not clicking
        observation, reward, terminated, truncated, info = env.step(0)
        if terminated:
            observation, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    play_flappy_bird()
