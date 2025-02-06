import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.image import load
from pygame.surfarray import pixels_alpha
from pygame.transform import rotate
import numpy as np
from itertools import cycle
from random import randint
from pygame import Rect

pygame.init()
pygame.display.init()

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        # First 5 values represent: bird_y, velocity_y, horizontal_distance, upper_pipe_y, lower_pipe_y
        self.observation_space = spaces.Box(
            low=np.array([0, -8, 0, -512, 0]), 
            high=np.array([512, 10, 288, 0, 512]), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = pygame.display.set_mode()
        self.clock = None

        self.screen_width = 258
        self.screen_height = 512
        self.base_image = load('assets/base.png').convert_alpha()
        self.background_image = load('assets/background-black.png').convert()
        self.pipe_images = [rotate(load('assets/pipe-green.png').convert_alpha(), 180),
                            load('assets/pipe-green.png').convert_alpha()]
        self.bird_images = [load('assets/redbird-upflap.png').convert_alpha(),
                    load('assets/redbird-midflap.png').convert_alpha(),
                    load('assets/redbird-downflap.png').convert_alpha()]
        self.bird_hitmask = [pixels_alpha(image).astype(bool) for image in self.bird_images]
        self.pipe_hitmask = [pixels_alpha(image).astype(bool) for image in self.pipe_images]
        self.pipe_gap_size = 100
        self.pipe_velocity_x = -4
        self.min_velocity_y = -8
        self.max_velocity_y = 10
        self.downward_speed = 1
        self.upward_speed = -9

        self.bird_index_generator = cycle([0, 1, 2, 1])
        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()

    def generate_pipe(self):
        x = self.screen_width + 10
        gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        return {"x_upper": x, "y_upper": gap_y - self.pipe_height, "x_lower": x, "y_lower": gap_y + self.pipe_gap_size}

    def is_collided(self):
        # Check if the bird touch ground
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height))
            # Check if the bird's bounding box overlaps to the bounding box of any pipe
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1:min_x1 + cropped_bbox.width,
                       min_y1:min_y1 + cropped_bbox.height] * self.pipe_hitmask[i][min_x2:min_x2 + cropped_bbox.width,
                                                              min_y2:min_y2 + cropped_bbox.height]):
                    return True
        return False
    
    def _get_obs(self):
        # Get next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x_lower"] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        if next_pipe:
            horizontal_distance = next_pipe["x_lower"] + self.pipe_width - self.bird_x
            upper_pipe_y = next_pipe["y_upper"]
            lower_pipe_y = next_pipe["y_lower"]
        else:
            horizontal_distance = self.screen_width
            upper_pipe_y = 0
            lower_pipe_y = 0

        return np.array([
            self.bird_y,
            self.current_velocity_y,
            horizontal_distance,
            upper_pipe_y,
            lower_pipe_y,
        ], dtype=np.float32)

    def _get_info(self):
        return dict({
            'score': self.score
        })

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.iter = self.bird_index = self.score = 0
        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()
        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.pipes = pipes

        self.current_velocity_y = 0
        self.is_flapped = False
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        reward = 0.1
        terminated = False
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True

        # Update score
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break

        # Update index and iteration
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # Update bird's position
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(self.current_velocity_y, self.bird_y - self.current_velocity_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x

        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]
        if self.is_collided():
            terminated = True
            reward = -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.blit(self.background_image, (0, 0))
        canvas.blit(self.base_image, (self.base_x, self.base_y))
        canvas.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            canvas.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            canvas.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

