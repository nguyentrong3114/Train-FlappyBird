import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self, render_enabled=True):
        self.render_enabled = render_enabled
        if render_enabled:
            pygame.init()
            self.width, self.height = 400, 600
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird - DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
        else:
            self.width, self.height = 400, 600

        self.gravity = 1
        self.jump_strength = -10
        self.pipe_width = 60
        self.pipe_gap = 150
        self.reset()

    def reset(self):
        self.bird_y = self.height // 2
        self.bird_vel = 0
        self.bird_radius = 15
        self.pipe_x = self.width
        self.pipe_height = random.randint(100, 400)
        self.score = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        if action == 1:
            self.bird_vel = self.jump_strength

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        self.pipe_x -= 5

        reward = 0.1
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_height = random.randint(100, 400)
            self.score += 1
            reward = 1.0

        self.done = False
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.height:
            self.done = True
            reward = -1

        if self.pipe_x < 50 + self.bird_radius < self.pipe_x + self.pipe_width:
            if self.bird_y - self.bird_radius < self.pipe_height or self.bird_y + self.bird_radius > self.pipe_height + self.pipe_gap:
                self.done = True
                reward = -1

        return self._get_state(), reward, self.done

    def render(self):
        if not self.render_enabled:
            return
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, 0, self.pipe_width, self.pipe_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.pipe_x, self.pipe_height + self.pipe_gap, self.pipe_width, self.height))
        pygame.draw.circle(self.screen, (255, 255, 0), (50, int(self.bird_y)), self.bird_radius)
        score_surface = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def _get_state(self):
        pipe_center_y = self.pipe_height + self.pipe_gap / 2
        vertical_distance = self.bird_y - pipe_center_y
        horizontal_distance = self.pipe_x - 50  # 50 là x cố định của chim
        return np.array([self.bird_y, self.bird_vel, horizontal_distance, vertical_distance], dtype=np.float32)
