import pygame
import sys
import numpy as np

CELL_SIZE = 100
GRID_COLOR = (200, 200, 200)
AGENT_COLOR = (0, 128, 255)
THREAT_COLOR = (255, 0, 0)
SAFE_COLOR = (0, 255, 0)
BG_COLOR = (245, 245, 245)

class GridRenderer:
    def __init__(self, grid_size=5, title="Angaza Simulation"):
        pygame.init()
        self.grid_size = grid_size
        self.window_size = grid_size * CELL_SIZE
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def render(self, agent_pos, threat_pos, safe_zone, fps=5):
        self.screen.fill(BG_COLOR)

        # Drawing grid
        for x in range(0, self.window_size, CELL_SIZE):
            for y in range(0, self.window_size, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        # Draw threat
        self._draw_cell(threat_pos, THREAT_COLOR)

        # Draw safe zone
        self._draw_cell(safe_zone, SAFE_COLOR)

        # Draw agent
        self._draw_cell(agent_pos, AGENT_COLOR)

        pygame.display.flip()
        self.clock.tick(fps)

        # Handling quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def close(self):
        pygame.quit()
