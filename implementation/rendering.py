import pygame
import sys
import numpy as np
import time
import os

CELL_SIZE = 100
GRID_COLOR = (200, 200, 200)
AGENT_COLOR = (0, 128, 255)
THREAT_COLOR = (255, 80, 80)
SAFE_COLOR = (100, 255, 100)
BG_COLOR = (245, 245, 245)
FLASH_DURATION = 0.5  
FLASH_INTERVAL = 0.1 
TRAIL_LENGTH = 10

class GridRenderer:
    def __init__(self, grid_size=5, title="Angaza Simulation"):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = CELL_SIZE
        self.window_size = grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.last_agent_pos = None
        self.agent_trail = []

        # Load images from assets/
        assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
        self.footsteps_img = self._load_image(os.path.join(assets_path, 'footprints.png'))
        self.threat_img = self._load_image(os.path.join(assets_path, 'threat.png'))
        self.safe_zone_img = self._load_image(os.path.join(assets_path, 'shield.png'))
        self.agent_img = self._load_image(os.path.join(assets_path, 'agent.png'))

    def _load_image(self, path):
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.scale(img, (self.cell_size, self.cell_size))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return pygame.Surface((self.cell_size, self.cell_size))  # Placeholder if image fails

    def render(self, agent_pos, threat_pos, safe_zone, fps=5, collided=False):
        self._handle_events()
        self.screen.fill(BG_COLOR)

        # Draw grid
        for x in range(0, self.window_size, self.cell_size):
            for y in range(0, self.window_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        # Update trail
        if self.last_agent_pos is not None and not np.array_equal(self.last_agent_pos, agent_pos):
            self.agent_trail.append(self.last_agent_pos.copy() if isinstance(self.last_agent_pos, np.ndarray) else self.last_agent_pos)
            if len(self.agent_trail) > TRAIL_LENGTH:
                self.agent_trail.pop(0)
        self.last_agent_pos = agent_pos.copy() if isinstance(agent_pos, np.ndarray) else agent_pos

        # Draw trail using 👣 image
        for pos in self.agent_trail:
            self._blit_image(pos, self.footsteps_img)

        # Draw threat and safe zone
        self._draw_cell(threat_pos, THREAT_COLOR)
        self._blit_image(threat_pos, self.threat_img)

        self._draw_cell(safe_zone, SAFE_COLOR)
        self._blit_image(safe_zone, self.safe_zone_img)

        # Draw agent
        if collided:
            self._flashing_effect(agent_pos)
        else:
            self._draw_cell(agent_pos, AGENT_COLOR)
            self._blit_image(agent_pos, self.agent_img)

        pygame.display.flip()
        self.clock.tick(fps)

    def _draw_cell(self, pos, color):
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)

    def _blit_image(self, pos, img):
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        self.screen.blit(img, rect)

    def _flashing_effect(self, agent_pos):
        start_time = time.time()
        while time.time() - start_time < FLASH_DURATION:
            self._draw_cell(agent_pos, (255, 255, 255))
            pygame.display.flip()
            time.sleep(FLASH_INTERVAL)

            self._draw_cell(agent_pos, AGENT_COLOR)
            self._blit_image(agent_pos, self.agent_img)
            pygame.display.flip()
            time.sleep(FLASH_INTERVAL)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
