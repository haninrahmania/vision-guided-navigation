import pygame
import random
import copy
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GREY = (80, 80, 80)

class Grid:
    def __init__(self, rows, cols, cell_size, obstacle_ratio=0.2):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.generate_obstacles(obstacle_ratio)

    def generate_obstacles(self, obstacle_ratio):
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < obstacle_ratio:
                    self.grid[r][c] = 1  # 1 means obstacle

    def is_obstacle(self, node):
        r, c = node
        return self.grid[r][c] in (1, 2)  # Treat inflated obstacles as impassable

    def in_bounds(self, node):
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def inflate_obstacles(self, radius):
        inflated = copy.deepcopy(self.grid)

        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] == 1:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny = y + dy
                            nx = x + dx

                            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                                # Circular check
                                if math.sqrt(dx**2 + dy**2) <= radius:
                                    if inflated[ny][nx] == 0:
                                        inflated[ny][nx] = 2  # 2 means inflated obstacle

        self.grid = inflated

    def draw(self, screen):
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                cell_value = self.grid[r][c]

                if cell_value == 1:
                    pygame.draw.rect(screen, BLACK, rect)

                elif cell_value == 2:
                    pygame.draw.rect(screen, DARK_GREY, rect)

                else:
                    pygame.draw.rect(screen, WHITE, rect)

                pygame.draw.rect(screen, GRAY, rect, 1)
