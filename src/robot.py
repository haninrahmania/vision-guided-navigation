import pygame

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Robot:
    def __init__(self, start, cell_size):
        self.position = start          # (row, col)
        self.cell_size = cell_size
        self.path = []
        self.step_index = 0

        self.trail = []
        self.trail_max = 60

    def set_path(self, path):
        self.path = path
        self.step_index = 0

    def clear_path(self):
        self.path = []
        self.step_index = 0

    def grid_pos(self):
        return self.position

    def update(self):
        if self.path and self.step_index < len(self.path):
            self.position = self.path[self.step_index]
            self.trail.append(self.position)
            if len(self.trail) > self.trail_max:
                self.trail.pop(0)
            self.step_index += 1

    def draw_trail(self, screen):
        for (r, c) in self.trail:
            rect = pygame.Rect(
                c * self.cell_size + self.cell_size // 4,
                r * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2
            )
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    def draw(self, screen):
        r, c = self.position
        x = c * self.cell_size + self.cell_size // 2
        y = r * self.cell_size + self.cell_size // 2
        radius = max(3, self.cell_size // 3)
        pygame.draw.circle(screen, BLUE, (x, y), radius)

    def draw_path(self, screen):
        if not self.path:
            return

        points = []
        for r, c in self.path:
            x = c * self.cell_size + self.cell_size // 2
            y = r * self.cell_size + self.cell_size // 2
            points.append((x, y))
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(screen, GREEN, rect, 1)

        if len(points) >= 2:
            pygame.draw.lines(screen, GREEN, False, points, 3)
        for p in points:
            pygame.draw.circle(screen, GREEN, p, max(2, self.cell_size // 6))