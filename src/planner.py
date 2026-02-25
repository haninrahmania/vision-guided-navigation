import heapq
import math

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.open_set = []
        self.closed_set = set()
        self.g_score = {}
        self.came_from = {}
        self.finished = False
        self.path = None

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, node):
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),      # cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)    # diagonal
        ]

        neighbors = []

        for dr, dc in directions:
            nr = node[0] + dr
            nc = node[1] + dc
            neighbor = (nr, nc)

            if not self.grid.in_bounds(neighbor):
                continue

            if self.grid.is_obstacle(neighbor):
                continue

            # Prevent corner cutting
            if dr != 0 and dc != 0:
                if self.grid.is_obstacle((node[0] + dr, node[1])) or \
                self.grid.is_obstacle((node[0], node[1] + dc)):
                    continue

            neighbors.append(neighbor)

        return neighbors


    def initialize(self, start, goal):
        self.start = start
        self.goal = goal
        self.open_set = []
        heapq.heappush(self.open_set, (0, start))

        self.came_from = {}
        self.g_score = {start: 0}
        self.closed_set = set()
        self.finished = False
        self.path = None

    def step(self):
        if self.finished:
            return
        
        if not self.open_set:
            self.finished = True
            self.path = None
            return

        _, current = heapq.heappop(self.open_set)

        if current in self.closed_set:
            return

        self.closed_set.add(current)

        if current == self.goal:
            self.finished = True
            self.path = self.reconstruct_path(current)
            return

        for neighbor in self.get_neighbors(current):
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])

            if dx == 1 and dy == 1:
                move_cost = math.sqrt(2)
            else:
                move_cost = 1

            tentative_g = self.g_score[current] + move_cost


            if neighbor in self.closed_set:
                continue

            if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                self.came_from[neighbor] = current
                self.g_score[neighbor] = tentative_g
                f_score = tentative_g + self.heuristic(neighbor, self.goal)
                heapq.heappush(self.open_set, (f_score, neighbor))

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def reset(self, start, goal):
        self.start = start
        self.goal = goal

        self.open_set = []
        self.closed_set = set()

        self.g_score = {start: 0}
        self.came_from = {}

        self.finished = False
        self.path = None

        # Initial f-score = heuristic(start, goal)
        initial_f = self.heuristic(start, goal)
        heapq.heappush(self.open_set, (initial_f, start))
