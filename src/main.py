import pygame

from grid import Grid
from planner import AStarPlanner
from robot import Robot

from vision.tracker import HSVKalmanTracker

ROWS = 20
COLS = 20
CELL_SIZE = 30

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 200, 255)
MAGENTA = (255, 0, 255)

def camera_to_grid(px, py, frame_w, frame_h):
    col = int(px / frame_w * COLS)
    row = int(py / frame_h * ROWS)

    row = max(0, min(ROWS - 1, row))
    col = max(0, min(COLS - 1, col))

    return (row, col)

def main():
    pygame.init()

    # UI font for indicators
    font = pygame.font.SysFont(None, 20)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Vision-Guided Navigation")

    clock = pygame.time.Clock()

    # ----------------------------------
    # WORLD SETUP
    # ----------------------------------
    grid = Grid(ROWS, COLS, CELL_SIZE, obstacle_ratio=0.1)
    grid.inflate_obstacles(radius=1)

    start = (0, 0)
    goal = (ROWS - 1, COLS - 1)

    grid.grid[start[0]][start[1]] = 0
    grid.grid[goal[0]][goal[1]] = 0

    planner = AStarPlanner(grid)
    planner.initialize(start, goal)

    robot = Robot(start, CELL_SIZE)

    # ----------------------------------
    # VISION TRACKER
    # ----------------------------------
    tracker = HSVKalmanTracker()

    last_goal = goal
    replan_cooldown = 0

    nodes_expanded = 0
    global last_plan_nodes
    last_plan_nodes = 0

    # Behavior filtering: require stability before replanning
    goal_candidate = None
    candidate_count = 0
    STABILITY_FRAMES = 5
    DEADZONE_RADIUS = 1  # cells

    running = True

    # ==================================
    # MAIN LOOP
    # ==================================
    while running:

        clock.tick(30)

        # ------------------------------
        # EVENTS
        # ------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # reset trails + replan to current goal
                    robot.clear_path()
                    planner.reset(robot.grid_pos(), goal)
                    nodes_expanded = 0
                    last_goal = goal
                    goal_candidate = None
                    candidate_count = 0
                elif event.key == pygame.K_c:
                    # clear tracker selection (re-pick color)
                    tracker.lower_hsv = None
                    tracker.upper_hsv = None
                    tracker.initialized = False
                    tracker.trail.clear()
                    replan_cooldown = 0
                    goal_candidate = None
                    candidate_count = 0

        # ------------------------------
        # VISION UPDATE (OpenCV) + behavior filtering
        # ------------------------------
        target = tracker.update()

        if (
            target is not None
            and tracker.frame_w is not None
            and tracker.frame_h is not None
        ):
            tx, ty = target

            new_goal = camera_to_grid(
                tx, ty,
                tracker.frame_w,
                tracker.frame_h
            )

            # Only accept free cells
            if grid.grid[new_goal[0]][new_goal[1]] == 0:

                # DEADZONE: treat small moves near last_goal as unchanged
                dist = abs(new_goal[0] - last_goal[0]) + abs(new_goal[1] - last_goal[1])
                if dist <= DEADZONE_RADIUS:
                    # inside deadzone -> reset candidate and keep current goal
                    goal_candidate = None
                    candidate_count = 0
                else:
                    # New candidate observed
                    if goal_candidate != new_goal:
                        goal_candidate = new_goal
                        candidate_count = 1
                    else:
                        candidate_count += 1

                    # If candidate persisted long enough and cooldown passed, accept
                    if candidate_count >= STABILITY_FRAMES and replan_cooldown <= 0:
                        start_now = robot.grid_pos()

                        planner.reset(start_now, new_goal)
                        robot.clear_path()
                        nodes_expanded = 0

                        last_goal = new_goal
                        goal = new_goal

                        replan_cooldown = 2
                        goal_candidate = None
                        candidate_count = 0

        if replan_cooldown > 0:
            replan_cooldown -= 1

        # ------------------------------
        # PLANNER STEP
        # ------------------------------
        if not planner.finished:
            planner.step()
            nodes_expanded = len(planner.closed_set)

        elif planner.finished:
            last_plan_nodes = nodes_expanded

            if planner.path and not robot.path:
                robot.set_path(planner.path)

        # ------------------------------
        # ROBOT UPDATE
        # ------------------------------
        robot.update()

        # ------------------------------
        # DRAW
        # ------------------------------
        screen.fill(WHITE)

        grid.draw(screen)

        # closed set
        for node in planner.closed_set:
            r, c = node
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, ORANGE, rect)

        # open set
        for _, node in planner.open_set:
            r, c = node
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, YELLOW, rect)

        # draw current goal (VERY IMPORTANT VISUALLY)
        gr, gc = goal
        goal_rect = pygame.Rect(
            gc * CELL_SIZE,
            gr * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(screen, CYAN, goal_rect, 3)
        gr, gc = goal
        cx = gc * CELL_SIZE + CELL_SIZE // 2
        cy = gr * CELL_SIZE + CELL_SIZE // 2
        size = CELL_SIZE // 2
        pygame.draw.line(screen, (0, 200, 255), (cx - size, cy), (cx + size, cy), 3)
        pygame.draw.line(screen, (0, 200, 255), (cx, cy - size), (cx, cy + size), 3)

        if planner.path:
            robot.draw_path(screen)

        robot.draw_trail(screen)
        robot.draw(screen)

        lines = [
            f"Goal: {goal}",
            f"Planner: {'RUNNING' if not planner.finished else ('FOUND' if planner.path else 'NO PATH')}",
            f"Closed-set (expanded): {len(planner.closed_set)}",
            f"Last plan expanded: {last_plan_nodes}",
        ]
        y = 8
        for line in lines:
            surf = font.render(line, True, (0, 0, 0))
            screen.blit(surf, (8, y))
            y += 18

        # Candidate visual indicator & stability counter
        if goal_candidate is not None:
            cr, cc = goal_candidate
            cand_rect = pygame.Rect(
                cc * CELL_SIZE,
                cr * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, MAGENTA, cand_rect, 3)

            # draw stability text near top-left of candidate cell
            text = f"{candidate_count}/{STABILITY_FRAMES}"
            surf = font.render(text, True, MAGENTA)
            screen.blit(surf, (cc * CELL_SIZE + 4, cr * CELL_SIZE + 4))

        # show message when candidate pending
        if goal_candidate is not None:
            info = "Candidate goal pending"
            info_surf = font.render(info, True, (50, 50, 50))
            screen.blit(info_surf, (8, 8))

        pygame.display.flip()

    tracker.release()
    pygame.quit()


if __name__ == "__main__":
    main()