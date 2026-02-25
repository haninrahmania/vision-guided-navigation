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

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def main():
    pygame.init()

    # UI font for indicators
    font = pygame.font.SysFont(None, 22)

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

    nodes_expanded = 0
    last_plan_nodes = 0

    # Behavior filtering: require stability before replanning
    STABILITY_FRAMES = 5
    REPLAN_COOLDOWN_FRAMES = 6        # minimum time between replans
    MIN_GOAL_CELL_DELTA = 2           # ignore tiny goal changes
    MIN_PATH_PROGRESS_STEPS = 6       # commit to current path for a bit
    MAX_REPLANS_PER_SECOND = 3

    last_goal = goal
    stable_goal = None
    stable_count = 0
    replan_cooldown = 0
    replans_in_window = 0
    window_start_ms = pygame.time.get_ticks()
    last_goal_change_ms = pygame.time.get_ticks()

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
                    # Reset stability state
                    stable_goal = None
                    stable_count = 0
                    # Reset rate limiting
                    replans_in_window = 0
                    window_start_ms = pygame.time.get_ticks()
                elif event.key == pygame.K_c:
                    # clear tracker selection (re-pick color)
                    tracker.lower_hsv = None
                    tracker.upper_hsv = None
                    tracker.initialized = False
                    tracker.trail.clear()
                    if hasattr(tracker, 'miss_count'):
                        tracker.miss_count = 0
                    replan_cooldown = 0
                    # Reset stability state
                    stable_goal = None
                    stable_count = 0
                    # Reset rate limiting
                    replans_in_window = 0
                    window_start_ms = pygame.time.get_ticks()

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

            # # Only accept free cells
            # if grid.grid[new_goal[0]][new_goal[1]] == 0:

            #     # DEADZONE: treat small moves near last_goal as unchanged
            #     dist = abs(new_goal[0] - last_goal[0]) + abs(new_goal[1] - last_goal[1])
            #     if dist <= DEADZONE_RADIUS:
            #         # inside deadzone -> reset candidate and keep current goal
            #         goal_candidate = None
            #         candidate_count = 0
            #     else:
            #         # New candidate observed
            #         if goal_candidate != new_goal:
            #             goal_candidate = new_goal
            #             candidate_count = 1
            #         else:
            #             candidate_count += 1

            #         # If candidate persisted long enough and cooldown passed, accept
            #         if candidate_count >= STABILITY_FRAMES and replan_cooldown <= 0:
            #             start_now = robot.grid_pos()

            #             planner.reset(start_now, new_goal)
            #             robot.clear_path()
            #             nodes_expanded = 0

            #             last_goal = new_goal
            #             goal = new_goal

            #             replan_cooldown = 2
            #             goal_candidate = None
            #             candidate_count = 0
            now_ms = pygame.time.get_ticks()

            # rolling 1-second budget window
            if now_ms - window_start_ms >= 1000:
                window_start_ms = now_ms
                replans_in_window = 0

            if grid.grid[new_goal[0]][new_goal[1]] == 0:
                # goal stability (temporal smoothing)
                if stable_goal == new_goal:
                    stable_count += 1
                else:
                    stable_goal = new_goal
                    stable_count = 0
                    last_goal_change_ms = now_ms

                # ignore tiny goal motion (deadzone)
                goal_moved_enough = manhattan(new_goal, last_goal) >= MIN_GOAL_CELL_DELTA

                # path commitment (hysteresis): don't abandon path instantly
                # BUT: if goal moved significantly, prioritize responsiveness over commitment
                progressed_enough = (robot.step_index >= MIN_PATH_PROGRESS_STEPS) or (not robot.path)

                # also allow replanning if planner already failed (no path) and goal changed
                planner_failed = (planner.finished and planner.path is None)

                can_replan = (
                    stable_count >= STABILITY_FRAMES
                    and goal_moved_enough
                    and replan_cooldown == 0
                    and replans_in_window < MAX_REPLANS_PER_SECOND
                    and (progressed_enough or goal_moved_enough)  # either condition works
                )

                # If planner failed, relax the "progress" requirement so you can recover
                if planner_failed and stable_count >= STABILITY_FRAMES and goal_moved_enough:
                    can_replan = can_replan or (
                        replan_cooldown == 0 and replans_in_window < MAX_REPLANS_PER_SECOND
                    )

                # Debug: show which conditions are blocking replan
                if stable_count >= STABILITY_FRAMES and goal_moved_enough and not can_replan:
                    blockers = []
                    if replan_cooldown > 0:
                        blockers.append(f"cooldown({replan_cooldown})")
                    if replans_in_window >= MAX_REPLANS_PER_SECOND:
                        blockers.append("rate_limit")
                    print(f"[REPLAN BLOCKED] {', '.join(blockers)}")

                if can_replan:
                    start_now = robot.grid_pos()

                    planner.reset(start_now, new_goal)
                    robot.clear_path()

                    last_goal = new_goal
                    goal = new_goal

                    replan_cooldown = REPLAN_COOLDOWN_FRAMES
                    replans_in_window += 1

        else:
            # Target lost - reset stability state
            stable_goal = None
            stable_count = 0

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
            f"Replans/s: {replans_in_window}/{MAX_REPLANS_PER_SECOND}",
        ]
        y = 8
        for line in lines:
            # Yellow with slight shadow for readability against any background
            surf = font.render(line, True, (255, 220, 0))
            screen.blit(surf, (8, y))
            y += 18

        # Stable goal visual indicator & stability counter
        if stable_goal is not None:
            sr, sc = stable_goal
            stable_rect = pygame.Rect(
                sc * CELL_SIZE,
                sr * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, MAGENTA, stable_rect, 3)

            # draw stability text near top-left of stable cell
            text = f"{stable_count}/{STABILITY_FRAMES}"
            surf = font.render(text, True, (0, 0, 0))  # Black for contrast on magenta
            screen.blit(surf, (sc * CELL_SIZE + 4, sr * CELL_SIZE + 4))

        # show message when stable goal pending
        if stable_goal is not None and stable_count < STABILITY_FRAMES:
            info = "Stable goal pending"
            info_surf = font.render(info, True, (255, 165, 0))  # Orange for visibility
            screen.blit(info_surf, (8, 100))

        # show message when target lost
        if target is None:
            info = "TARGET LOST"
            info_surf = font.render(info, True, (255, 0, 0))
            screen.blit(info_surf, (8, 100))

        pygame.display.flip()

    tracker.release()
    pygame.quit()


if __name__ == "__main__":
    main()