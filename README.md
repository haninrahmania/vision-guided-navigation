# Vision-Guided Navigation (Current Version)
This project integrates real-time computer vision tracking with grid-based robot navigation to demonstrate perception-driven planning in a robotics pipeline.

### The system combines:
- Visual perception (HSV tracking + Kalman filtering)
- State estimation
- Dynamic A* path planning
- Continuous replanning in response to live observations

### System Overview
The architecture follows a standard robotics loop:

Camera Input
      ↓
Object Tracking (HSV Segmentation)
      ↓
Kalman Filter State Estimation
      ↓
Camera → Grid Coordinate Mapping
      ↓
Dynamic A* Replanning
      ↓
Robot Execution (Simulator)

### The project runs using two synchronized windows:

- OpenCV window → perception + tracking visualization

- Pygame window → navigation simulator

## Feature 1 -- Perception-Driven Dynamic Replanning
The navigation system no longer relies on static goals.

Instead:
- A colored object is tracked using webcam input.
- The filtered object position becomes the robot's navigation target.
- The planner dynamically updates the goal during runtime.

### Core Integration Concepts
1. Visual Tracking
- HSV color segmentation (click-to-select calibration)
- Morphological filtering for noise reduction
- Largest-blob detection
- Kalman filtering for motion estimation

Tracking outputs:

(x, y) camera coordinates

2. Camera → Grid Mapping
Tracked positions are converted into navigation goals:

grid_col = x / frame_width  × COLS
grid_row = y / frame_height × ROWS

This allows perception data to directly drive planning.

3. Dynamic A* Replanning
Whenever the tracked goal changes:
- Planner resets from robot's current position.
- New path is generated.
- Robot adapts in real time.

This demonstrates:
- reactive planning
- perception-informed autonomy
- real-time system integration

## Feature 2 -- Behavioral Stability (Temporal Smoothing)
Directly replanning every frame causes instability due to tracking noise.

Observed behavior:
- excessive replanning
- jittery robot motion
- unstable goal switching
- Temporal Stability Filtering

To stabilize behavior, a temporal filtering strategy was added:

### Concept
A new goal must remain stable for several frames before triggering replanning.

Pseudo-behavior:

if goal is stable for N frames:
    accept goal
    replan

This introduces:
- decision-level smoothing
- reduced planner resets
- more natural motion

## Resulting Behavior

The robot now:
- continuously tracks a moving visual target
- replans paths in real time
- moves smoothly instead of reacting to noise
- demonstrates a full perception → planning pipeline

## Running the Project
python main.py

### Steps:
1. Open webcam window appears.
2. Click an object to select tracking color.
3. Move the object.
4. Robot dynamically replans toward the tracked target.

## Future Improvements
- Path hysteresis for even smoother motion
- Velocity-aware planning
- Multi-target tracking
- Costmap-based planning instead of binary grid
- Predictive goal following using Kalman velocity

## Version 0.1.1 -- Visual Debugging & Demo Polish
This update focuses on improving visual clarity, algorithm transparency, and overall presentation quality for real-time robotics demonstrations.

The goal of this version is to make system behavior immediately understandable to observers by visualizing internal state and motion history.

1. Motion Trail Visualization
Two trajectory trails were added:

### Perception Trail (OpenCV)
- Displays the recent filtered positions from the Kalman tracker.
- Visualizes tracking stability over time.
- Helps observe sensor noise vs filtered motion.

This makes it easier to see:
- smoothing effects
- estimation consistency
- object motion patterns

### Robot Trail (Pygame)
- Stores and renders recent robot positions.
- Shows actual navigation history.

This provides:
- clearer understanding of path execution
- visibility into replanning behavior
- demonstration of motion smoothness

2. Real-Time Planning HUD
A lightweight HUD overlay was added to display planner metrics during execution.

### Displayed information:
- Current navigation goal
- Planner state (RUNNING / FOUND / NO PATH)
- Nodes expanded during search
- Nodes expanded in last completed plan

Purpose:
- expose algorithm internals visually
- improve debugging and analysis
- mimic robotics research visualization tools

3. Goal Crosshair Visualization
A high-visibility crosshair now marks the current navigation target.

This directly illustrates:
Camera Tracking → Grid Mapping → Navigation Goal

### Benefits:
- immediate understanding of perception→planning link
- easier debugging of coordinate conversion
- clearer demo communication

## Version 0.1.2 -- Replanning Hysteresis & Budgeted Control Loop
- Introduced decision-level stability mechanisms to reduce replanning thrash in dynamic target tracking.
- Added deadzone thresholds, minimum path commitment, cooldown, and replanning rate budget (replans/sec) to produce smoother and more realistic autonomous behavior.
- Improved robustness when planner temporarily fails (unreachable goals).

## Version 0.2.0 -- ArUco Marker Following (Closed-Loop Control)
This version introduces a complete **perception → control** pipeline using ArUco marker detection for closed-loop robot following behavior.

### Closed-Loop ArUco Follow Behavior

**Architecture:**
```
Webcam Input
      ↓
ArUco Detection (cv2.aruco)
      ↓
6DoF Pose Estimation (solvePnP)
      ↓
FollowController (distance + heading)
      ↓
Robot Drive Commands (linear, angular)
      ↓
Robot Execution (simulation/abstraction)
```

### Key Features

1. **ArUco Marker Detection**
   - Real-time 4x4 ArUco marker detection using OpenCV
   - Robust corner refinement (subpixel accuracy)
   - Configurable marker size (default: 5cm)

2. **6DoF Pose Estimation**
   - Camera-to-marker translation vector (tvec) for distance
   - Rotation vector (rvec) for orientation
   - Fallback camera intrinsics for quick setup

3. **FollowController**
   - **Distance-based control:** Stop when close, forward when far
   - **Heading correction:** Left/right steering based on marker center offset
   - **Deadzone handling:** Prevents jitter when centered
   - **Action states:** SEARCH, FORWARD, TURN LEFT, TURN RIGHT

4. **Robot Drive Abstraction**
   - Unified drive command interface
   - Simulated robot interface for testing
   - Command translation: (linear, angular) → (left_speed, right_speed)

### Achievements
- Integrated ArUco detection with FollowController
- Distance-based stop + forward logic working
- Left/right steering using marker center offset
- Control output connected to robot drive abstraction
- Full perception → control loop operational

### Running the ArUco Follower
```bash
python -m src.main_follow
```

### Behavior
- Robot tracks marker in real-time
- Maintains target distance (~0.35m default)
- Centers marker in camera view
- Stops when marker is lost or at target distance

## Version 0.2.1 -- Smooth Visual Servo Control (P-Control + Slew Limiting)

This update adds a more realistic closed-loop controller for ArUco-guided following, improving stability and "robot-like" motion compared to discrete left/right turning.

### Continuous Steering (P-Control)

Instead of binary "turn left/right" decisions, steering is computed continuously from the marker's horizontal pixel error.

**Parameters:**
- `cx` = detected marker center in pixels
- `cx0` = image center (frame_width / 2)

**Error calculation:**
```
err_px = cx - cx0
err_n  = err_px / (frame_width / 2)  # normalized to [-1, +1]
```

**Angular command (P-controller):**
```
ang = clamp(kp * err_n, -max_ang, +max_ang)
```

This produces smooth turning proportional to how far the marker is from center.

### Speed Scheduling (Alignment + Distance)

Forward speed is automatically reduced when:
- The robot is turning sharply (misaligned)
- The robot is close to the target (small Z distance)

This prevents overshoot and creates more controlled approaches.

### Slew-Rate Limiting (Anti-Jitter)

Linear and angular commands are smoothed using a maximum per-frame change limit ("slew").

This reduces twitching caused by small detection noise and makes behavior feel more stable and hardware-ready.

### Outcome

- Smoother and more realistic tracking
- Less jitter near the centerline
- Cleaner transitions between SEARCH → TURN → FORWARD → STOP
- Better foundation for porting to real differential-drive motors (TB6612FNG)
