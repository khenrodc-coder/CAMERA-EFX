from pathlib import Path
import numpy as np

# Camera settings
CAMERA_INDEX = 0  # 0 = default camera, will auto-detect
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DETECTION_WIDTH = 640
TARGET_FPS = 30
CAMERA_BUFFER_SIZE = 1

# Model settings
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_DETECTION_CONFIDENCE = 0.7
HAND_PRESENCE_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
MAX_HANDS = 2

# Legacy alias for backward compatibility
HAND_CONFIDENCE = HAND_DETECTION_CONFIDENCE

# Landmark smoothing
LANDMARK_SMOOTHING_ALPHA = 0.35
LANDMARK_HISTORY_SIZE = 5

# Finger detection thresholds
FINGER_EXTENSION_RATIO = 1.1  # tip_dist / pip_dist for extension
THUMB_EXTENSION_ANGLE = np.pi * 0.6  # ~108 degrees

# Portal geometry
PORTAL_SMOOTHING = 0.45  # EMA weight at 30 FPS; adjusted for elapsed time.
MIN_PORTAL_AREA = 700.0
TRACKING_HOLD = 0.25
TRACKING_FADE = 0.35

# Gesture settings
GESTURE_DWELL = 0.08
GESTURE_RELEASE = 0.12
GESTURE_COOLDOWN = 0.35
PINCH_CLOSE_RATIO = 0.23
PINCH_RELEASE_RATIO = 0.42

# Advanced gesture thresholds
GESTURE_CONFIDENCE_THRESHOLD = 0.6
OPEN_PALM_THRESHOLD = 4  # min extended fingers
FIST_THRESHOLD = 0       # max extended fingers
PEACE_THRESHOLD = 2      # exact extended fingers (index, middle)
POINTING_THRESHOLD = 1   # exact extended finger (index)

# Two-hand gestures
BOTH_HANDS_OPEN_DWELL = 0.5
HAND_SEPARATION_SPEED = 100.0  # pixels/sec for expand/shrink
ROTATION_SPEED = 2.0  # rad/sec for rotation gesture

# Visual effects
PARTICLE_COUNT = 48
GLOW_INTENSITY = 1.0
PORTAL_BORDER_WIDTH = 4
ENERGY_RING_COUNT = 3
LIGHTNING_SEGMENTS = 8

# Portal animation
PORTAL_OPEN_DURATION = 0.5
PORTAL_CLOSE_DURATION = 0.4
ENERGY_CHARGE_DURATION = 1.0
SHOCKWAVE_DURATION = 0.6

# Display
SHOW_FPS = True
SHOW_LANDMARKS = False
SHOW_GESTURE_DEBUG = False
MIRROR_PREVIEW = True

# Performance presets
PERFORMANCE_PRESETS = {
    "low": {
        "FRAME_WIDTH": 640,
        "FRAME_HEIGHT": 480,
        "DETECTION_WIDTH": 320,
        "PARTICLE_COUNT": 16,
        "ENERGY_RING_COUNT": 1,
        "LIGHTNING_SEGMENTS": 4,
    },
    "medium": {
        "FRAME_WIDTH": 1280,
        "FRAME_HEIGHT": 720,
        "DETECTION_WIDTH": 640,
        "PARTICLE_COUNT": 32,
        "ENERGY_RING_COUNT": 2,
        "LIGHTNING_SEGMENTS": 6,
    },
    "high": {
        "FRAME_WIDTH": 1920,
        "FRAME_HEIGHT": 1080,
        "DETECTION_WIDTH": 640,
        "PARTICLE_COUNT": 64,
        "ENERGY_RING_COUNT": 4,
        "LIGHTNING_SEGMENTS": 10,
    },
}
CURRENT_PRESET = "medium"
CURRENT_PORTAL_HUE = 0.6