"""Enhanced hand tracking with finger states, palm orientation, smoothing, and jitter reduction."""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

import config


@dataclass
class FingerState:
    """State of a single finger."""
    extended: bool = False
    confidence: float = 0.0
    tip_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    pip_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    mcp_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass
class Hand:
    """Enhanced hand data with finger states and palm geometry."""
    points: np.ndarray                          # (21, 2) landmark positions in image coordinates
    handedness: str                             # "Left" or "Right"
    confidence: float                           # Detection confidence [0, 1]
    presence_confidence: float                  # Presence confidence [0, 1]
    tracking_confidence: float                  # Tracking confidence [0, 1]

    # Finger states (thumb, index, middle, ring, pinky)
    fingers: list[FingerState] = field(default_factory=lambda: [FingerState() for _ in range(5)])

    # Palm geometry
    palm_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    palm_normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # 3D normal
    palm_size: float = 0.0                      # Palm diameter in pixels
    palm_angle: float = 0.0                     # Palm rotation in image plane (radians)

    # Derived gesture features
    pinch_distance: float = 0.0                 # Thumb-index tip distance
    pinch_ratio: float = 0.0                    # Normalized by palm size
    finger_spread: float = 0.0                  # Average finger separation


class LandmarkSmoother:
    """Temporal smoothing for hand landmarks using exponential moving average."""

    def __init__(self, alpha: float = 0.3, max_history: int = 5):
        self.alpha = alpha
        self.max_history = max_history
        self.history: deque[np.ndarray] = deque(maxlen=max_history)
        self.smoothed: Optional[np.ndarray] = None
        self.initialized = False

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to landmarks."""
        if not self.initialized:
            self.smoothed = landmarks.copy()
            self.initialized = True
        else:
            # Exponential moving average
            self.smoothed = self.alpha * landmarks + (1 - self.alpha) * self.smoothed
        self.history.append(landmarks.copy())
        return self.smoothed

    def reset(self):
        self.history.clear()
        self.smoothed = None
        self.initialized = False


class HandTracker:
    """Enhanced hand tracker with finger state detection, palm geometry, and smoothing."""

    # Landmark indices for finger tips, PIPs, MCPs, and wrist
    FINGER_INDICES = {
        'thumb': {'tip': 4, 'pip': 3, 'mcp': 2, 'cmc': 1},
        'index': {'tip': 8, 'pip': 6, 'mcp': 5},
        'middle': {'tip': 12, 'pip': 10, 'mcp': 9},
        'ring': {'tip': 16, 'pip': 14, 'mcp': 13},
        'pinky': {'tip': 20, 'pip': 18, 'mcp': 17},
    }
    WRIST = 0
    PALM_KEYPOINTS = [0, 1, 5, 9, 13, 17]  # Wrist + finger MCPs

    def __init__(self):
        if not config.MODEL_PATH.is_file():
            raise RuntimeError("Hand model missing. Run: python download_model.py")

        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(config.MODEL_PATH),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        ))

        # Smoothers for each hand (by handedness)
        self.smoothers: dict[str, LandmarkSmoother] = {}
        self.timestamp = -1
        self.last_hands: dict[str, Hand] = {}  # For lost hand recovery

    def detect(self, frame: np.ndarray, now: float) -> list[Hand]:
        """Detect hands and compute enhanced features."""
        height, width = frame.shape[:2]
        scale = min(1.0, config.DETECTION_WIDTH / width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        result = self.detector.detect_for_video(image, self.timestamp)

        hands = []
        current_handedness = set()

        for landmarks, labels in zip(result.hand_landmarks, result.handedness):
            handedness = labels[0].category_name if labels else "Unknown"
            current_handedness.add(handedness)

            # Convert to image coordinates
            points = np.array([(p.x * width, p.y * height) for p in landmarks],
                              dtype=np.float32)

            # Apply temporal smoothing
            if handedness not in self.smoothers:
                self.smoothers[handedness] = LandmarkSmoother(
                    alpha=config.LANDMARK_SMOOTHING_ALPHA,
                    max_history=config.LANDMARK_HISTORY_SIZE
                )
            points = self.smoothers[handedness].update(points)

            # Compute finger states
            fingers = self._compute_finger_states(points, handedness)

            # Compute palm geometry
            palm_center, palm_normal, palm_size, palm_angle = self._compute_palm_geometry(
                points, landmarks
            )

            # Compute derived gesture features
            pinch_dist = np.linalg.norm(points[4] - points[8])
            pinch_ratio = pinch_dist / max(palm_size, 1.0)
            finger_spread = self._compute_finger_spread(points)

            # Get confidence scores
            detection_conf = labels[0].score if labels else 0.0
            # MediaPipe doesn't expose presence/tracking separately in this API version
            presence_conf = detection_conf
            tracking_conf = detection_conf

            hand = Hand(
                points=points,
                handedness=handedness,
                confidence=detection_conf,
                presence_confidence=presence_conf,
                tracking_confidence=tracking_conf,
                fingers=fingers,
                palm_center=palm_center,
                palm_normal=palm_normal,
                palm_size=palm_size,
                palm_angle=palm_angle,
                pinch_distance=pinch_dist,
                pinch_ratio=pinch_ratio,
                finger_spread=finger_spread,
            )
            hands.append(hand)
            self.last_hands[handedness] = hand

        # Clean up smoothers for lost hands
        lost_hands = set(self.smoothers.keys()) - current_handedness
        for h in lost_hands:
            self.smoothers[h].reset()

        return hands

    def _compute_finger_states(self, points: np.ndarray, handedness: str) -> list[FingerState]:
        """Determine extended/folded state for each finger."""
        fingers = []
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        for name in finger_names:
            idx = self.FINGER_INDICES[name]
            tip = points[idx['tip']]
            pip = points[idx['pip']]
            mcp = points[idx['mcp']]

            if name == 'thumb':
                # Thumb: check angle between tip-pip-mcp
                v1 = pip - mcp
                v2 = tip - pip
                angle = self._angle_between(v1, v2)
                # Thumb extended when angle is large (straight)
                extended = angle > config.THUMB_EXTENSION_ANGLE
                confidence = min(1.0, angle / np.pi)
            else:
                # Other fingers: extended when tip is further from wrist than PIP
                wrist = points[self.WRIST]
                tip_dist = np.linalg.norm(tip - wrist)
                pip_dist = np.linalg.norm(pip - wrist)
                extended = tip_dist > pip_dist * config.FINGER_EXTENSION_RATIO
                confidence = min(1.0, tip_dist / max(pip_dist, 1.0))

            fingers.append(FingerState(
                extended=extended,
                confidence=confidence,
                tip_position=tip,
                pip_position=pip,
                mcp_position=mcp,
            ))

        return fingers

    def _compute_palm_geometry(self, points: np.ndarray,
                                landmarks) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute palm center, normal, size, and rotation angle."""
        # Palm center: average of wrist and finger MCPs
        palm_points = points[self.PALM_KEYPOINTS]
        palm_center = np.mean(palm_points, axis=0).astype(np.float32)

        # Palm size: max distance from center to MCPs
        mcp_points = points[[5, 9, 13, 17]]
        distances = np.linalg.norm(mcp_points - palm_center, axis=1)
        palm_size = float(np.max(distances) * 2)

        # Palm angle: orientation of line from wrist to middle finger MCP
        wrist = points[0]
        middle_mcp = points[9]
        palm_vector = middle_mcp - wrist
        palm_angle = float(np.arctan2(palm_vector[1], palm_vector[0]))

        # 3D palm normal (approximate from 2D landmarks)
        # Use cross product of index-mcp to pinky-mcp vectors in normalized coords
        if hasattr(landmarks[0], 'z'):
            # Has 3D data
            idx_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
            pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
            wrist_3d = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            v1 = idx_mcp - wrist_3d
            v2 = pinky_mcp - wrist_3d
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            palm_normal = (normal / norm).astype(np.float32) if norm > 0 else np.array([0, 0, 1], dtype=np.float32)
        else:
            palm_normal = np.array([0, 0, 1], dtype=np.float32)

        return palm_center, palm_normal, palm_size, palm_angle

    def _compute_finger_spread(self, points: np.ndarray) -> float:
        """Compute average angular spread between adjacent fingers."""
        mcp_indices = [5, 9, 13, 17]  # Index, middle, ring, pinky MCPs
        wrist = points[0]
        vectors = []
        for idx in mcp_indices:
            v = points[idx] - wrist
            norm = np.linalg.norm(v)
            if norm > 0:
                vectors.append(v / norm)

        if len(vectors) < 2:
            return 0.0

        angles = []
        for i in range(len(vectors) - 1):
            angle = self._angle_between(vectors[i], vectors[i + 1])
            angles.append(angle)

        return float(np.mean(angles)) if angles else 0.0

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle between two vectors in radians [0, pi]."""
        dot = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
        return float(np.arccos(dot))

    def close(self) -> None:
        self.detector.close()