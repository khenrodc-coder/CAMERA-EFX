"""Advanced gesture recognition system with multiple gesture types and state management."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np

import config
from hand_tracking import Hand, FingerState


class GestureType(Enum):
    """Supported gesture types."""
    NONE = auto()
    PINCH = auto()
    OPEN_PALM = auto()
    CLOSED_FIST = auto()
    PEACE_SIGN = auto()
    THUMBS_UP = auto()
    POINTING = auto()
    TWO_FINGER = auto()  # Index + middle extended
    SNAP = auto()  # Thumb + middle finger snap

    # Two-hand gestures
    BOTH_HANDS_OPEN = auto()
    HANDS_EXPANDING = auto()
    HANDS_SHRINKING = auto()
    HANDS_ROTATING_CW = auto()
    HANDS_ROTATING_CCW = auto()
    BOTH_HANDS_PINCH = auto()  # Both hands pinching
    HANDS_MOVING_UP = auto()
    HANDS_MOVING_DOWN = auto()


@dataclass
class GestureState:
    """State of a recognized gesture."""
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    handedness: str = ""  # "Left", "Right", or "Both"
    start_time: float = 0.0
    duration: float = 0.0
    data: dict = field(default_factory=dict)  # Additional gesture-specific data


class GestureRecognizer:
    """Advanced gesture recognition with multiple gesture types and cooldowns."""

    def __init__(self):
        # Cooldown tracking per gesture type
        self.cooldowns: dict[GestureType, float] = {}
        # State tracking for continuous gestures
        self.last_hand_positions: dict[str, np.ndarray] = {}
        self.last_hand_angles: dict[str, float] = {}
        self.both_hands_open_start: Optional[float] = None
        # Finger snap detection state
        self.last_snap_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # (thumb_tip, middle_tip)
        self.last_snap_time: dict[str, float] = {}
        self.reset()

    def reset(self):
        """Reset all gesture state."""
        self.current_gesture = GestureState()
        self.pinch_state = PinchState()
        self.cooldowns.clear()
        self.last_hand_positions.clear()
        self.last_hand_angles.clear()
        self.both_hands_open_start = None
        self.last_snap_positions.clear()
        self.last_snap_time.clear()

    def update(self, hands: list[Hand], now: float) -> Optional[GestureState]:
        """
        Update gesture recognition with current hand data.
        Returns a GestureState if a new gesture was recognized, None otherwise.
        """
        # Update pinch detection (legacy compatibility)
        pinch_triggered = self.pinch_state.update(hands, now)

        # Check cooldowns
        self._update_cooldowns(now)

        # Single-hand gestures
        single_hand_gesture = self._recognize_single_hand_gestures(hands, now)

        # Two-hand gestures
        two_hand_gesture = self._recognize_two_hand_gestures(hands, now)

        # Determine highest priority gesture
        new_gesture = self._resolve_gesture_priority(
            pinch_triggered, single_hand_gesture, two_hand_gesture, now
        )

        if new_gesture and new_gesture.gesture != GestureType.NONE:
            # Check cooldown
            if self._check_cooldown(new_gesture.gesture, now):
                self.current_gesture = new_gesture
                self._set_cooldown(new_gesture.gesture, now)
                return self.current_gesture

        # Update current gesture duration
        if self.current_gesture.gesture != GestureType.NONE:
            self.current_gesture.duration = now - self.current_gesture.start_time

        return None

    def _recognize_single_hand_gestures(self, hands: list[Hand], now: float) -> Optional[GestureState]:
        """Recognize single-hand gestures."""
        for hand in hands:
            if hand.confidence < config.GESTURE_CONFIDENCE_THRESHOLD:
                continue

            # Finger snap has highest priority - check FIRST
            snap_gesture = self._detect_finger_snap(hand, now)
            if snap_gesture:
                return snap_gesture

            extended_count = sum(1 for f in hand.fingers if f.extended)
            finger_states = [f.extended for f in hand.fingers]

            # Open palm: 4+ fingers extended
            if extended_count >= config.OPEN_PALM_THRESHOLD:
                return GestureState(
                    gesture=GestureType.OPEN_PALM,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={"extended_count": extended_count, "palm_center": hand.palm_center}
                )

            # Closed fist: 0 fingers extended
            if extended_count <= config.FIST_THRESHOLD:
                return GestureState(
                    gesture=GestureType.CLOSED_FIST,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={"palm_center": hand.palm_center}
                )

            # Peace sign: exactly index and middle extended
            if (extended_count == config.PEACE_THRESHOLD and
                finger_states[1] and finger_states[2] and  # index, middle
                not finger_states[0] and not finger_states[3] and not finger_states[4]):
                return GestureState(
                    gesture=GestureType.PEACE_SIGN,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={"palm_center": hand.palm_center}
                )

            # Thumbs up: only thumb extended
            if (extended_count == 1 and finger_states[0] and
                not any(finger_states[1:])):
                return GestureState(
                    gesture=GestureType.THUMBS_UP,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={"palm_center": hand.palm_center}
                )

            # Pointing: only index extended
            if (extended_count == config.POINTING_THRESHOLD and
                finger_states[1] and not finger_states[0] and
                not any(finger_states[2:])):
                return GestureState(
                    gesture=GestureType.POINTING,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={
                        "palm_center": hand.palm_center,
                        "point_direction": hand.fingers[1].tip_position - hand.palm_center
                    }
                )

            # Two finger: index + middle extended (like peace but thumb may be extended)
            if (finger_states[1] and finger_states[2] and
                not finger_states[3] and not finger_states[4]):
                return GestureState(
                    gesture=GestureType.TWO_FINGER,
                    confidence=hand.confidence,
                    handedness=hand.handedness,
                    start_time=now,
                    data={"palm_center": hand.palm_center}
                )

        return None

    def _recognize_two_hand_gestures(self, hands: list[Hand], now: float) -> Optional[GestureState]:
        """Recognize two-hand compound gestures."""
        if len(hands) != 2:
            self.both_hands_open_start = None
            return None

        left_hand = next((h for h in hands if h.handedness == "Left"), None)
        right_hand = next((h for h in hands if h.handedness == "Right"), None)

        if not left_hand or not right_hand:
            return None

        # Both hands open
        left_extended = sum(1 for f in left_hand.fingers if f.extended)
        right_extended = sum(1 for f in right_hand.fingers if f.extended)

        if left_extended >= config.OPEN_PALM_THRESHOLD and right_extended >= config.OPEN_PALM_THRESHOLD:
            if self.both_hands_open_start is None:
                self.both_hands_open_start = now
            elif now - self.both_hands_open_start >= config.BOTH_HANDS_OPEN_DWELL:
                return GestureState(
                    gesture=GestureType.BOTH_HANDS_OPEN,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=self.both_hands_open_start,
                    data={
                        "left_palm": left_hand.palm_center,
                        "right_palm": right_hand.palm_center,
                        "center": (left_hand.palm_center + right_hand.palm_center) / 2
                    }
                )
        else:
            self.both_hands_open_start = None

        # Track hand positions for motion gestures
        left_pos = left_hand.palm_center
        right_pos = right_hand.palm_center

        # Calculate hand separation and movement
        separation = np.linalg.norm(right_pos - left_pos)
        center = (left_pos + right_pos) / 2

        # Hands expanding/shrinking
        prev_separation = getattr(self, '_prev_separation', separation)
        separation_speed = (separation - prev_separation) * 30  # pixels per second at 30fps
        self._prev_separation = separation

        if abs(separation_speed) > config.HAND_SEPARATION_SPEED:
            if separation_speed > 0:
                return GestureState(
                    gesture=GestureType.HANDS_EXPANDING,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"separation": separation, "speed": separation_speed, "center": center}
                )
            else:
                return GestureState(
                    gesture=GestureType.HANDS_SHRINKING,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"separation": separation, "speed": separation_speed, "center": center}
                )

        # Hands rotating - check angle change
        left_angle = left_hand.palm_angle
        right_angle = right_hand.palm_angle
        avg_angle = (left_angle + right_angle) / 2

        prev_avg_angle = getattr(self, '_prev_avg_angle', avg_angle)
        angle_delta = self._angle_diff(prev_avg_angle, avg_angle)
        rotation_speed = angle_delta * 30  # rad/sec
        self._prev_avg_angle = avg_angle

        if abs(rotation_speed) > config.ROTATION_SPEED:
            if rotation_speed > 0:
                return GestureState(
                    gesture=GestureType.HANDS_ROTATING_CW,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"rotation_speed": rotation_speed, "center": center}
                )
            else:
                return GestureState(
                    gesture=GestureType.HANDS_ROTATING_CCW,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"rotation_speed": rotation_speed, "center": center}
                )

        # Both hands pinching
        if left_hand.pinch_ratio < config.PINCH_CLOSE_RATIO and right_hand.pinch_ratio < config.PINCH_CLOSE_RATIO:
            return GestureState(
                gesture=GestureType.BOTH_HANDS_PINCH,
                confidence=min(left_hand.confidence, right_hand.confidence),
                handedness="Both",
                start_time=now,
                data={
                    "left_pinch": left_hand.pinch_distance,
                    "right_pinch": right_hand.pinch_distance,
                    "center": center
                }
            )

        # Hands moving up/down (vertical movement of center)
        prev_center = getattr(self, '_prev_center', center)
        vertical_speed = (center[1] - prev_center[1]) * 30  # pixels/sec (negative = up)
        self._prev_center = center

        if abs(vertical_speed) > config.HAND_SEPARATION_SPEED:  # Reuse threshold
            if vertical_speed < 0:  # Moving up (negative Y)
                return GestureState(
                    gesture=GestureType.HANDS_MOVING_UP,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"vertical_speed": vertical_speed, "center": center}
                )
            else:
                return GestureState(
                    gesture=GestureType.HANDS_MOVING_DOWN,
                    confidence=min(left_hand.confidence, right_hand.confidence),
                    handedness="Both",
                    start_time=now,
                    data={"vertical_speed": vertical_speed, "center": center}
                )

        return None

    def _detect_finger_snap(self, hand: Hand, now: float) -> Optional[GestureState]:
        """Detect finger snap gesture (thumb + middle finger).

        A snap is detected when:
        1. Fingers were close together in previous frame (normalized distance < SNAP_MAX_DISTANCE)
        2. Middle finger moves away rapidly from thumb (high relative velocity)
        """
        # Check cooldown per hand
        if hand.handedness in self.last_snap_time:
            if now - self.last_snap_time[hand.handedness] < config.SNAP_COOLDOWN:
                return None

        # Get thumb tip (index 4) and middle finger tip (index 12)
        thumb_tip = hand.points[4]
        middle_tip = hand.points[12]

        # Check velocity - need previous position to calculate
        prev_positions = self.last_snap_positions.get(hand.handedness)
        if prev_positions is not None:
            prev_thumb, prev_middle = prev_positions

            # Check if fingers WERE close in previous frame (snap starts with fingers touching)
            prev_distance = np.linalg.norm(prev_thumb - prev_middle)
            prev_normalized = prev_distance / max(hand.palm_size, 1.0)

            # Calculate velocity of middle finger relative to thumb
            thumb_vel = np.linalg.norm(thumb_tip - prev_thumb)
            middle_vel = np.linalg.norm(middle_tip - prev_middle)
            relative_vel = abs(middle_vel - thumb_vel)

            # Snap detected if: was close AND moving apart rapidly
            if prev_normalized < config.SNAP_MAX_DISTANCE and relative_vel > config.SNAP_MIN_VELOCITY / 30.0:
                # Record snap time
                self.last_snap_time[hand.handedness] = now
                self.last_snap_positions[hand.handedness] = (thumb_tip.copy(), middle_tip.copy())
                return GestureState(
                    gesture=GestureType.SNAP,
                    confidence=min(1.0, relative_vel / (config.SNAP_MIN_VELOCITY / 30.0)),
                    handedness=hand.handedness,
                    start_time=now,
                    data={
                        "thumb_tip": thumb_tip,
                        "middle_tip": middle_tip,
                        "velocity": relative_vel * 30,
                        "palm_center": hand.palm_center
                    }
                )

        # Update last positions
        self.last_snap_positions[hand.handedness] = (thumb_tip.copy(), middle_tip.copy())
        return None

    def _resolve_gesture_priority(self, pinch_triggered: bool,
                                   single: Optional[GestureState],
                                   two_hand: Optional[GestureState],
                                   now: float) -> Optional[GestureState]:
        """Resolve which gesture takes priority."""
        # SNAP has highest priority (unique gesture, should trigger immediately)
        if single and single.gesture == GestureType.SNAP:
            return single

        # Pinch has high priority for effect switching (legacy)
        if pinch_triggered:
            return GestureState(
                gesture=GestureType.PINCH,
                confidence=1.0,
                handedness="Either",
                start_time=now,
                data={}
            )

        # Two-hand gestures have priority over single-hand
        if two_hand:
            return two_hand

        # Single-hand gestures
        if single:
            return single

        return None

    def _check_cooldown(self, gesture: GestureType, now: float) -> bool:
        """Check if gesture is off cooldown."""
        if gesture not in self.cooldowns:
            return True
        return now - self.cooldowns[gesture] >= config.GESTURE_COOLDOWN

    def _set_cooldown(self, gesture: GestureType, now: float):
        """Set cooldown for gesture."""
        self.cooldowns[gesture] = now

    def _update_cooldowns(self, now: float):
        """Clean up expired cooldowns."""
        expired = [g for g, t in self.cooldowns.items() if now - t > config.GESTURE_COOLDOWN * 2]
        for g in expired:
            del self.cooldowns[g]

    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Shortest signed difference between two angles."""
        diff = a2 - a1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff


class PinchState:
    """Legacy pinch detection for backward compatibility."""

    def __init__(self, dwell: float = config.GESTURE_DWELL,
                 cooldown: float = config.GESTURE_COOLDOWN,
                 release: float = config.GESTURE_RELEASE):
        self.dwell = dwell
        self.cooldown = cooldown
        self.release = release
        self.reset()

    def reset(self):
        self.started = None
        self.released = None
        self.latched = False
        self.pinching = False
        self.required_hands = 0
        self.last_switch = float("-inf")
        self.progress = 0.0

    def update(self, hands, now: float) -> bool:
        """Accept either Hand objects or raw numpy arrays for backward compatibility."""
        ratios = []
        for hand in hands:
            if hasattr(hand, 'pinch_ratio'):
                if hand.pinch_ratio > 0:
                    ratios.append(hand.pinch_ratio)
            elif isinstance(hand, np.ndarray):
                # Raw landmark array - compute pinch ratio
                ratio = pinch_ratio(hand)
                if ratio is not None and ratio > 0:
                    ratios.append(ratio)
        self.progress = 0.0
        if not ratios:
            self.started = self.released = None
            self.pinching = False
            return False

        touching = min(ratios) <= config.PINCH_CLOSE_RATIO
        separated = min(ratios) >= config.PINCH_RELEASE_RATIO
        self.pinching = touching or (self.pinching and not separated)

        if separated:
            self.started = None
            if len(ratios) < self.required_hands:
                self.released = None
                return False
            if self.released is None:
                self.released = now
            if now - self.released >= self.release:
                self.latched = False
                self.required_hands = 0
            return False

        self.released = None
        if self.latched:
            return False
        if not touching:
            self.started = None
            return False

        if self.started is None:
            self.started = now
        self.progress = min(1.0, (now - self.started) / self.dwell)
        if now - self.started < self.dwell:
            return False

        self.latched = True
        self.required_hands = len(ratios)
        self.started = None
        if now - self.last_switch < self.cooldown:
            return False
        self.last_switch = now
        return True


# Backward compatibility alias
PinchSwitch = PinchState


def pinch_ratio(points: np.ndarray) -> float | None:
    """Legacy function for backward compatibility."""
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    palm = max(np.linalg.norm(points[9] - points[0]),
               np.linalg.norm(points[17] - points[5]))
    if palm < 5:
        return None
    return float(np.linalg.norm(points[4] - points[8]) / palm)