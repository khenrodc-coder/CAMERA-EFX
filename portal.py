"""Cinematic portal rendering with energy effects, particles, distortion, and animation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import math

import config
from filters import GEOMETRIC_COLORS


def order_quad(points: np.ndarray, min_area: float = config.MIN_PORTAL_AREA):
    """Order four points into a convex quadrilateral (TL, TR, BR, BL)."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    hull = cv2.convexHull(points, clockwise=False).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < min_area:
        return None
    if np.min(np.linalg.norm(hull - np.roll(hull, 1, axis=0), axis=1)) < 5:
        return None
    start = np.lexsort((hull[:, 0], hull[:, 1]))[0]
    return np.roll(hull, -start, axis=0)


@dataclass
class PortalState:
    """Portal state with temporal smoothing and hold/fade (backward compatible)."""
    smoothing: float = config.PORTAL_SMOOTHING
    hold: float = config.TRACKING_HOLD
    fade: float = config.TRACKING_FADE

    # Portal geometry
    corners: Optional[np.ndarray] = None
    last_seen: Optional[float] = None
    last_update: Optional[float] = None

    def reset(self) -> None:
        self.corners = None
        self.last_seen = None
        self.last_update = None

    def update(self, points, now: float):
        """Update portal state and return corners with opacity (original behavior)."""
        candidate = order_quad(points) if points is not None else None

        if candidate is not None:
            if self.corners is not None:
                # Match cyclic corners by distance, independent of handedness.
                choices = [np.roll(candidate, k, axis=0) for k in range(4)]
                candidate = min(choices, key=lambda p: np.sum((p - self.corners) ** 2))
                dt = max(0.001, now - self.last_update)
                weight = 1 - (1 - self.smoothing) ** (dt * 30)
                smoothed = self.corners + weight * (candidate - self.corners)
                if (cv2.isContourConvex(smoothed)
                        and cv2.contourArea(smoothed) >= config.MIN_PORTAL_AREA):
                    candidate = smoothed
            self.corners = candidate
            self.last_seen = now
        self.last_update = now

        if self.corners is None:
            return None, 0.0

        age = now - self.last_seen
        alpha = max(0.0, 1.0 - max(0.0, age - self.hold) / self.fade)
        if alpha == 0:
            self.corners = None
        return self.corners, alpha


# ============================================================
# Cinematic Portal Rendering (Enhanced)
# ============================================================

# Module-level portal state for cinematic features (separate from PortalState)
_cinematic_state = {
    'state': 'closed',
    'state_start_time': 0.0,
    'energy_level': 0.0,
    'charging': False,
    'charge_start_time': 0.0,
    'grabbed': False,
    'grab_offset': None,
    'grab_hand': '',
    'color_hue': 0.6,
    'intensity': 1.0,
    'rotation': 0.0,
    'scale': 1.0,
}


def _reset_cinematic_state():
    """Reset cinematic portal state."""
    global _cinematic_state
    _cinematic_state = {
        'state': 'closed',
        'state_start_time': 0.0,
        'energy_level': 0.0,
        'charging': False,
        'charge_start_time': 0.0,
        'grabbed': False,
        'grab_offset': None,
        'grab_hand': '',
        'color_hue': 0.6,
        'intensity': 1.0,
        'rotation': 0.0,
        'scale': 1.0,
    }


def _update_cinematic_state(corners, opacity, now):
    """Update cinematic portal state based on tracking."""
    has_tracking = corners is not None and opacity > 0

    if _cinematic_state['state'] == 'closed':
        if has_tracking:
            _cinematic_state['state'] = 'opening'
            _cinematic_state['state_start_time'] = now
    elif _cinematic_state['state'] == 'opening':
        if not has_tracking:
            _cinematic_state['state'] = 'closing'
            _cinematic_state['state_start_time'] = now
        elif now - _cinematic_state['state_start_time'] >= config.PORTAL_OPEN_DURATION:
            _cinematic_state['state'] = 'open'
    elif _cinematic_state['state'] == 'open':
        if not has_tracking:
            _cinematic_state['state'] = 'closing'
            _cinematic_state['state_start_time'] = now
        elif _cinematic_state['charging']:
            _cinematic_state['state'] = 'charging'
            _cinematic_state['state_start_time'] = now
    elif _cinematic_state['state'] == 'charging':
        if not _cinematic_state['charging']:
            _cinematic_state['state'] = 'releasing'
            _cinematic_state['state_start_time'] = now
        elif not has_tracking:
            _cinematic_state['state'] = 'closing'
            _cinematic_state['state_start_time'] = now
    elif _cinematic_state['state'] == 'releasing':
        if now - _cinematic_state['state_start_time'] >= config.SHOCKWAVE_DURATION:
            _cinematic_state['state'] = 'open'
            _cinematic_state['charging'] = False
    elif _cinematic_state['state'] == 'closing':
        if has_tracking:
            _cinematic_state['state'] = 'open'
        elif now - _cinematic_state['state_start_time'] >= config.PORTAL_CLOSE_DURATION:
            _cinematic_state['state'] = 'closed'


def render_portal(frame: np.ndarray, corners: np.ndarray, opacity: float,
                  effect: Callable, now: float, decorate: bool = True) -> np.ndarray:
    """Render portal with backward-compatible compositing when decorate=False."""
    # Update cinematic state
    _update_cinematic_state(corners, opacity, now)

    output = frame.copy()
    if opacity <= 0 or corners is None:
        return output

    height, width = frame.shape[:2]
    # Use original padding/feathering when not decorating for test compatibility
    if decorate:
        padding = 48
        blur_kernel = (15, 15)
    else:
        padding = 24
        blur_kernel = (5, 5)

    x0, y0 = np.maximum(np.floor(corners.min(axis=0)) - padding, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(corners.max(axis=0)) + padding + 1,
                       [width, height]).astype(int)
    if x0 >= x1 or y0 >= y1:
        return output

    roi = frame[y0:y1, x0:x1]
    local_corners = np.rint(corners - [x0, y0]).astype(np.int32)

    # Create feathered mask
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_corners, 255, lineType=cv2.LINE_AA)
    feather = cv2.GaussianBlur(mask, blur_kernel, 0).astype(np.float32) / 255
    feather *= mask.astype(np.float32) / 255
    alpha = (feather * opacity)[..., None]

    # Apply effect
    if effect in GEOMETRIC_COLORS:
        filtered = effect(roi, now, corners=corners - [x0, y0])
    else:
        filtered = effect(roi, now)

    # Base compositing - original formula for backward compatibility
    if decorate:
        portal_color = hsv_to_bgr(_cinematic_state['color_hue'], 0.8, 1.0)
        blended = roi.astype(np.float32) * (1 - alpha) + filtered * alpha * 0.7
        # Add portal-specific cinematic effects
        blended = add_cinematic_effects(blended, local_corners, mask, alpha,
                                        portal_color, opacity, now)
    else:
        # Original formula for test compatibility
        blended = roi.astype(np.float32) * (1 - alpha) + filtered * alpha

    output[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def add_cinematic_effects(blended: np.ndarray, corners: np.ndarray,
                          mask: np.ndarray, alpha: np.ndarray,
                          color: tuple, opacity: float, now: float) -> np.ndarray:
    """Add cinematic portal effects: rings, particles, lightning, glow."""

    # Energy rings
    blended = draw_energy_rings(blended, corners, color, opacity, now)

    # Border glow
    blended = draw_border_glow(blended, corners, color, opacity)

    # Lightning/electric arcs
    blended = draw_lightning(blended, corners, color, opacity, now)

    # Core glow (additive)
    blended = draw_core_glow(blended, corners, color, opacity)

    # Particles
    blended = draw_particles(blended, corners, color, opacity, now)

    # Shockwave if releasing
    if _cinematic_state['state'] == 'releasing':
        blended = draw_shockwave(blended, corners, color, opacity, now)

    return blended


def draw_energy_rings(blended: np.ndarray, corners: np.ndarray,
                      color: tuple, opacity: float, now: float) -> np.ndarray:
    """Draw animated concentric energy rings."""
    center = np.mean(corners, axis=0).astype(np.float32)
    radius = np.mean(np.linalg.norm(corners - center, axis=1))

    for i in range(config.ENERGY_RING_COUNT):
        ring_radius = radius * (0.85 + i * 0.08)
        phase = (now * 0.5 + i * 0.8) % (2 * np.pi)
        ring_alpha = opacity * 0.4 * (1.0 + 0.3 * np.sin(phase))

        ring_color = tuple(int(c * (0.7 + 0.3 * np.sin(phase))) for c in color)
        cv2.circle(blended, tuple(np.rint(center).astype(int)),
                   int(ring_radius), ring_color, 2, cv2.LINE_AA)

        cv2.circle(blended, tuple(np.rint(center).astype(int)),
                   int(ring_radius * 1.1), ring_color, 1, cv2.LINE_AA)

    return blended


def draw_border_glow(blended: np.ndarray, corners: np.ndarray,
                     color: tuple, opacity: float) -> np.ndarray:
    """Draw glowing border around portal."""
    cv2.polylines(blended, [corners.astype(np.int32)], True, color,
                  config.PORTAL_BORDER_WIDTH, cv2.LINE_AA)

    border_layer = np.zeros_like(blended)
    cv2.polylines(border_layer, [corners.astype(np.int32)], True, color,
                  config.PORTAL_BORDER_WIDTH * 2, cv2.LINE_AA)
    glow = cv2.GaussianBlur(border_layer, (0, 0), 8)
    blended += glow.astype(np.float32) * config.GLOW_INTENSITY * opacity

    return blended


def draw_lightning(blended: np.ndarray, corners: np.ndarray,
                   color: tuple, opacity: float, now: float) -> np.ndarray:
    """Draw electric lightning arcs across portal."""
    center = np.mean(corners, axis=0)

    for i in range(config.LIGHTNING_SEGMENTS):
        seed = int(now * 1000 + i * 12345) % 100000
        np.random.seed(seed)

        edge = np.random.randint(4)
        t = np.random.random()
        start = corners[edge] * (1 - t) + corners[(edge + 1) % 4] * t

        mid = (start + center) * 0.5
        jitter = np.random.uniform(-20, 20, 2)
        mid += jitter

        end = center + np.random.uniform(-10, 10, 2)

        pts = np.array([start, mid, end], dtype=np.float32)
        pts = subdivide_lightning(pts, 3)

        for j in range(len(pts) - 1):
            thickness = max(1, int(3 * (1 - j / len(pts))))
            alpha_val = opacity * 0.6 * (1 - j / len(pts))
            lightning_color = tuple(int(c * alpha_val * 255) for c in color)
            cv2.line(blended, tuple(np.rint(pts[j]).astype(int)),
                     tuple(np.rint(pts[j + 1]).astype(int)),
                     lightning_color, thickness, cv2.LINE_AA)

        if np.random.random() < 0.3:
            branch_start = pts[len(pts) // 2]
            branch_end = branch_start + np.random.uniform(-30, 30, 2)
            cv2.line(blended, tuple(np.rint(branch_start).astype(int)),
                     tuple(np.rint(branch_end).astype(int)),
                     tuple(int(c * opacity * 0.4 * 255) for c in color), 1, cv2.LINE_AA)

    np.random.seed()
    return blended


def subdivide_lightning(pts: np.ndarray, iterations: int) -> np.ndarray:
    """Subdivide lightning path for natural jagged look."""
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            mid = (pts[i] + pts[i + 1]) * 0.5
            dir_vec = pts[i + 1] - pts[i]
            length = np.linalg.norm(dir_vec)
            if length > 0:
                perp = np.array([-dir_vec[1], dir_vec[0]]) / length
                mid += perp * np.random.uniform(-length * 0.3, length * 0.3)
            new_pts.extend([mid, pts[i + 1]])
        pts = np.array(new_pts)
    return pts


def draw_core_glow(blended: np.ndarray, corners: np.ndarray,
                   color: tuple, opacity: float) -> np.ndarray:
    """Draw bright core in portal center (additive blending)."""
    center = np.mean(corners, axis=0).astype(np.float32)
    radius = np.mean(np.linalg.norm(corners - center, axis=1)) * 0.3

    h, w = blended.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    gradient = np.clip(1.0 - dist / radius, 0, 1)
    gradient = gradient ** 2

    glow_color = np.array(color, dtype=np.float32) * 255
    for c in range(3):
        blended[:, :, c] += gradient * glow_color[c] * opacity * 0.5

    return blended


def draw_particles(blended: np.ndarray, corners: np.ndarray,
                   color: tuple, opacity: float, now: float) -> np.ndarray:
    """Draw animated particles orbiting and emanating from portal."""
    center = np.mean(corners, axis=0).astype(np.float32)
    radius = np.mean(np.linalg.norm(corners - center, axis=1))

    # Orbiting particles
    for i in range(config.PARTICLE_COUNT):
        orbit_radius = radius * (0.4 + 0.5 * (i / config.PARTICLE_COUNT))
        angle = now * (0.5 + 0.3 * (i % 5)) + i * 2 * np.pi / config.PARTICLE_COUNT

        x = center[0] + orbit_radius * np.cos(angle)
        y = center[1] + orbit_radius * 0.6 * np.sin(angle)

        size = 1 + (i % 4)
        particle_alpha = opacity * (0.5 + 0.5 * np.sin(now * 2 + i))

        particle_color = tuple(int(c * particle_alpha * 255) for c in color)
        cv2.circle(blended, (int(x), int(y)), size, particle_color, -1, cv2.LINE_AA)

        if i % 3 == 0:
            trail_angle = angle - 0.2
            tx = center[0] + orbit_radius * np.cos(trail_angle)
            ty = center[1] + orbit_radius * 0.6 * np.sin(trail_angle)
            cv2.circle(blended, (int(tx), int(ty)), 1,
                       tuple(int(c * particle_alpha * 0.5 * 255) for c in color), -1, cv2.LINE_AA)

    # Radial emission particles
    for i in range(config.PARTICLE_COUNT // 2):
        emit_angle = (now * 3 + i * 4 * np.pi / (config.PARTICLE_COUNT // 2)) % (2 * np.pi)
        emit_dist = (now * 100 + i * 50) % (radius * 1.5)
        x = center[0] + emit_dist * np.cos(emit_angle)
        y = center[1] + emit_dist * np.sin(emit_angle)
        alpha = opacity * max(0, 1 - emit_dist / (radius * 1.5))
        size = max(1, int(3 * alpha))
        cv2.circle(blended, (int(x), int(y)), size,
                   tuple(int(c * alpha * 255) for c in color), -1, cv2.LINE_AA)

    return blended


def draw_shockwave(blended: np.ndarray, corners: np.ndarray,
                   color: tuple, opacity: float, now: float) -> np.ndarray:
    """Draw expanding shockwave ring."""
    center = np.mean(corners, axis=0).astype(np.float32)
    radius = np.mean(np.linalg.norm(corners - center, axis=1))

    progress = (now - _cinematic_state['state_start_time']) / config.SHOCKWAVE_DURATION
    progress = min(1.0, max(0.0, progress))
    wave_radius = radius * (1.0 + 0.5 * progress)
    wave_alpha = opacity * 0.5 * (1.0 - progress)

    cv2.circle(blended, tuple(np.rint(center).astype(int)),
               int(wave_radius), tuple(int(c * wave_alpha * 255) for c in color),
               3, cv2.LINE_AA)

    return blended


def hsv_to_bgr(h: float, s: float, v: float) -> tuple:
    """Convert HSV (0-1) to BGR (0-255)."""
    h = h * 180
    s = s * 255
    v = v * 255
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


# Cinematic portal control functions (for gesture integration)
def cinematic_start_charge(now: float):
    """Start energy charging."""
    if _cinematic_state['state'] in ('open', 'charging'):
        _cinematic_state['charging'] = True
        _cinematic_state['charge_start_time'] = now
        _cinematic_state['state'] = 'charging'


def cinematic_release_energy(now: float):
    """Release charged energy."""
    if _cinematic_state['charging']:
        _cinematic_state['charging'] = False
        _cinematic_state['state'] = 'releasing'
        _cinematic_state['state_start_time'] = now
        _cinematic_state['energy_level'] = 0.0


def cinematic_update_charge(now: float):
    """Update charge level during charging."""
    if _cinematic_state['charging'] and _cinematic_state['state'] == 'charging':
        elapsed = now - _cinematic_state['charge_start_time']
        _cinematic_state['energy_level'] = min(1.0, elapsed / config.ENERGY_CHARGE_DURATION)


def cinematic_grab(hand: str, cursor: np.ndarray, now: float):
    """Grab portal for dragging."""
    _cinematic_state['grabbed'] = True
    _cinematic_state['grab_hand'] = hand
    _cinematic_state['grab_offset'] = cursor.copy()


def cinematic_release_grab():
    """Release portal grab."""
    _cinematic_state['grabbed'] = False
    _cinematic_state['grab_offset'] = None
    _cinematic_state['grab_hand'] = ''


def cinematic_move_grabbed(cursor: np.ndarray, corners: np.ndarray):
    """Move portal while grabbed."""
    if _cinematic_state['grabbed'] and _cinematic_state['grab_offset'] is not None and corners is not None:
        return cursor - _cinematic_state['grab_offset']
    return None


def cinematic_scale_portal(factor: float, center: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Scale portal around center."""
    if corners is not None:
        return center + (corners - center) * factor
    return corners


def cinematic_rotate_portal(angle: float, center: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Rotate portal around center."""
    if corners is not None:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return center + (corners - center) @ rot.T
    return corners


def cinematic_shift_color(delta_hue: float):
    """Shift portal color hue."""
    _cinematic_state['color_hue'] = (_cinematic_state['color_hue'] + delta_hue) % 1.0


def cinematic_set_intensity(intensity: float):
    """Set portal intensity."""
    _cinematic_state['intensity'] = np.clip(intensity, 0.1, 2.0)


def cinematic_reset():
    """Reset cinematic portal state."""
    _reset_cinematic_state()


# Expose cinematic state for UI
def get_cinematic_state():
    """Get current cinematic portal state."""
    return _cinematic_state.copy()