"""BGR uint8 effects for portal rendering. Includes cinematic and geometric effects."""

import cv2
import numpy as np


# ============================================================
# Classic Image Filters
# ============================================================

def thermal(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def pencil_sketch(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(255 - gray, (0, 0), 3.0)
    sketch = cv2.divide(gray, np.maximum(255 - blurred, 1), scale=256)
    sketch = cv2.addWeighted(sketch, 0.88, gray, 0.12, 15)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def rgb_glitch(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    shift = max(1, round(width * 0.012 * (0.7 + 0.3 * np.sin(t * 2))))
    out = frame.copy()
    out[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    out[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
    band = int((0.5 + 0.45 * np.sin(t * 0.8)) * (height - 1))
    end = min(height, band + max(1, height // 18))
    out[band:end] = np.roll(out[band:end], shift * 2, axis=1)
    out[::5] = (out[::5].astype(np.uint16) * 9 // 10).astype(np.uint8)
    return out


def monochrome(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def dual_tone(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    weight = np.clip((gray - 0.15) / 0.7, 0, 1)[..., None]
    pink = np.array([132, 25, 225], dtype=np.float32)
    yellow = np.array([65, 235, 255], dtype=np.float32)
    return (pink + weight * (yellow - pink)).astype(np.uint8)


def inversion(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    return cv2.bitwise_not(frame)


def sepia(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    matrix = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    return cv2.transform(frame, matrix)


def blur(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    return cv2.GaussianBlur(frame, (19, 19), 0)


def cartoon(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame, 5, 55, 4)
    poster = (smooth // 32) * 32 + 16
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    return cv2.bitwise_and(poster, poster, mask=255 - edges)


def neon_edges(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    strength = np.clip(cv2.magnitude(dx, dy) * 0.012, 0, 1)
    tone = gray.astype(np.float32) / 255
    color = np.stack((np.full_like(tone, 255), 235 - 180 * tone,
                      70 + 185 * tone), axis=2)
    edges = color * strength[..., None]
    glow = cv2.GaussianBlur(edges, (0, 0), 2)
    return np.clip(frame * 0.10 + edges * 0.85 + glow * 0.6, 0, 255).astype(np.uint8)


def pixel_art(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (max(1, width // 8), max(1, height // 8)),
                       interpolation=cv2.INTER_AREA)
    palette = (small // 32) * 32 + 16
    return cv2.resize(palette, (width, height), interpolation=cv2.INTER_NEAREST)


def comic_halftone(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luminance = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255
    y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
    distance_sq = ((x % 6) - 2.5) ** 2 + ((y % 6) - 2.5) ** 2
    ink = distance_sq < (1 - luminance) * (36 / np.pi)
    paper = (frame // 64) * 64 + 32
    out = cv2.addWeighted(paper, 0.8, np.full_like(frame, 245), 0.2, 0)
    out[ink | (cv2.Canny(gray, 80, 180) != 0)] = (35, 25, 40)
    return out


def vhs(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    chroma = cv2.GaussianBlur(ycc[:, :, 1:], (7, 3), 0)
    offset = int(round(2 * np.sin(t * 1.4)))
    columns = np.clip(np.arange(width) - offset, 0, width - 1)
    ycc[:, :, 1:] = chroma[:, columns]
    faded = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR).astype(np.float32) * 0.88 + 14
    rows = np.arange(height, dtype=np.float32)[:, None, None]
    scanlines = 1 - 0.07 * (rows % 3 == 0)
    band_y = (t * 35) % (height + 40) - 20
    tracking = 5 * np.exp(-((rows - band_y) / 7) ** 2)
    columns = np.arange(width, dtype=np.float32)[None, :, None]
    grain = 2 * np.sin(columns * 1.7 + rows * 2.3 + t * 8)
    return np.clip(faded * scanlines + tracking + grain, 0, 255).astype(np.uint8)


# ============================================================
# Cinematic Portal Effects (New)
# ============================================================

def portal_energy_core(frame: np.ndarray, t: float = 0.0,
                       corners: np.ndarray | None = None) -> np.ndarray:
    """Animated energy core with pulsing, swirling patterns."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Normalized distance
    ndist = np.clip(dist / radius, 0, 1)

    # Multiple animated layers
    # Layer 1: Pulsing core
    pulse = 0.5 + 0.5 * np.sin(t * 3)
    core = np.exp(-ndist * 8) * pulse

    # Layer 2: Rotating spiral arms
    spiral = 0.5 + 0.5 * np.sin(angle * 4 - t * 2 + ndist * 10)
    spiral *= np.exp(-ndist * 3)

    # Layer 3: Radial waves
    waves = 0.5 + 0.5 * np.sin(ndist * 20 - t * 4)
    waves *= np.exp(-ndist * 2) * (1 - ndist)

    # Combine layers
    energy = core * 0.5 + spiral * 0.3 + waves * 0.2
    energy = np.clip(energy, 0, 1)

    # Colorize with plasma-like palette
    hue = (angle / (2 * np.pi) + t * 0.1) % 1.0
    sat = 0.8 + 0.2 * energy
    val = energy

    hsv = np.stack([hue * 180, sat * 255, val * 255], axis=2).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Blend with original
    return cv2.addWeighted(frame, 0.1, bgr, 0.9, 0)


def portal_dimensional_rift(frame: np.ndarray, t: float = 0.0,
                            corners: np.ndarray | None = None) -> np.ndarray:
    """Dimensional rift effect with distortion and chromatic aberration."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    # Create distortion map
    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    ndist = np.clip(dist / radius, 0, 1)

    # Radial distortion (barrel/pincushion alternating)
    distortion = 1.0 + 0.3 * np.sin(t * 2) * np.exp(-ndist * 3) * (1 - ndist)
    # Chromatic aberration offset
    ca_offset = 5 * np.sin(t * 5) * ndist

    # Create remap coordinates
    map_x = center[0] + dx * distortion
    map_y = center[1] + dy * distortion

    # Apply to each channel with chromatic aberration
    map_x_r = np.clip(map_x + ca_offset, 0, width - 1).astype(np.float32)
    map_x_g = np.clip(map_x, 0, width - 1).astype(np.float32)
    map_x_b = np.clip(map_x - ca_offset, 0, width - 1).astype(np.float32)
    map_y_f = np.clip(map_y, 0, height - 1).astype(np.float32)

    r = cv2.remap(frame[:, :, 2], map_x_r, map_y_f, cv2.INTER_LINEAR)
    g = cv2.remap(frame[:, :, 1], map_x_g, map_y_f, cv2.INTER_LINEAR)
    b = cv2.remap(frame[:, :, 0], map_x_b, map_y_f, cv2.INTER_LINEAR)

    distorted = np.stack([b, g, r], axis=2)

    # Add energy glow at edges
    edge_glow = np.exp(-(ndist - 0.9)**2 / 0.02) * (0.5 + 0.5 * np.sin(t * 6))
    glow_color = np.array([0, 200, 255], dtype=np.float32)
    for c in range(3):
        distorted[:, :, c] = np.clip(distorted[:, :, c] + edge_glow * glow_color[c], 0, 255)

    return distorted.astype(np.uint8)


def portal_void(frame: np.ndarray, t: float = 0.0,
                corners: np.ndarray | None = None) -> np.ndarray:
    """Dark void with event horizon and accretion disk."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    ndist = np.clip(dist / radius, 0, 1)

    # Event horizon (black center)
    horizon = np.exp(-ndist * 15)

    # Accretion disk (rotating)
    disk_angle = angle * 3 - t * 1.5
    disk = 0.5 + 0.5 * np.sin(disk_angle)
    disk *= np.exp(-(ndist - 0.6)**2 / 0.05)  # Ring at 60% radius

    # Relativistic jets (top/bottom)
    jet = np.exp(-((angle - np.pi/2)**2) / 0.1) * np.exp(-ndist * 2)
    jet += np.exp(-((angle + np.pi/2)**2) / 0.1) * np.exp(-ndist * 2)

    # Hawking radiation (random noise at horizon)
    noise = np.random.random((height, width)) * 0.1
    hawking = noise * np.exp(-(ndist - 1.0)**2 / 0.01)

    # Combine
    intensity = horizon * 0.1 + disk * 0.8 + jet * 0.6 + hawking

    # Color: deep blues and purples for disk, white for jets
    hue = np.full_like(intensity, (0.65 + 0.1 * np.sin(t)) % 1.0)
    sat = np.full_like(intensity, 0.9)
    val = intensity

    hsv = np.stack([hue * 180, sat * 255, val * 255], axis=2).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def portal_fire(frame: np.ndarray, t: float = 0.0,
                corners: np.ndarray | None = None) -> np.ndarray:
    """Fire/plasma portal with realistic flame simulation."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    ndist = np.clip(dist / radius, 0, 1)

    # Flame noise using multiple octaves
    flame = np.zeros((height, width), dtype=np.float32)
    for octave in range(4):
        freq = 2 ** octave
        amp = 1.0 / (2 ** octave)
        # Perlin-like noise using sine waves
        noise = np.sin(x * freq * 0.02 + t * 5) * np.sin(y * freq * 0.02 + t * 3)
        noise += np.sin(x * freq * 0.03 - t * 2) * np.sin(y * freq * 0.03 + t * 4)
        flame += noise * amp

    flame = (flame + 1) * 0.5  # Normalize to 0-1

    # Shape flame to portal
    flame *= np.exp(-ndist * 3) * (1 + 0.5 * np.sin(angle * 5 - t * 4))

    # Rising motion
    rise = np.sin((y / height) * 10 - t * 4) * 0.5 + 0.5
    flame *= rise * (1 - ndist * 0.5)

    # Color gradient: white -> yellow -> orange -> red -> black
    # Map flame intensity to temperature color
    r = np.clip(flame * 1.5, 0, 1)
    g = np.clip(flame * 0.8, 0, 1)
    b = np.clip(flame * 0.2, 0, 1)

    fire_bgr = np.stack([b * 255, g * 255, r * 255], axis=2).astype(np.uint8)

    # Add ember particles
    ember_mask = (flame > 0.8) & (np.random.random((height, width)) < 0.01)
    fire_bgr[ember_mask] = [255, 255, 255]

    return fire_bgr


def portal_ice(frame: np.ndarray, t: float = 0.0,
               corners: np.ndarray | None = None) -> np.ndarray:
    """Ice/crystal portal with fractal patterns."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    ndist = np.clip(dist / radius, 0, 1)

    # Fractal-like crystal pattern
    crystal = np.zeros((height, width), dtype=np.float32)
    for i in range(6):  # 6-fold symmetry
        crystal += np.cos(angle * 6 + i * np.pi / 3 + t * 0.5) * np.exp(-ndist * 2)
    crystal = (crystal + 1) * 0.5

    # Radial crystal growth
    growth = np.sin(ndist * 30 - t * 2) * 0.5 + 0.5
    growth *= np.exp(-ndist * 1.5)

    ice = crystal * 0.6 + growth * 0.4
    ice = np.clip(ice, 0, 1)

    # Ice colors: cyan to white
    r = ice * 0.5
    g = ice * 0.8
    b = ice

    ice_bgr = np.stack([b * 255, g * 255, r * 255], axis=2).astype(np.uint8)

    # Add specular highlights
    highlight = np.exp(-(ndist - 0.3)**2 / 0.02) * (0.5 + 0.5 * np.sin(t * 3))
    for c in range(3):
        ice_bgr[:, :, c] = np.clip(ice_bgr[:, :, c] + highlight * 255, 0, 255)

    return ice_bgr


def portal_matrix(frame: np.ndarray, t: float = 0.0,
                  corners: np.ndarray | None = None) -> np.ndarray:
    """Digital rain / matrix code portal."""
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        return frame.copy()
    if corners is None:
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)

    center = np.mean(corners, axis=0)
    radius = max(1.0, np.mean(np.linalg.norm(corners - center, axis=1)))

    # Create falling characters effect
    char_height = 20
    char_width = 12
    cols = max(1, width // char_width)
    rows = max(1, height // char_height + 2)

    # Generate random "code" columns
    np.random.seed(int(t * 10) % 1000)
    speeds = np.random.uniform(0.5, 2.0, cols)
    offsets = np.random.uniform(0, rows, cols)
    chars = np.random.randint(33, 126, (cols, rows))

    # Build matrix rain
    matrix = np.zeros((height, width), dtype=np.float32)
    for c in range(cols):
        x_pos = c * char_width
        for r in range(rows):
            y_pos = int((r + offsets[c] + t * speeds[c] * 10) % rows) * char_height
            if 0 <= y_pos < height - char_height:
                # Draw character as block
                matrix[y_pos:y_pos+char_height, x_pos:x_pos+char_width] = 1.0

    # Shape to portal
    y, x = np.ogrid[:height, :width]
    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    ndist = np.clip(dist / radius, 0, 1)
    matrix *= np.exp(-ndist * 2)

    # Green color
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 1] = (matrix * 255).astype(np.uint8)  # Green channel

    # Add scanlines
    scanlines = 1 - 0.3 * ((y % 4) == 0)
    bgr = (bgr * scanlines[..., None]).astype(np.uint8)

    return bgr


# ============================================================
# Geometric Effects (Enhanced from original)
# ============================================================

def _polygon(sides: int, radius: float, angle: float, center=None) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, sides + 1) + angle
    pts = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    if center is not None:
        pts += np.array(center)
    return pts


def _geometry_overlay(frame: np.ndarray, paths: list, color: tuple,
                      corners: np.ndarray | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if corners is None:
        corners = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], np.float32)
    source = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(source, np.asarray(corners, np.float32))
    ink = np.zeros((height, width), np.uint8)
    thickness = max(1, min(height, width) // 260)
    for path in paths:
        points = cv2.perspectiveTransform(np.asarray(path, np.float32)[None], transform)
        cv2.polylines(ink, [np.rint(points[0]).astype(np.int32)], False,
                      255, thickness, cv2.LINE_AA)
    glow = cv2.GaussianBlur(ink, (0, 0), 2).astype(np.float32) / 255
    alpha = (ink.astype(np.float32) / 255 * 0.9)[..., None]
    tint = np.array(color, np.float32)
    background = frame.astype(np.float32) * 0.72
    output = background * (1 - alpha) + tint * alpha
    output += glow[..., None] * tint * 0.65
    return np.clip(output, 0, 255).astype(np.uint8)


def mystic_seal(frame: np.ndarray, t: float = 0.0,
                corners: np.ndarray | None = None) -> np.ndarray:
    paths = [_polygon(12, 0.93, t * 0.08), _polygon(12, 0.83, -t * 0.06),
             _polygon(4, 0.76, t * 0.16), _polygon(4, 0.76, -t * 0.12 + np.pi / 4),
             _polygon(6, 0.32, -t * 0.2)]
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False) + t * 0.08:
        direction = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-direction[1], direction[0]])
        paths.append([direction * 0.85, direction * 0.9])
        paths.append([direction * 0.8 - tangent * 0.025,
                      direction * 0.76, direction * 0.8 + tangent * 0.025])
    return _geometry_overlay(frame, paths, (45, 170, 255), corners)


def wireframe_grid(frame: np.ndarray, t: float = 0.0,
                   corners: np.ndarray | None = None) -> np.ndarray:
    paths = []
    for position in np.linspace(-1, 1, 9):
        paths.extend(([[-1, position], [1, position]],
                      [[position, -1], [position, 1]]))
    for position in np.linspace(-1, 1, 5):
        paths.extend(([[-1, position], [-position, 1]],
                      [[position, -1], [1, -position]]))
    for radius in (0.25, 0.5, 0.75):
        paths.append(_polygon(4, radius, np.pi / 4 + t * 0.05))
    sweep = np.sin(t * 0.6) * 0.9
    paths.extend(([[-1, sweep], [1, sweep]], [[sweep, -1], [sweep, 1]]))
    return _geometry_overlay(frame, paths, (255, 215, 80), corners)


def astral_lattice(frame: np.ndarray, t: float = 0.0,
                   corners: np.ndarray | None = None) -> np.ndarray:
    rings = [_polygon(6, radius, t * speed + np.pi / 6)
             for radius, speed in ((0.94, 0.09), (0.64, -0.13), (0.34, 0.18))]
    paths = list(rings)
    for outer, inner in zip(rings, rings[1:]):
        for i in range(6):
            paths.extend(([outer[i], inner[i]], [outer[i], inner[(i + 1) % 6]]))
    paths.append(_polygon(3, 0.22, -t * 0.2))
    return _geometry_overlay(frame, paths, (255, 110, 210), corners)


# New geometric effects
def sacred_geometry(frame: np.ndarray, t: float = 0.0,
                    corners: np.ndarray | None = None) -> np.ndarray:
    """Flower of Life and Metatron's Cube inspired patterns."""
    paths = []
    # Flower of Life - 7 circles
    for i in range(7):
        angle = i * 2 * np.pi / 7
        paths.append(_polygon(64, 0.35, angle + t * 0.02))

    # Central circle
    paths.append(_polygon(64, 0.7, t * 0.01))

    # Metatron's Cube - 13 circles
    for i in range(13):
        angle = i * 2 * np.pi / 13
        r = 0.5 + 0.2 * np.sin(t * 0.5 + i)
        paths.append(_polygon(64, r, angle))

    # Connecting lines (deterministic subset)
    for i in range(13):
        for j in range(i+1, 13):
            # Deterministic: connect every 3rd and 5th
            if (i + j) % 3 == 0 or (i * j) % 5 == 0:
                angle1 = i * 2 * np.pi / 13
                angle2 = j * 2 * np.pi / 13
                paths.append([np.array([np.cos(angle1), np.sin(angle1)]) * 0.5,
                              np.array([np.cos(angle2), np.sin(angle2)]) * 0.5])

    return _geometry_overlay(frame, paths, (180, 230, 255), corners)


def hexagonal_grid(frame: np.ndarray, t: float = 0.0,
                   corners: np.ndarray | None = None) -> np.ndarray:
    """Animated hexagonal grid with pulsing cells."""
    paths = []
    # Hexagonal grid
    spacing = 0.25
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = spacing * (3/2 * q)
            y = spacing * (np.sqrt(3) * (r + q/2))
            if x**2 + y**2 < 1.1:
                paths.append(_polygon(6, 0.12, t * 0.03, center=[x, y]))

    # Pulsing hexagon in center
    pulse = 0.3 + 0.2 * np.sin(t * 2)
    paths.append(_polygon(6, pulse, t * 0.1))

    return _geometry_overlay(frame, paths, (100, 255, 180), corners)




# Color mapping for geometric effects
GEOMETRIC_COLORS = {
    mystic_seal: (45, 170, 255),
    wireframe_grid: (255, 215, 80),
    astral_lattice: (255, 110, 210),
    sacred_geometry: (180, 230, 255),
    hexagonal_grid: (100, 255, 180),
}

# All effects
EFFECTS = (
    ("Thermal", thermal),
    ("Pencil Sketch", pencil_sketch),
    ("RGB Glitch", rgb_glitch),
    ("Monochrome", monochrome),
    ("Dual-Tone", dual_tone),
    ("Color Inversion", inversion),
    ("Sepia", sepia),
    ("Blur", blur),
    ("Cartoon", cartoon),
    ("Neon Edges", neon_edges),
    ("Pixel Art", pixel_art),
    ("Comic Halftone", comic_halftone),
    ("VHS", vhs),
    # Cinematic portal effects
    ("Energy Core", portal_energy_core),
    ("Dimensional Rift", portal_dimensional_rift),
    ("Void", portal_void),
    ("Fire", portal_fire),
    ("Ice", portal_ice),
    ("Matrix", portal_matrix),
    # Geometric
    ("Mystic Seal", mystic_seal),
    ("Wireframe Grid", wireframe_grid),
    ("Astral Lattice", astral_lattice),
    ("Sacred Geometry", sacred_geometry),
    ("Hexagonal Grid", hexagonal_grid),
)