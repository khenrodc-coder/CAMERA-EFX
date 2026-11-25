"""Hand Portal FX - Cinematic hand-controlled portal with advanced gesture recognition."""

import argparse
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

import config
from filters import EFFECTS
from gestures import GestureRecognizer, GestureType, PinchState
from hand_tracking import HandTracker
from portal import PortalState, render_portal, cinematic_start_charge, cinematic_release_energy, cinematic_update_charge, cinematic_grab, cinematic_release_grab, cinematic_move_grabbed, cinematic_scale_portal, cinematic_rotate_portal, cinematic_shift_color, cinematic_set_intensity, cinematic_reset, get_cinematic_state, cinematic_trigger_vanish, cinematic_trigger_reappear

WINDOW = "Hand Portal FX"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

CAMERA_HELP = (
    "Check camera permissions and ensure no other app is using the camera. "
    "Use --camera INDEX to select another camera. "
    "Use --list-cameras to see available cameras."
)


def list_cameras(max_test: int = 10) -> list[tuple[int, tuple[int, int]]]:
    """List available cameras with their resolutions."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, (width, height)))
            cap.release()
    return available


def open_camera(index: int, width: int, height: int, fps: int = 30):
    """Open camera with specified parameters."""
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {index}. {CAMERA_HELP}")

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

        # Try to set auto-exposure and focus
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure

    except Exception:
        camera.release()
        raise
    return camera


def draw_text(frame, text, position, scale=0.5, color=(235, 235, 235), thickness=1):
    """Draw outlined text."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (12, 12, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.7, color=(20, 20, 30)):
    """Draw semi-transparent panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_slider(frame, x, y, w, value, label, min_val=0.0, max_val=1.0,
                color=(100, 230, 140), bg_color=(40, 40, 50)):
    """Draw a slider control."""
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + 20), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 20), (80, 80, 90), 1)

    # Fill
    fill_w = int(w * (value - min_val) / (max_val - min_val))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + 20), color, -1)

    # Label
    draw_text(frame, f"{label}: {value:.2f}", (x, y - 5), 0.4, (200, 200, 200))

    return (x, y, x + w, y + 20)  # Return bounds for hit testing


def draw_button(frame, x, y, w, h, label, active=False,
                color=(60, 60, 80), active_color=(100, 200, 120),
                text_color=(230, 230, 230)):
    """Draw a button."""
    btn_color = active_color if active else color
    cv2.rectangle(frame, (x, y), (x + w, y + h), btn_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 120), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = x + (w - text_size[0]) // 2
    ty = y + (h + text_size[1]) // 2
    draw_text(frame, label, (tx, ty), 0.5, text_color)

    return (x, y, x + w, y + h)


class UIManager:
    """Manages the on-screen UI overlay."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.show_help = True
        self.show_settings = False
        self.show_gesture_debug = config.SHOW_GESTURE_DEBUG
        self.drag_slider = None
        self.slider_bounds = {}
        self.button_bounds = {}

    def update_layout(self, width: int, height: int):
        self.width = width
        self.height = height

    def draw(self, frame, portal, gesture_recognizer, effect_index, effect_name,
             fps, camera_index, gesture_state=None):
        """Draw complete UI overlay."""
        self.slider_bounds = {}
        self.button_bounds = {}

        # Top bar - effect info
        self._draw_top_bar(frame, effect_index, effect_name, fps)

        # Bottom bar - status and controls
        self._draw_bottom_bar(frame, portal, gesture_state, camera_index)

        # Right panel - settings (if open)
        if self.show_settings:
            self._draw_settings_panel(frame, portal)

        # Left panel - gesture debug (if enabled)
        if self.show_gesture_debug and gesture_state:
            self._draw_gesture_debug(frame, gesture_state)

        # Help text
        if self.show_help:
            self._draw_help(frame)

    def _draw_top_bar(self, frame, effect_index, effect_name, fps):
        """Draw top info bar."""
        bar_h = 40
        draw_panel(frame, 0, 0, self.width, bar_h, 0.8)

        # Effect indicator
        draw_text(frame, f"Effect: {effect_index + 1:02d}/{len(EFFECTS):02d}  {effect_name}",
                  (15, 28), 0.65, (255, 220, 160))

        # FPS
        if config.SHOW_FPS:
            draw_text(frame, f"{fps:.0f} FPS", (self.width - 100, 28), 0.55, (180, 255, 180))

    def _draw_bottom_bar(self, frame, portal, gesture_state, camera_index):
        """Draw bottom status bar."""
        bar_h = 60
        y = self.height - bar_h
        draw_panel(frame, 0, y, self.width, bar_h, 0.8)

        # Get cinematic state
        from portal import get_cinematic_state
        cinematic_state = get_cinematic_state()

        # Portal status
        status_text = f"Portal: {cinematic_state['state'].upper()}"
        status_color = {
            'closed': (100, 100, 100),
            'opening': (255, 200, 100),
            'open': (100, 255, 100),
            'charging': (100, 200, 255),
            'releasing': (255, 150, 255),
            'closing': (255, 100, 100),
        }.get(cinematic_state['state'], (200, 200, 200))
        draw_text(frame, status_text, (15, y + 20), 0.5, status_color)

        # Energy level
        if cinematic_state['state'] in ('charging', 'open'):
            energy_text = f"Energy: {cinematic_state['energy_level']:.0%}"
            draw_text(frame, energy_text, (15, y + 42), 0.4, (100, 200, 255))

        # Vanish state
        vanish_state = cinematic_state.get('vanish_state', 'visible')
        if vanish_state != 'visible':
            vanish_text = f"Vanish: {vanish_state.upper()}"
            if vanish_state == 'vanished':
                vanish_text += " - SNAP to reappear"
            draw_text(frame, vanish_text, (15, y + 42), 0.4, (255, 150, 255))

        # Gesture status
        if gesture_state and gesture_state.gesture != GestureType.NONE:
            gesture_text = f"Gesture: {gesture_state.gesture.name.replace('_', ' ')} ({gesture_state.confidence:.0%})"
            draw_text(frame, gesture_text, (250, y + 20), 0.45, (255, 220, 160))

        # Camera info
        cam_text = f"Cam: {camera_index}  [{'Mirror' if config.MIRROR_PREVIEW else 'Normal'}]"
        draw_text(frame, cam_text, (self.width - 200, y + 20), 0.4, (180, 180, 180))

        # Controls hint
        hint = "H:Help  S:Settings  G:Gesture Debug  R:Reset  Q:Quit"
        draw_text(frame, hint, (self.width - 480, y + 42), 0.35, (150, 150, 150))

    def _draw_settings_panel(self, frame, portal):
        """Draw settings panel on the right."""
        panel_w = 280
        x = self.width - panel_w - 10
        y = 50
        h = self.height - 100

        draw_panel(frame, x, y, panel_w, h, 0.9, (15, 15, 25))

        draw_text(frame, "SETTINGS", (x + 15, y + 25), 0.6, (255, 220, 160))

        row_y = y + 50
        spacing = 35

        # Portal intensity
        self.slider_bounds['intensity'] = draw_slider(
            frame, x + 15, row_y, 250, portal.intensity, "Intensity", 0.1, 2.0)
        row_y += spacing

        # Glow intensity
        self.slider_bounds['glow'] = draw_slider(
            frame, x + 15, row_y, 250, config.GLOW_INTENSITY, "Glow", 0.0, 2.0)
        row_y += spacing

        # Particle count
        self.slider_bounds['particles'] = draw_slider(
            frame, x + 15, row_y, 250, config.PARTICLE_COUNT / 100, "Particles", 0.0, 1.0)
        row_y += spacing

        # Portal size
        self.slider_bounds['size'] = draw_slider(
            frame, x + 15, row_y, 250, portal.scale, "Size", 0.3, 3.0)
        row_y += spacing

        # Smoothing
        self.slider_bounds['smoothing'] = draw_slider(
            frame, x + 15, row_y, 250, config.PORTAL_SMOOTHING, "Smoothing", 0.0, 1.0)
        row_y += spacing

        # Color hue
        self.slider_bounds['hue'] = draw_slider(
            frame, x + 15, row_y, 250, config.CURRENT_PORTAL_HUE, "Color Hue", 0.0, 1.0)
        row_y += spacing

        # Buttons
        btn_y = row_y + 10
        self.button_bounds['reset'] = draw_button(frame, x + 15, btn_y, 120, 30, "Reset All")
        self.button_bounds['preset_low'] = draw_button(frame, x + 145, btn_y, 60, 30, "Low")
        self.button_bounds['preset_med'] = draw_button(frame, x + 210, btn_y, 60, 30, "Med")

    def _draw_gesture_debug(self, frame, gesture_state):
        """Draw gesture recognition debug info."""
        panel_w = 250
        x = 10
        y = 50
        h = 200

        draw_panel(frame, x, y, panel_w, h, 0.85, (15, 15, 25))
        draw_text(frame, "GESTURE DEBUG", (x + 10, y + 22), 0.5, (255, 220, 160))

        lines = [
            f"Current: {gesture_state.gesture.name}",
            f"Confidence: {gesture_state.confidence:.2f}",
            f"Hand: {gesture_state.handedness}",
            f"Duration: {gesture_state.duration:.2f}s",
        ]
        for i, line in enumerate(lines):
            draw_text(frame, line, (x + 10, y + 45 + i * 20), 0.4, (200, 200, 200))

    def _draw_help(self, frame):
        """Draw help overlay."""
        lines = [
            "GESTURES:",
            "  Pinch (thumb+index) - Next effect",
            "  Both hands open - Activate portal",
            "  Hands apart/together - Expand/shrink portal",
            "  Rotate hands - Rotate portal",
            "  Pinch both hands - Charge energy",
            "  Release pinch - Release energy",
            "  Hands up/down - Intensity up/down",
            "  Snap (thumb+middle) - Vanish/Reappear",
            "",
            "KEYBOARD:",
            "  N/P - Next/Previous effect",
            "  H - Toggle help",
            "  S - Toggle settings",
            "  G - Toggle gesture debug",
            "  L - Toggle landmarks",
            "  F - Toggle FPS",
            "  M - Toggle mirror",
            "  R - Reset portal & tracker",
            "  V - Vanish/Reappear",
            "  C - Cycle camera",
            "  Q/ESC - Quit",
        ]

        x = 15
        y = self.height - 20 - len(lines) * 18
        draw_panel(frame, x - 5, y - 5, 300, len(lines) * 18 + 10, 0.85, (10, 10, 20))

        for i, line in enumerate(lines):
            color = (255, 220, 160) if line.endswith(':') else (200, 200, 200)
            draw_text(frame, line, (x, y + i * 18), 0.38, color)

    def handle_mouse(self, event, x, y, flags, param):
        """Handle mouse events for UI interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check sliders
            for name, (x1, y1, x2, y2) in self.slider_bounds.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.drag_slider = name
                    return True

            # Check buttons
            for name, (x1, y1, x2, y2) in self.button_bounds.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return ('button', name)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_slider:
            return ('slider_drag', self.drag_slider, x)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_slider:
                self.drag_slider = None
                return ('slider_release',)

        return False

    def get_slider_value(self, x, bounds, min_val=0.0, max_val=1.0):
        """Calculate slider value from mouse position."""
        x1, y1, x2, y2 = bounds
        rel = np.clip((x - x1) / (x2 - x1), 0, 1)
        return min_val + rel * (max_val - min_val)


def load_settings():
    """Load settings from file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(settings):
    """Save settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Failed to save settings: {e}")


def apply_preset(preset_name: str):
    """Apply a performance preset."""
    if preset_name in config.PERFORMANCE_PRESETS:
        preset = config.PERFORMANCE_PRESETS[preset_name]
        for key, value in preset.items():
            setattr(config, key, value)
        config.CURRENT_PRESET = preset_name
        print(f"Applied preset: {preset_name}")


def run(args) -> None:
    """Main application loop."""
    # Handle args for backward compatibility with tests
    if not hasattr(args, 'fps'):
        args.fps = False
    if not hasattr(args, 'low_res'):
        args.low_res = False
    if not hasattr(args, 'frames'):
        args.frames = 0

    # Load settings
    settings = load_settings()
    camera_index = settings.get('camera_index', config.CAMERA_INDEX)
    effect_index = settings.get('effect_index', 0)
    show_fps = settings.get('show_fps', config.SHOW_FPS)
    show_landmarks = settings.get('show_landmarks', config.SHOW_LANDMARKS)
    mirror = settings.get('mirror', config.MIRROR_PREVIEW)
    preset = settings.get('preset', config.CURRENT_PRESET)

    # Apply preset
    apply_preset(preset)

    # Override from command line
    if args.camera is not None:
        camera_index = args.camera
    if args.low_res:
        config.FRAME_WIDTH, config.FRAME_HEIGHT = 640, 480
    if args.fps:
        show_fps = True

    # Initialize
    camera = None
    tracker = None
    portal = PortalState()
    gesture_recognizer = GestureRecognizer()
    pinch_switch = PinchState()  # Legacy compatibility

    try:
        width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT
        camera = open_camera(camera_index, width, height, config.TARGET_FPS)
        tracker = HandTracker()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)

        ui = UIManager(width, height)
        try:
            cv2.setMouseCallback(WINDOW, ui.handle_mouse)
        except cv2.error:
            # Window not available (e.g., in tests), skip mouse callback
            pass

        changed_until = time.monotonic() + 2
        feedback, feedback_until = "", 0.0
        fps, previous = 0.0, time.monotonic()
        failures, frame_count = 0, 0
        reported_size = False

        # For gesture-based portal control
        last_gesture_time = 0
        gesture_cooldown = 0.3

        while True:
            ok, frame = camera.read()
            now = time.monotonic()

            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= 30:
                    raise RuntimeError(f"Camera stopped returning frames. {CAMERA_HELP}")
                if cv2.waitKey(20) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            failures = 0

            if not reported_size:
                actual_h, actual_w = frame.shape[:2]
                print(f"Camera {camera_index}: {actual_w} x {actual_h}; press Q or Esc to quit.")
                if (actual_w, actual_h) != (width, height):
                    print(f"Requested {width} x {height}; fitting frames within that size.")
                reported_size = True

            # Scale frame to target size
            scale = min(1.0, width / frame.shape[1], height / frame.shape[0])
            if scale < 1:
                frame = cv2.resize(frame, (round(frame.shape[1] * scale),
                                          round(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)

            if mirror:
                frame = cv2.flip(frame, 1)

            # Detect hands
            hands = tracker.detect(frame, now)

            # Update gestures
            gesture_state = gesture_recognizer.update(hands, now)

            # Legacy pinch for effect switching
            if pinch_switch.update([hand.points for hand in hands], now):
                effect_index = (effect_index + 1) % len(EFFECTS)
                changed_until = feedback_until = now + 1.5
                feedback = f"Effect: {EFFECTS[effect_index][0]}"

            # Portal control points (thumb + index of both hands)
            points = None
            if len(hands) == 2 and not pinch_switch.pinching:
                points = np.concatenate([hand.points[[4, 8]] for hand in hands])
                points = np.clip(points, [0, 0],
                               [frame.shape[1] - 1, frame.shape[0] - 1])

            # Handle advanced gestures for portal control
            if gesture_state and gesture_state.gesture != GestureType.NONE:
                if now - last_gesture_time > gesture_cooldown:
                    # Handle snap gesture immediately with frame
                    if gesture_state.gesture == GestureType.SNAP:
                        cinematic_state = get_cinematic_state()
                        if cinematic_state['vanish_state'] == 'visible':
                            cinematic_trigger_vanish(frame, now)
                        elif cinematic_state['vanish_state'] == 'vanished':
                            cinematic_trigger_reappear(now)
                    else:
                        handle_gesture_action(gesture_state, portal, hands, now)
                    last_gesture_time = now

            # Update portal
            corners, opacity = portal.update(points, now)

            # Update cinematic charge level
            cinematic_update_charge(now)

            # Handle grab movement
            cinematic_state = get_cinematic_state()
            if cinematic_state['grabbed'] and cinematic_state['grab_offset'] is not None and portal.corners is not None:
                # This would be handled by gesture if we had continuous hand position
                pass

            # Get current effect
            effect_name, effect = EFFECTS[effect_index]

            # Render
            output = frame
            if corners is not None and opacity > 0:
                output = render_portal(frame, corners, opacity, effect, now)

            # Draw landmarks
            if show_landmarks:
                for hand in hands:
                    for point in hand.points:
                        cv2.circle(output, tuple(point.astype(int)), 2, (100, 230, 140), -1)

            # Calculate FPS
            elapsed = now - previous
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed
            previous = now

            # Draw UI
            ui.draw(output, portal, gesture_recognizer, effect_index, effect_name,
                   fps, camera_index, gesture_state)

            # Feedback messages
            if now < feedback_until:
                draw_text(output, feedback, (20, 60))
            elif pinch_switch.pinching:
                message = ("Separate fingertips to re-arm" if pinch_switch.latched
                           else "Pinch thumb + index to switch")
                draw_text(output, message, (20, 60))
                if pinch_switch.progress > 0:
                    cv2.line(output, (20, 70),
                             (20 + int(180 * pinch_switch.progress), 70),
                             (255, 220, 160), 3, cv2.LINE_AA)

            cv2.imshow(WINDOW, output)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            # Effect switching
            if key in (ord("n"), ord("N")):
                effect_index = (effect_index + 1) % len(EFFECTS)
                changed_until = now + 1.5
            elif key in (ord("p"), ord("P")):
                effect_index = (effect_index - 1) % len(EFFECTS)
                changed_until = now + 1.5

            # UI toggles
            elif key in (ord("h"), ord("H")):
                ui.show_help = not ui.show_help
            elif key in (ord("s"), ord("S")):
                ui.show_settings = not ui.show_settings
            elif key in (ord("g"), ord("G")):
                ui.show_gesture_debug = not ui.show_gesture_debug
                config.SHOW_GESTURE_DEBUG = ui.show_gesture_debug
            elif key in (ord("l"), ord("L")):
                show_landmarks = not show_landmarks
            elif key in (ord("f"), ord("F")):
                show_fps = not show_fps
                config.SHOW_FPS = show_fps
            elif key in (ord("m"), ord("M")):
                mirror = not mirror
                config.MIRROR_PREVIEW = mirror
            elif key in (ord("r"), ord("R")):
                portal.reset()
                cinematic_reset()
                gesture_recognizer.reset()
                pinch_switch.reset()
                tracker.close()
                tracker = HandTracker()
                feedback, feedback_until = "Tracking reset", now + 1.5
            elif key in (ord("v"), ord("V")):
                # Keyboard trigger for vanish/reappear (testing)
                cinematic_state = get_cinematic_state()
                if cinematic_state['vanish_state'] == 'visible':
                    cinematic_trigger_vanish(frame, now)
                    feedback, feedback_until = "Vanish triggered", now + 1.5
                elif cinematic_state['vanish_state'] == 'vanished':
                    cinematic_trigger_reappear(now)
                    feedback, feedback_until = "Reappear triggered", now + 1.5
            elif key in (ord("c"), ord("C")):
                # Cycle camera
                cameras = list_cameras()
                if cameras:
                    current_idx = next((i for i, (idx, _) in enumerate(cameras) if idx == camera_index), 0)
                    next_idx = (current_idx + 1) % len(cameras)
                    camera_index = cameras[next_idx][0]
                    camera.release()
                    camera = open_camera(camera_index, width, height, config.TARGET_FPS)
                    feedback, feedback_until = f"Camera: {camera_index}", now + 1.5

            frame_count += 1
            if args.frames and frame_count >= args.frames:
                break

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Handle mouse UI interaction
            # (Mouse callback handles this asynchronously)

    finally:
        # Save settings
        settings = {
            'camera_index': camera_index,
            'effect_index': effect_index,
            'show_fps': show_fps,
            'show_landmarks': show_landmarks,
            'mirror': mirror,
            'preset': config.CURRENT_PRESET,
        }
        save_settings(settings)

        if camera is not None:
            camera.release()
        try:
            if tracker is not None:
                tracker.close()
        finally:
            cv2.destroyAllWindows()


def handle_gesture_action(gesture_state, portal, hands, now):
    """Handle portal actions based on recognized gestures."""
    g = gesture_state.gesture
    cinematic_state = get_cinematic_state()

    if g == GestureType.BOTH_HANDS_OPEN:
        # Activate portal (already handled by portal state machine)
        pass

    elif g == GestureType.HANDS_EXPANDING:
        # Scale portal
        if cinematic_state['state'] in ('open', 'charging') and portal.corners is not None:
            center = np.mean(portal.corners, axis=0)
            portal.corners = cinematic_scale_portal(1.1, center, portal.corners)

    elif g == GestureType.HANDS_SHRINKING:
        if cinematic_state['state'] in ('open', 'charging') and portal.corners is not None:
            center = np.mean(portal.corners, axis=0)
            portal.corners = cinematic_scale_portal(0.9, center, portal.corners)

    elif g == GestureType.HANDS_ROTATING_CW:
        if cinematic_state['state'] in ('open', 'charging') and portal.corners is not None:
            center = np.mean(portal.corners, axis=0)
            portal.corners = cinematic_rotate_portal(0.1, center, portal.corners)

    elif g == GestureType.HANDS_ROTATING_CCW:
        if cinematic_state['state'] in ('open', 'charging') and portal.corners is not None:
            center = np.mean(portal.corners, axis=0)
            portal.corners = cinematic_rotate_portal(-0.1, center, portal.corners)

    elif g == GestureType.BOTH_HANDS_PINCH:
        if not cinematic_state['charging']:
            cinematic_start_charge(now)

    elif g == GestureType.HANDS_MOVING_UP:
        cinematic_set_intensity(cinematic_state['intensity'] + 0.1)
        cinematic_shift_color(0.02)

    elif g == GestureType.HANDS_MOVING_DOWN:
        cinematic_set_intensity(cinematic_state['intensity'] - 0.1)
        cinematic_shift_color(-0.02)

    elif g == GestureType.OPEN_PALM and cinematic_state['grabbed']:
        # Release grab
        cinematic_release_grab()

    elif g == GestureType.CLOSED_FIST and cinematic_state['state'] == "open" and not cinematic_state['grabbed']:
        # Grab portal with fist
        if hands:
            hand = hands[0]
            cinematic_grab(hand.handedness, hand.palm_center, now)
            # Move portal to follow hand
            new_center = cinematic_move_grabbed(hand.palm_center, portal.corners)
            if new_center is not None and portal.corners is not None:
                center = np.mean(portal.corners, axis=0)
                delta = new_center - center
                portal.corners += delta

    elif g == GestureType.SNAP:
        # Finger snap - toggle vanish/reappear
        cinematic_state = get_cinematic_state()
        if cinematic_state['vanish_state'] == 'visible':
            # Need the current frame for vanish effect
            # We'll trigger this in the main loop where we have the frame
            pass
        elif cinematic_state['vanish_state'] == 'vanished':
            cinematic_trigger_reappear(now)

    # Update portal color from cinematic state
    cinematic_state = get_cinematic_state()
    config.CURRENT_PORTAL_HUE = cinematic_state['color_hue']


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand Portal FX - Cinematic hand-controlled portal")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    parser.add_argument("--low-res", action="store_true", help="Request 640 x 480 capture")
    parser.add_argument("--fps", action="store_true", help="Show measured loop FPS")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames (startup check)")
    parser.add_argument("--preset", choices=["low", "medium", "high"], help="Performance preset")
    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras()
        print("Available cameras:")
        for idx, (w, h) in cameras:
            print(f"  {idx}: {w}x{h}")
        return 0

    if args.frames < 0 or (args.camera is not None and args.camera < 0):
        parser.error("--frames and --camera must be nonnegative")

    if args.preset:
        apply_preset(args.preset)

    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Hand Portal FX: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())