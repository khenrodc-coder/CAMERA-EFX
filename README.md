# Hand Portal FX

A cinematic, real-time hand-tracking visual effects application. Control a science-fiction energy portal using nothing but your hands in front of a webcam.

![Hand Portal FX](docs/demo.gif)

## Features

### Hand Tracking
- **Real-time 21-landmark detection** via MediaPipe Hand Landmarker
- **Left/Right hand identification** with handedness classification
- **Finger state tracking** (extended/folded) for all 5 fingers
- **Palm orientation & normal** computation
- **Temporal smoothing** with configurable EMA for jitter-free tracking
- **Two-hand simultaneous tracking**

### Gesture Recognition
| Gesture | Hands | Description |
|---------|-------|-------------|
| Pinch | 1 | Thumb + index fingertip touch - cycle effects |
| Open Palm | 1 | All fingers extended |
| Closed Fist | 1 | All fingers folded |
| Peace Sign | 1 | Index + middle extended |
| Thumbs Up | 1 | Only thumb extended |
| Pointing | 1 | Only index extended |
| Two Finger | 1 | Index + middle extended |
| **Both Hands Open** | 2 | Activate portal |
| **Hands Expanding** | 2 | Move hands apart - enlarge portal |
| **Hands Shrinking** | 2 | Move hands together - shrink portal |
| **Hands Rotating** | 2 | Rotate both hands - rotate portal |
| **Both Hands Pinch** | 2 | Charge portal energy |
| **Hands Moving Up** | 2 | Increase intensity / shift color |
| **Hands Moving Down** | 2 | Decrease intensity / shift color |

### Cinematic Portal Effects
- **Energy Core** - Pulsing, swirling plasma core
- **Dimensional Rift** - Chromatic aberration distortion
- **Void** - Black hole with accretion disk
- **Fire** - Realistic flame simulation
- **Ice** - Crystalline fractal patterns
- **Matrix** - Digital rain code
- **Mystic Seal** - Rotating geometric rings
- **Wireframe Grid** - Animated cyan lattice
- **Astral Lattice** - Connected hexagonal rings
- **Sacred Geometry** - Flower of Life patterns
- **Hexagonal Grid** - Pulsing honeycomb

Plus 8 classic image filters: Thermal, Pencil Sketch, RGB Glitch, Monochrome, Dual-Tone, Inversion, Sepia, Blur, Cartoon, Neon Edges, Pixel Art, Comic Halftone, VHS.

### Visual Polish
- Feathered portal edges with Gaussian blur
- Multi-layer additive glow rendering
- Animated concentric energy rings
- Electric lightning arcs
- Orbiting and radial emission particles
- Expanding shockwave on energy release
- Dynamic color shifting (HSV hue wheel)
- Configurable intensity, glow, particle count

### Camera System
- Automatic camera detection with `--list-cameras`
- Manual selection via `--camera INDEX` or config
- Configurable resolution and FPS
- Cross-platform (Windows DirectShow, macOS AVFoundation, Linux V4L2)
- Mirror mode toggle
- Graceful error handling and recovery

### User Interface
- Modern semi-transparent overlay panels
- Real-time FPS counter
- Gesture recognition status display
- Portal state indicator (closed/opening/open/charging/releasing/closing)
- Energy level meter
- Interactive settings panel with sliders:
  - Portal intensity
  - Glow intensity
  - Particle density
  - Portal size
  - Smoothing factor
  - Color hue
- Gesture debug panel
- Keyboard shortcuts for all major functions
- Mouse-adjustable sliders
- Settings persistence (JSON)

### Performance
- Three presets: Low (640×480), Medium (1280×720), High (1920×1080)
- Adaptive detection resolution
- Optimized particle rendering
- Single-threaded synchronous loop for minimal latency
- Configurable target FPS
- Graceful degradation under load

## Installation

### Requirements
- Python 3.10+
- Windows 10/11, macOS 13+, or Linux
- Webcam

### Setup

```bash
# Clone and enter directory
cd Hand-Portal-FX

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe hand model (~8 MB)
python download_model.py

# Run
python app.py
```

## Usage

### Quick Start
```bash
python app.py --fps
```

### Command Line Options
```bash
python app.py --help

Options:
  --camera INDEX        Camera index (default: 0)
  --list-cameras        List available cameras and exit
  --low-res             Request 640×480 capture
  --fps                 Show FPS counter
  --frames N            Exit after N frames (testing)
  --preset PRESET       Performance preset: low, medium, high
```

### Controls

#### Gestures
- **Pinch (thumb + index)** on either hand → Next effect
- **Both hands open** (hold 0.5s) → Activate portal
- **Hands apart/together** → Expand/shrink portal
- **Rotate both hands** → Rotate portal
- **Both hands pinch** → Charge energy
- **Release pinch** → Release energy shockwave
- **Hands up/down** → Increase/decrease intensity & shift color
- **Closed fist** (on open portal) → Grab & drag portal
- **Open palm** (while grabbed) → Release portal

#### Keyboard
| Key | Action |
|-----|--------|
| `N` / `P` | Next / Previous effect |
| `H` | Toggle help overlay |
| `S` | Toggle settings panel |
| `G` | Toggle gesture debug |
| `L` | Toggle landmark display |
| `F` | Toggle FPS display |
| `M` | Toggle mirror mode |
| `R` | Reset portal, gestures, tracker |
| `C` | Cycle camera |
| `Q` / `Esc` | Quit |

#### Mouse
- Click & drag sliders in settings panel
- Click buttons in settings panel

### Configuration

Settings are persisted to `settings.json`:
```json
{
  "camera_index": 0,
  "effect_index": 0,
  "show_fps": true,
  "show_landmarks": false,
  "mirror": true,
  "preset": "medium"
}
```

Runtime config in `config.py`:
```python
# Tracking
HAND_DETECTION_CONFIDENCE = 0.7
LANDMARK_SMOOTHING_ALPHA = 0.35
FINGER_EXTENSION_RATIO = 1.1

# Portal
PORTAL_SMOOTHING = 0.45
PARTICLE_COUNT = 48
GLOW_INTENSITY = 1.0
ENERGY_RING_COUNT = 3

# Gestures
GESTURE_DWELL = 0.08
GESTURE_COOLDOWN = 0.35
PINCH_CLOSE_RATIO = 0.23
```

## Architecture

```
hand_tracking.py   → MediaPipe wrapper, finger states, palm geometry
gestures.py        → GestureRecognizer, PinchState, 13 gesture types
portal.py          → PortalState (tracking), cinematic rendering
filters.py         → 20 effects (6 cinematic + 5 geometric + 9 classic)
app.py             → Main loop, camera, UI, input handling
config.py          → All tunable parameters
download_model.py  → Fetches verified MediaPipe model
```

## Testing

```bash
# Run all tests
python -m unittest discover -s tests -v

# Specific test modules
python -m unittest tests.test_geometry -v
python -m unittest tests.test_filters -v
python -m unittest tests.test_gestures -v
```

## Performance Tips

1. Use `--low-res` or `--preset low` on slower machines
2. Close other camera applications
3. Ensure good lighting for hand tracking
4. Keep hands within camera frame
5. Avoid extreme backlighting

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Run `--list-cameras`, try `--camera 1` |
| Permission denied | Grant camera access in OS settings |
| Low FPS | Lower preset, reduce particle count |
| Jittery tracking | Increase smoothing, improve lighting |
| Gestures not recognized | Adjust thresholds in config.py |

## Credits

- MediaPipe for hand landmark detection
- OpenCV for computer vision pipeline
- Inspired by [Doctor Strange portal effects](https://youtube.com/shorts/XqYy4Qovh5c)

## License

MIT License - see LICENSE file.

---

**Built with Python, OpenCV, MediaPipe, NumPy**