# HandSurfers

HandSurfers enables real-time webcam gesture control for keyboard-driven games (such as Subway Surfers) or any application controlled with arrow keys (`Left`, `Right`, `Up`, `Down`).

The repository provides two independent implementations:
1. **Python + MediaPipe**: Machine-learning hand landmark tracking with three gesture detection variants.
2. **C++ + OpenCV**: Fast skin-color segmentation in the YCrCb color space without neural network dependencies.

---

## Control Modes & Implementations

### 1. Python Scripts (MediaPipe Hand Landmarker)

All Python implementations use Google MediaPipe's `HandLandmarker` model (`hand_landmarker.task`). If the model is not found in the project root, the script automatically downloads it on first run (~8 MB). Keyboard inputs are sent via the native Windows `SendInput` API (`ctypes.windll.user32`).

#### A. Palm Tracking (`hand_surfers.py`)
- **Tracking**: Computes the palm center by averaging the wrist landmark (`0`) and the four knuckle MCP landmarks (`5`, `9`, `13`, `17`).
- **Mechanism**: A neutral bounding box sits at the center of the screen.
  - Moving the palm outside the box triggers the arrow key corresponding to the largest displacement:
    - **Left / Right**: Move hand left or right
    - **Up (Jump)**: Move hand above the neutral box
    - **Down (Roll)**: Move hand below the neutral box
  - The palm must return inside the neutral box before another gesture can be triggered.
- **Parameters**: Cooldown = `0.30s`, Smoothing = `0.5`, Neutral box = `20%` width × `24%` height.

#### B. Fingertip Tracking (`hand_surfers_finger.py`)
- **Tracking**: Identifies and tracks the highest extended fingertip among index (`8`), middle (`12`), ring (`16`), and pinky (`20`).
- **Mechanism**: Similar neutral box approach as `hand_surfers.py`, but tailored for fingertip pointing with lower latency.
- **Optimizations**:
  - Tighter neutral box (`10%` width × `12%` height).
  - Faster cooldown (`0.18s`) and higher responsiveness smoothing (`0.8`).
  - Webcam capture configured to MJPG at 30 FPS (`CAP_PROP_BUFFERSIZE = 1`) to eliminate frame queue latency in low light.
  - Real-time on-screen FPS display.

#### C. Dynamic Swipe / Flick Detection (`hand_surfers_swipe.py`)
- **Tracking**: Tracks the highest extended fingertip using a rolling temporal history window (`SwipeDetector`, 0.25 seconds).
- **Mechanism**: **No neutral box required.**
  - Detects rapid flicks/swipes exceeding distance (`10%` of frame) and velocity thresholds (`8%` of frame within 0.1s).
  - **Pause-to-Rearm Protection**: After firing a move, the fingertip must remain still for at least `0.12s` (`STILL_DIST < 0.03`) before the detector rearms. This prevents triggering an accidental opposite action when returning the hand.
- **Visual Feedback**:
  - **Green circle**: Armed and ready for a flick.
  - **Red circle**: Paused / cooling down.

---

### 2. C++ Implementation (`main.cpp`)

A standalone C++17 application using OpenCV for computer vision and OS-level keyboard events.

- **Tracking**:
  - Converts frames to the **YCrCb** color space and segments skin tones.
  - Constrains detection to a Region of Interest (ROI) spanning the right ~55% of the frame to isolate the hand and ignore the user's face on the left.
  - Applies Gaussian blur and morphological opening/closing to filter noise, extracting the largest contour centroid (minimum area threshold: 3000).
- **Interactive Calibration**:
  - Starts uncalibrated. Place hand over the central green calibration box and press `c`. The system samples the mean YCrCb values and computes detection thresholds.
  - Skin tolerance can be adjusted live using `-` and `+`/`=`.
- **Gesture Firing**:
  - Bounded by a neutral box centered inside the ROI.
  - Moving the hand centroid outside triggers the dominant directional key (`Left`, `Right`, `Up`, `Down`).
  - Re-arms when the hand returns inside the neutral box. Cooldown: `350ms`.
- **OS Support**:
  - Windows: Uses `SendInput` (`VK_LEFT`, `VK_RIGHT`, `VK_UP`, `VK_DOWN`).
  - macOS: Uses CoreGraphics `CGEventCreateKeyboardEvent` / `CGEventPost`.
  - Linux (X11): Uses `xdotool key <name>`.

---

## Keyboard Hotkeys (Preview Window)

| Key | Mode | Action |
| :--- | :--- | :--- |
| `k` | All | Toggle sending keyboard events ON / OFF |
| `q` or `ESC` | All | Quit application |
| `r` | Python (`hand_surfers.py`, `hand_surfers_finger.py`) | Re-center the neutral box to current hand position |
| `c` | C++ (`main.cpp`) | Calibrate skin tone using the green sample box |
| `-` | C++ (`main.cpp`) | Tighten skin color tolerance (decrease by 2) |
| `=` / `+` | C++ (`main.cpp`) | Loosen skin color tolerance (increase by 2) |

---

## Getting Started

### Python Setup

1. **Requirements**:
   - Python 3.8+ (tested on Windows 10/11)
   - Webcam

2. **Install dependencies**:
   ```bash
   pip install opencv-python mediapipe
   ```

3. **Run desired mode**:
   ```bash
   # Option 1: Palm tracking with neutral box
   python hand_surfers.py

   # Option 2: Fingertip tracking (faster response, smaller box)
   python hand_surfers_finger.py

   # Option 3: Swipe / flick gestures (no box required)
   python hand_surfers_swipe.py
   ```

*Note: The Python input injection uses `ctypes.windll.user32` and is designed for Windows.*

---

### C++ Build (CMake)

1. **Requirements**:
   - CMake 3.10+
   - C++17 compatible compiler (e.g. MSVC, GCC, Clang)
   - OpenCV library installed and accessible via `CMAKE_PREFIX_PATH` or system path

2. **Build**:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

3. **Run**:
   ```bash
   # Windows
   .\hand_surfers.exe

   # macOS / Linux
   ./hand_surfers
   ```

4. **Calibrate**: Hold your hand inside the green square and press `c` to begin tracking.

---

## Project Structure

```text
HandSurfers/
├── CMakeLists.txt              # CMake build configuration for C++ executable
├── main.cpp                    # C++ OpenCV skin-segmentation implementation
├── hand_surfers.py             # Python + MediaPipe palm center tracking
├── hand_surfers_finger.py      # Python + MediaPipe fingertip tracking
├── hand_surfers_swipe.py       # Python + MediaPipe flick/swipe detector
├── hand_landmarker.task        # MediaPipe Hand Landmarker model file
└── README.md                   # Project documentation
```

