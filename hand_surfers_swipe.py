"""
HandSurfers (Python + MediaPipe)
Control Subway Surfers (or any arrow-key game) with your hand via webcam.

Move your hand out of the white box in the middle:
    left / right -> Left / Right arrow
    up           -> Up arrow    (jump)
    down         -> Down arrow  (roll)
Bring your hand back to the white box before the next move.

Keys in the preview window:
    k = toggle sending keys on/off
    q = quit
"""
import ctypes
import math
import os
import time
import urllib.request
from collections import deque
from ctypes import wintypes

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ---- send arrow keys through the Windows SendInput API (no extra library needed) ----
_PUL = ctypes.POINTER(ctypes.c_ulong)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _PUL)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _PUL)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


_VK = {"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28}


def press(key):
    user32 = ctypes.windll.user32
    vk = _VK[key]
    scan = user32.MapVirtualKeyW(vk, 0)
    for flags in (0x0001, 0x0001 | 0x0002):       # EXTENDEDKEY down, then EXTENDEDKEY|KEYUP
        inp = _INPUT(type=1)                      # INPUT_KEYBOARD
        inp.u.ki = _KEYBDINPUT(vk, scan, flags, 0, None)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

WINDOW = 0.25        # seconds: a swipe must happen within this time
SWIPE_DIST = 0.10    # how far the fingertip must travel in WINDOW (fraction of the frame)
SWIPE_SPEED = 0.08   # ...and at least this much of it within the last 0.1 s, so slow drifts do not count
COOLDOWN = 0.25      # minimum seconds between two moves
STILL_TIME = 0.12    # the finger must pause this long before the next move is allowed
STILL_DIST = 0.03    # ...moving less than this (fraction of frame) counts as paused
SMOOTH = 0.8         # 0..1, higher = less smoothing, lower = smoother but laggier

# (fingertip landmark, joint below it) for index, middle, ring, pinky
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]

# pairs of landmark indexes to draw the hand skeleton
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class SwipeDetector:
    """Fires a key when the fingertip makes a quick flick; no neutral box needed.
    After a flick the finger must pause briefly, so pulling your hand back
    to where it started does not trigger the opposite move."""

    def __init__(self):
        self.hist = deque()
        self.armed = True
        self.last_fire = 0.0

    def reset(self):
        self.hist.clear()
        self.armed = True

    def update(self, t, x, y):
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > WINDOW:
            self.hist.popleft()
        if len(self.hist) < 2:
            return None

        if not self.armed:
            recent = [p for p in self.hist if t - p[0] <= STILL_TIME]
            if len(recent) >= 2 and recent[-1][0] - recent[0][0] >= STILL_TIME * 0.6:
                moved = math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
                if moved < STILL_DIST:
                    self.armed = True
            return None

        if t - self.last_fire < COOLDOWN:
            return None
        _, x0, y0 = self.hist[0]
        dx, dy = x - x0, y - y0
        if max(abs(dx), abs(dy)) < SWIPE_DIST:
            return None
        recent = [p for p in self.hist if t - p[0] <= 0.1]
        if len(recent) < 2 or max(abs(x - recent[0][1]), abs(y - recent[0][2])) < SWIPE_SPEED:
            return None
        if abs(dx) >= abs(dy):
            key = "left" if dx < 0 else "right"
        else:
            key = "up" if dy < 0 else "down"
        self.armed = False
        self.last_fire = t
        self.hist.clear()
        return key


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Opened camera {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # MJPG + 30 fps keeps the webcam from dropping to 15 fps in dim light (a big source of lag)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise SystemExit("Could not open a webcam. Close other apps using it and check "
                     "Windows Settings > Privacy & security > Camera.")


def main():
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = open_camera()

    send_keys = True
    last_action = "-"
    pos = None                # smoothed fingertip position (normalized)
    tracked = None            # landmark index of the fingertip being followed
    swipe = SwipeDetector()
    t0 = time.time()
    fps, fps_t, fps_n = 0.0, time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera returned no frame. Exiting.")
            break
        frame = cv2.flip(frame, 1)          # mirror so it feels natural
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)


        target = None             # (x, y) of the tracked fingertip, normalized
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            # A finger held up has its tip above the joint below it. Track the highest such finger,
            # and keep following the same finger until it is no longer held up.
            extended = [t for t, j in FINGERS if lm[t].y < lm[j].y - 0.02]
            if tracked not in extended:
                tracked = min(extended, key=lambda t: lm[t].y) if extended else None
            if tracked is not None:
                target = (lm[tracked].x, lm[tracked].y)

            # draw hand
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (0, 0, 255), -1)

        if target is not None:
            pos = target if pos is None else (
                SMOOTH * target[0] + (1 - SMOOTH) * pos[0], SMOOTH * target[1] + (1 - SMOOTH) * pos[1])
            key = swipe.update(time.time(), pos[0], pos[1])
            if key:
                if send_keys:
                    press(key)
                last_action = key.upper()
            # green dot = ready for a flick, red = pausing after a move
            color = (0, 200, 0) if swipe.armed else (0, 0, 255)
            cv2.circle(frame, (int(pos[0] * w), int(pos[1] * h)), 10, color, -1)
        else:
            pos = None
            tracked = None
            swipe.reset()

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {last_action}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "keys: ON" if send_keys else "keys: OFF", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(frame, "hold one finger up and flick it | k = keys | q = quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("HandSurfers", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("k"):
            send_keys = not send_keys

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
