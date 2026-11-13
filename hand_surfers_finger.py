"""
HandSurfers (Python + MediaPipe)
Control Subway Surfers (or any arrow-key game) with your hand via webcam.

Move your hand out of the white box in the middle:
    left / right -> Left / Right arrow
    up           -> Up arrow    (jump)
    down         -> Down arrow  (roll)
Bring your hand back to the white box before the next move.

Keys in the preview window:
    k = toggle sending keys on/off     r = re-center the white box on your hand
    q = quit
"""
import ctypes
import os
import time
import urllib.request
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

COOLDOWN = 0.18      # seconds between key presses
HALF_W = 0.05        # neutral box half-width  (fraction of frame width)
HALF_H = 0.06        # neutral box half-height (fraction of frame height)
SMOOTH = 0.8         # 0..1, higher = less smoothing, lower = smoother but laggier

# (fingertip landmark, joint below it) for index, middle, ring, pinky
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]

# pairs of landmark indexes to draw the hand skeleton
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


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

    center = (0.5, 0.5)       # neutral box center (normalized)
    send_keys = True
    armed = True
    last_fire = 0.0
    last_action = "-"
    pos = None                # smoothed fingertip position (normalized)
    tracked = None            # landmark index of the fingertip being followed
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

        neutral = (int((center[0] - HALF_W) * w), int((center[1] - HALF_H) * h),
                   int((center[0] + HALF_W) * w), int((center[1] + HALF_H) * h))

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
            cv2.circle(frame, (int(pos[0] * w), int(pos[1] * h)), 10, (255, 0, 255), -1)

            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            inside = abs(dx) <= HALF_W and abs(dy) <= HALF_H
            if inside:
                armed = True
            elif armed and (time.time() - last_fire) >= COOLDOWN:
                # direction with the biggest overshoot outside the box
                over_x = abs(dx) - HALF_W
                over_y = abs(dy) - HALF_H
                if over_x >= over_y:
                    key = "left" if dx < 0 else "right"
                else:
                    key = "up" if dy < 0 else "down"
                if send_keys:
                    press(key)
                last_action = key.upper()
                last_fire = time.time()
                armed = False
        else:
            pos = None
            tracked = None
            armed = True

        cv2.rectangle(frame, neutral[:2], neutral[2:], (255, 255, 255), 2)
        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {last_action}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "keys: ON" if send_keys else "keys: OFF", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(frame, "hold one finger up | k = keys | r = recenter | q = quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("HandSurfers", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("k"):
            send_keys = not send_keys
        if k == ord("r") and pos is not None:
            center = pos

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
