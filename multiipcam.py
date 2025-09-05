import cv2
import numpy as np
import datetime
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Camera sources ---
urls = {
    "myphone cam": os.getenv("FRONT_DOOR_CAM"),
    "maindoor cam": os.getenv("BACKYARD_CAM"),
    "Laptop Cam": 0
}

caps = {name: cv2.VideoCapture(link) for name, link in urls.items()}

# --- Recording setup ---
recorders = {name: None for name in caps.keys()}
recording_flags = {name: None for name in caps.keys()}  # "MANUAL"

# --- UI States ---
fullscreen_camera = None
lowlight_mode = 0  # 0=Normal, 1=Gray, 2=Brightness/Contrast


# ---------------- HELPER FUNCTIONS ---------------- #

def start_recording(name, frame, mode):
    filename = f"{name}_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recorders[name] = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    recording_flags[name] = mode
    print(f"[INFO] {mode} recording started for {name}: {filename}")


def stop_recording(name, mode):
    if recorders[name] is not None:
        recorders[name].release()
        recorders[name] = None
        print(f"[INFO] {mode} recording stopped for {name}")
    recording_flags[name] = None


def mouse_callback(event, x, y, flags, param):
    global fullscreen_camera
    if event == cv2.EVENT_LBUTTONDOWN and fullscreen_camera is None:
        row = y // 300
        col = x // 400
        index = row * 2 + col
        cam_names = list(caps.keys())
        if index < len(cam_names):
            fullscreen_camera = cam_names[index]
            print(f"[INFO] Fullscreen mode: {fullscreen_camera}")
    elif event == cv2.EVENT_LBUTTONDOWN and fullscreen_camera is not None:
        fullscreen_camera = None
        print("[INFO] Back to dashboard view.")


# ---------------- MAIN ---------------- #

cv2.namedWindow("Surveillance Dashboard")
cv2.setMouseCallback("Surveillance Dashboard", mouse_callback)

print("[INFO] Motion detection removed.")
print("[INFO] Press 's' → Snapshot, 'r' → Manual Recording, 'l' → Toggle Low-Light Modes, 'q' → Quit.")

while True:
    frames = []
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Offline", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (400, 300))

            # Low-light mode filters
            if lowlight_mode == 1:  # Gray mode
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif lowlight_mode == 2:  # Brightness/contrast boost
                frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

            # Manual recording
            if recording_flags[name] == "MANUAL" and recorders[name] is not None:
                recorders[name].write(frame)

            # Overlay
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(frame)

    # Dashboard / Fullscreen
    if fullscreen_camera is None:
        while len(frames) < 4:
            frames.append(np.zeros((300, 400, 3), dtype=np.uint8))

        top_row = np.hstack(frames[:2])
        bottom_row = np.hstack(frames[2:4])
        dashboard = np.vstack([top_row, bottom_row])
        cv2.imshow("Surveillance Dashboard", dashboard)
    else:
        cam_index = list(caps.keys()).index(fullscreen_camera)
        big_frame = cv2.resize(frames[cam_index], (800, 600))
        cv2.imshow("Surveillance Dashboard", big_frame)

    # Keys
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        for name, frame in zip(caps.keys(), frames):
            filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Snapshot saved: {filename}")

    elif key == ord('r'):
        if fullscreen_camera is None:
            print("[WARN] Go fullscreen on a camera to record manually.")
        else:
            if recording_flags[fullscreen_camera] != "MANUAL":
                start_recording(fullscreen_camera, frames[list(caps.keys()).index(fullscreen_camera)], "MANUAL")
            else:
                stop_recording(fullscreen_camera, "MANUAL")

    elif key == ord('l'):
        lowlight_mode = (lowlight_mode + 1) % 3
        mode_name = ["Normal", "Gray", "Brightness/Contrast"][lowlight_mode]
        print(f"[INFO] Low-Light Mode: {mode_name}")

    elif key == ord('q'):
        break

# Cleanup
for cap in caps.values():
    cap.release()
for rec in recorders.values():
    if rec is not None:
        rec.release()
cv2.destroyAllWindows()
