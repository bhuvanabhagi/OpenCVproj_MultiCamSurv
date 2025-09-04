import cv2
import numpy as np
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Put your IP webcam links here ---
urls = {
    "myphone cam": os.getenv("FRONT_DOOR_CAM"),
    "maindoor cam": os.getenv("BACKYARD_CAM"),
    "Laptop Cam": 0 
}

# Create VideoCapture objects
caps = {name: cv2.VideoCapture(link) for name, link in urls.items()}

# VideoWriter for recording
recording = False
rec_writer = None
rec_camera = None  # Which camera is being recorded

# Fullscreen mode
fullscreen_camera = None  # None = dashboard view

# Mouse click state
click_coords = None

def mouse_callback(event, x, y, flags, param):
    global fullscreen_camera, click_coords
    if event == cv2.EVENT_LBUTTONDOWN and fullscreen_camera is None:
        # Dashboard is 800x600 (2x2 grid of 400x300)
        row = y // 300
        col = x // 400
        index = row * 2 + col
        cam_names = list(caps.keys())
        if index < len(cam_names):
            fullscreen_camera = cam_names[index]
            print(f"[INFO] Fullscreen mode: {fullscreen_camera}")
    elif event == cv2.EVENT_LBUTTONDOWN and fullscreen_camera is not None:
        # If already in fullscreen, clicking anywhere returns to dashboard
        fullscreen_camera = None
        print("[INFO] Back to dashboard view.")

cv2.namedWindow("Surveillance Dashboard")
cv2.setMouseCallback("Surveillance Dashboard", mouse_callback)

print("[INFO] Press 's' to save snapshot.")
print("[INFO] Click a camera in dashboard to view fullscreen, click again to return.")
print("[INFO] Press 'r' to start/stop recording the current fullscreen camera.")
print("[INFO] Press 'q' to quit.")

while True:
    frames = []
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            # If camera offline, show black screen
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Offline", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (400, 300))
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frames.append(frame)

    # --- Decide what to show ---
    if fullscreen_camera is None:
        # Dashboard mode (2x2 grid)
        while len(frames) < 4:
            frames.append(np.zeros((300, 400, 3), dtype=np.uint8))

        top_row = np.hstack(frames[:2])
        bottom_row = np.hstack(frames[2:4])
        dashboard = np.vstack([top_row, bottom_row])
        cv2.imshow("Surveillance Dashboard", dashboard)

    else:
        # Fullscreen single camera
        cam_index = list(caps.keys()).index(fullscreen_camera)
        frame = frames[cam_index]
        big_frame = cv2.resize(frame, (800, 600))  # fixed size
        cv2.imshow("Surveillance Dashboard", big_frame)

        # If recording, save frame
        if recording and rec_writer is not None and rec_camera == fullscreen_camera:
            rec_writer.write(big_frame)

    key = cv2.waitKey(1) & 0xFF

    # --- Key Controls ---
    if key == ord('s'):
        # Save snapshots
        for name, frame in zip(caps.keys(), frames):
            filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Snapshot saved: {filename}")

    elif key == ord('r'):
        # Toggle recording
        if fullscreen_camera is None:
            print("[WARN] Go fullscreen on a camera first to record.")
        else:
            if not recording:
                # Start recording (AVI with XVID codec for compatibility)
                filename = f"{fullscreen_camera}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                rec_writer = cv2.VideoWriter(filename, fourcc, 20.0, (800, 600))
                rec_camera = fullscreen_camera
                recording = True
                print(f"[INFO] Recording started for {fullscreen_camera}: {filename}")
                print("[INFO] Press 'r' again to stop recording.")
            else:
                # Stop recording
                recording = False
                rec_writer.release()
                rec_writer = None
                print("[INFO] Recording stopped and saved.")

    elif key == ord('q'):
        break

# Release resources
for cap in caps.values():
    cap.release()
if rec_writer:
    rec_writer.release()
cv2.destroyAllWindows()
