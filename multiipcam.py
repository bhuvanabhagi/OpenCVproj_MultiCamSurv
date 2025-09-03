import cv2
import numpy as np
import datetime
import cv2
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
# --- Put your IP webcam links here ---
urls = {
    "Front Door": os.getenv("FRONT_DOOR_CAM"),
    "Backyard": os.getenv("BACKYARD_CAM"),
    "Laptop Cam": 0 
}

# Create VideoCapture objects
caps = {name: cv2.VideoCapture(link) for name, link in urls.items()}

print("[INFO] Press 's' to save snapshot, 'q' to quit.")

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
            # Resize to consistent size
            frame = cv2.resize(frame, (400, 300))
            # Add label and timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(frame)

    # --- Arrange feeds in a grid (2x2) ---
    # Fill empty spots with black frames if fewer than 4
    while len(frames) < 4:
        frames.append(np.zeros((300, 400, 3), dtype=np.uint8))

    top_row = np.hstack(frames[:2])   # First 2 feeds
    bottom_row = np.hstack(frames[2:4])  # Next 2 feeds (or black)
    dashboard = np.vstack([top_row, bottom_row])

    # Show in one window
    cv2.imshow("Surveillance Dashboard", dashboard)

    key = cv2.waitKey(1) & 0xFF

    # Save snapshot
    if key == ord('s'):
        for name, frame in zip(caps.keys(), frames):
            filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Snapshot saved: {filename}")

    # Quit
    if key == ord('q'):
        break

# Release resources
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
