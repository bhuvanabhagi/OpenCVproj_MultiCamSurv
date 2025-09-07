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
    # "Laptop Cam": 0
}

caps = {name: cv2.VideoCapture(link) for name, link in urls.items()}

# --- Recording setup ---
recorders = {name: None for name in caps.keys()}
recording_flags = {name: None for name in caps.keys()}  # "MANUAL" or "MOTION"

# --- Motion Detection Setup ---
background_subtractors = {name: cv2.createBackgroundSubtractorMOG2(detectShadows=True) for name in caps.keys()}
motion_detected = {name: False for name in caps.keys()}
motion_timers = {name: 0 for name in caps.keys()}  # Timer to track motion duration
no_motion_timers = {name: 0 for name in caps.keys()}  # Timer for no motion (to stop recording)

# Motion detection parameters
MIN_CONTOUR_AREA = 1500  # Minimum area to consider as human/animal motion
MOTION_THRESHOLD = 10000  # Total motion area threshold
NO_MOTION_DELAY = 3.0  # Seconds to wait before stopping recording after motion stops
MOTION_LOG_FILE = "motion_log.txt"

# --- UI States ---
fullscreen_camera = None
lowlight_mode = 0  # 0=Normal, 1=Gray, 2=Brightness/Contrast
motion_detection_enabled = True  # Toggle for motion detection


# ---------------- HELPER FUNCTIONS ---------------- #

def log_motion_event(camera_name, event_type):
    """Log motion events to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MOTION_LOG_FILE, "a") as f:
        f.write(f"{timestamp} - {event_type} on {camera_name}\n")
    print(f"[MOTION] {event_type} on {camera_name} at {timestamp}")


def detect_motion(frame, bg_subtractor, camera_name):
    """
    Smart motion detection that filters out small movements (fans, leaves)
    Returns True if significant human/animal-like motion is detected
    """
    if not motion_detection_enabled:
        return False
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_motion_area = 0
    significant_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter out small movements (fans, leaves, small objects)
        if area > MIN_CONTOUR_AREA:
            # Additional filtering based on contour properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Human/animal-like shapes typically have reasonable aspect ratios
            if 0.2 < aspect_ratio < 5.0 and w > 30 and h > 50:
                significant_contours.append(contour)
                total_motion_area += area
    
    # Draw motion areas on frame for visualization
    if significant_contours:
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return total_motion_area > MOTION_THRESHOLD


def start_recording(name, frame, mode):
    """Start recording for a camera"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_{mode}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recorders[name] = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    recording_flags[name] = mode
    print(f"[INFO] {mode} recording started for {name}: {filename}")
    
    if mode == "MOTION":
        log_motion_event(name, "Recording started - Motion detected")


def stop_recording(name, mode):
    """Stop recording for a camera"""
    if recorders[name] is not None:
        recorders[name].release()
        recorders[name] = None
        print(f"[INFO] {mode} recording stopped for {name}")
        
        if mode == "MOTION":
            log_motion_event(name, "Recording stopped - No motion")
    
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

print("[INFO] Smart Motion Detection ENABLED - Filters out fans, leaves, small movements")
print("[INFO] Motion-triggered recording ENABLED")
print("[INFO] Controls:")
print("  's' → Snapshot all cameras")
print("  'r' → Manual recording toggle (fullscreen mode)")
print("  'm' → Toggle motion detection ON/OFF")
print("  'l' → Toggle low-light modes")
print("  'q' → Quit")

# Initialize motion log file
with open(MOTION_LOG_FILE, "w") as f:
    f.write(f"Motion Detection Log - Started at {datetime.datetime.now()}\n")
    f.write("-" * 50 + "\n")

while True:
    frames = []
    current_time = time.time()
    
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Offline", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (400, 300))
            original_frame = frame.copy()  # Keep original for motion detection

            # --- MOTION DETECTION ---
            if motion_detection_enabled:
                motion_now = detect_motion(original_frame, background_subtractors[name], name)
                
                if motion_now:
                    motion_detected[name] = True
                    motion_timers[name] = current_time
                    no_motion_timers[name] = 0  # Reset no-motion timer
                    
                    # Start motion recording if not already recording
                    if recording_flags[name] != "MOTION" and recording_flags[name] != "MANUAL":
                        start_recording(name, frame, "MOTION")
                
                else:
                    # No motion detected
                    if motion_detected[name]:
                        if no_motion_timers[name] == 0:
                            no_motion_timers[name] = current_time  # Start no-motion timer
                        elif current_time - no_motion_timers[name] > NO_MOTION_DELAY:
                            # Stop motion recording after delay
                            if recording_flags[name] == "MOTION":
                                stop_recording(name, "MOTION")
                            motion_detected[name] = False
                            no_motion_timers[name] = 0

            # Low-light mode filters
            if lowlight_mode == 1:  # Gray mode
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif lowlight_mode == 2:  # Brightness/contrast boost
                frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

            # Recording (both manual and motion-triggered)
            if recorders[name] is not None:
                recorders[name].write(frame)

            # --- OVERLAYS ---
            # Camera name
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Motion status
            if motion_detection_enabled:
                if motion_detected[name]:
                    # Red border and text for motion
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 3)
                    cv2.putText(frame, "MOTION DETECTED", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Recording status
            if recording_flags[name] == "MANUAL":
                cv2.putText(frame, "REC (MANUAL)", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif recording_flags[name] == "MOTION":
                cv2.putText(frame, "REC (MOTION)", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frames.append(frame)

    # Dashboard / Fullscreen display
    if fullscreen_camera is None:
        while len(frames) < 4:
            frames.append(np.zeros((300, 400, 3), dtype=np.uint8))

        top_row = np.hstack(frames[:2])
        bottom_row = np.hstack(frames[2:4])
        dashboard = np.vstack([top_row, bottom_row])
        
        # Add motion detection status to dashboard
        status_text = "Motion Detection: ON" if motion_detection_enabled else "Motion Detection: OFF"
        cv2.putText(dashboard, status_text, (10, dashboard.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if motion_detection_enabled else (0, 0, 255), 2)
        
        cv2.imshow("Surveillance Dashboard", dashboard)
    else:
        cam_index = list(caps.keys()).index(fullscreen_camera)
        big_frame = cv2.resize(frames[cam_index], (800, 600))
        cv2.imshow("Surveillance Dashboard", big_frame)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Snapshot all cameras
        for name, frame in zip(caps.keys(), frames):
            filename = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Snapshot saved: {filename}")

    elif key == ord('r'):  # Manual recording toggle
        if fullscreen_camera is None:
            print("[WARN] Go fullscreen on a camera to record manually.")
        else:
            if recording_flags[fullscreen_camera] != "MANUAL":
                # Stop motion recording if active before starting manual
                if recording_flags[fullscreen_camera] == "MOTION":
                    stop_recording(fullscreen_camera, "MOTION")
                start_recording(fullscreen_camera, frames[list(caps.keys()).index(fullscreen_camera)], "MANUAL")
            else:
                stop_recording(fullscreen_camera, "MANUAL")

    elif key == ord('m'):  # Toggle motion detection
        motion_detection_enabled = not motion_detection_enabled
        status = "ENABLED" if motion_detection_enabled else "DISABLED"
        print(f"[INFO] Motion Detection {status}")
        
        # Stop all motion recordings when disabling
        if not motion_detection_enabled:
            for name in caps.keys():
                if recording_flags[name] == "MOTION":
                    stop_recording(name, "MOTION")
                motion_detected[name] = False

    elif key == ord('l'):  # Toggle low-light modes
        lowlight_mode = (lowlight_mode + 1) % 3
        mode_name = ["Normal", "Gray", "Brightness/Contrast"][lowlight_mode]
        print(f"[INFO] Low-Light Mode: {mode_name}")

    elif key == ord('q'):  # Quit
        break

# Cleanup
print("[INFO] Shutting down surveillance system...")
for cap in caps.values():
    cap.release()
for rec in recorders.values():
    if rec is not None:
        rec.release()
cv2.destroyAllWindows()

print(f"[INFO] Motion log saved to: {MOTION_LOG_FILE}")
print("[INFO] Surveillance system stopped.")