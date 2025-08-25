import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import sys

# Initialize
screen_w, screen_h = pyautogui.size()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye and iris landmarks
LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]
LEFT_IRIS = 468

# Thresholds & Timers
blink_threshold = 0.015
click_hold_time = 1.5
scroll_cooldown = 0.5
blink_cooldown = 0.3
dead_zone_top = 0.35
dead_zone_bottom = 0.55

prev_x, prev_y = 0, 0
alpha = 0.3
last_scroll_time = time.time()
click_hold_active = False
eye_closed_start = None
blink_times = []
pause_control = False
shutdown_trigger_time = None
last_blink_time = 0

# Utils
def get_normalized_eye_height(landmarks, eye_points):
    top = landmarks.landmark[eye_points[0]]
    bottom = landmarks.landmark[eye_points[1]]
    return abs(top.y - bottom.y)

def now():
    return time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect eye states
            left_height = get_normalized_eye_height(face_landmarks, LEFT_EYE)
            right_height = get_normalized_eye_height(face_landmarks, RIGHT_EYE)
            eyes_closed = left_height < blink_threshold and right_height < blink_threshold
            left_closed = left_height < blink_threshold and right_height >= blink_threshold
            right_closed = right_height < blink_threshold and left_height >= blink_threshold

            # App shutdown if both eyes closed for 5 seconds
            if eyes_closed:
                if shutdown_trigger_time is None:
                    shutdown_trigger_time = now()
                elif now() - shutdown_trigger_time > 5:
                    print("💤 Eyes closed for 5 seconds. Eye control shutting down.")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
            else:
                shutdown_trigger_time = None

            current_time = now()

            # # 🔥 Left eye only = CLOSE browser/app
            # if left_closed:
            #     print("👁️ Left Eye Closed — Closing App/Web")
            #     pyautogui.hotkey('ctrl', 'w')
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     sys.exit()

            # 🔥 Right eye only = SELECTION / Drag Mode
            if right_closed:
                if not click_hold_active:
                    print("🟦 Selection Mode ON (Right Eye)")
                    pyautogui.mouseDown()
                    click_hold_active = True
            else:
                if click_hold_active:
                    print("⬜ Selection Complete (Released)")
                    pyautogui.mouseUp()
                    click_hold_active = False

            # Blink-based actions (ignore if selection active)
            if eyes_closed and current_time - last_blink_time > blink_cooldown:
                blink_times.append(current_time)
                last_blink_time = current_time

            blink_times = [t for t in blink_times if current_time - t < 1.2]
            blink_count = len(blink_times)

            if blink_count == 1:
                print("👁️ Single Blink: LEFT CLICK")
                pyautogui.click()
                blink_times.clear()
            elif blink_count == 2:
                print("👁️👁️ Double Blink: DOUBLE CLICK")
                pyautogui.doubleClick()
                blink_times.clear()
            elif blink_count == 3:
                pause_control = not pause_control
                print("⏸️ Eye Control Toggled:", "Paused" if pause_control else "Resumed")
                blink_times.clear()
            elif blink_count >= 4:
                print("👁️👁️👁️👁️ Right Click")
                pyautogui.rightClick()
                blink_times.clear()

            if pause_control:
                cv2.putText(frame, "PAUSED", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Eye Control System", frame)
                continue

            # Move Cursor
            iris = face_landmarks.landmark[LEFT_IRIS]
            x = int(iris.x * w)
            y = int(iris.y * h)
            screen_x = int(iris.x * screen_w)
            screen_y = int(iris.y * screen_h)
            smooth_x = int(alpha * screen_x + (1 - alpha) * prev_x)
            smooth_y = int(alpha * screen_y + (1 - alpha) * prev_y)
            prev_x, prev_y = smooth_x, smooth_y
            pyautogui.moveTo(smooth_x, smooth_y)

            # Scroll
            if current_time - last_scroll_time > scroll_cooldown:
                if iris.y < dead_zone_top:
                    pyautogui.scroll(80)
                    print("🔼 Scroll Up")
                    last_scroll_time = current_time
                elif iris.y > dead_zone_bottom:
                    pyautogui.scroll(-70)
                    print("🔽 Scroll Down")
                    last_scroll_time = current_time

            # Visuals
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.line(frame, (0, int(dead_zone_top * h)), (w, int(dead_zone_top * h)), (255, 255, 0), 2)
            cv2.line(frame, (0, int(dead_zone_bottom * h)), (w, int(dead_zone_bottom * h)), (255, 0, 255), 2)

            if click_hold_active:
                cv2.putText(frame, "SELECTING", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Eye Control System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
