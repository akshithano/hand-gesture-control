import cv2
import mediapipe as mp
import sys
print(sys.executable)

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # fix 2
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # fix 2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils
points = []
mode = "draw"
frame_count = 0
last_result = None

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fix 3 - non writeable
    rgb.flags.writeable = False
    frame_count += 1
    if frame_count % 2 == 0:          # fix 1 - skip every 2nd frame
        last_result = hands.process(rgb)
    result = last_result
    rgb.flags.writeable = True

    if result and result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=handLms,
                connections=mp_hands.HAND_CONNECTIONS
            )
            h, w, c = frame.shape
            index_finger = handLms.landmark[8]
            cx = int(index_finger.x * w)
            cy = int(index_finger.y * h)

            fingers = fingers_up(handLms)
            if fingers[1] and fingers[2]:
                mode = "erase"
                points.append(None)
            elif fingers[1] and not fingers[2]:
                mode = "draw"
                if points and points[-1] is not None:
                    last = points[-1]
                    if abs(last[0] - cx) < 50 and abs(last[1] - cy) < 50:
                        points.append((cx, cy, mode))
                    else:
                        points.append(None)
                        points.append((cx, cy, mode))
                else:
                    points.append((cx, cy, mode))
            else:
                points.append(None)
    else:
        points.append(None)

    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        x1, y1, m1 = points[i-1]
        x2, y2, m2 = points[i]
        if m1 == "erase" or m2 == "erase":
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 20)
        else:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Show mode on screen
    mode_text = "ERASE" if mode == "erase" else "DRAW"
    color = (0, 0, 255) if mode == "erase" else (0, 255, 0)
    cv2.putText(frame, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        points = []
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()