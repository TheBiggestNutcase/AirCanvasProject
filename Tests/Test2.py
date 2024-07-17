import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Change to 0 if this doesn't work

# Read a frame to get the dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to grab initial frame")
    exit()

# Create a black canvas with the same dimensions as the frame
canvas = np.zeros(frame.shape, dtype=np.uint8)

# Drawing settings
drawing = False
prev_x, prev_y = 0, 0
thickness = 5
colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
color_index = 0

# Eraser settings
erasing = False
eraser_thickness = thickness * 5

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Convert to pixel coordinates
            h, w, _ = frame.shape
            current_x, current_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Check if index and thumb are in contact (for drawing)
            if distance(thumb_tip, index_tip) < 0.05:
                drawing = True
                erasing = False
            # Check if index and middle are in contact (for erasing)
            elif distance(index_tip, middle_tip) < 0.05:
                drawing = False
                erasing = True
            else:
                drawing = False
                erasing = False
            
            if drawing:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = current_x, current_y
                cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), colors[color_index], thickness)
            elif erasing:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = current_x, current_y
                cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), (0, 0, 0), eraser_thickness)
            
            prev_x, prev_y = current_x, current_y
    else:
        prev_x, prev_y = 0, 0
    
    # Combine the canvas and the frame
    result = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    
    # Display the current color
    cv2.rectangle(result, (10, 10), (30, 30), colors[color_index], -1)
    
    cv2.imshow("Air Canvas", result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros(frame.shape, dtype=np.uint8)
    elif key == ord('n'):
        color_index = (color_index + 1) % len(colors)
        print(f"Color changed to {colors[color_index]}")  # Debug print

cap.release()
cv2.destroyAllWindows()