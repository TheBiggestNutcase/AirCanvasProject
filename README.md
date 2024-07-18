# AirCanvasProject

```python
import cv2
import numpy as np
import mediapipe as mp
import math
```
These are the necessary library imports. OpenCV (cv2) for image processing, NumPy for numerical operations, MediaPipe for hand tracking, and math for mathematical operations.

```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
```
This initializes MediaPipe's hand detection model. We're setting it to detect one hand and use a 50% confidence threshold.

```python
cap = cv2.VideoCapture(1)
```
This opens the webcam. If it doesn't work, try changing 1 to 0.

```python
ret, frame = cap.read()
if not ret:
    print("Failed to grab initial frame")
    exit()

canvas = np.zeros(frame.shape, dtype=np.uint8)
```
This reads an initial frame to get the dimensions and creates a black canvas of the same size.

```python
drawing = False
prev_x, prev_y = 0, 0
thickness = 5
colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
color_index = 0

erasing = False
eraser_thickness = thickness * 5
```
These are the drawing and erasing settings. It defines colors, line thickness, and eraser thickness.

```python
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
```
This function calculates the Euclidean distance between two points.

```python
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
This is the main loop. It reads each frame, flips it horizontally, and converts it to RGB for MediaPipe.

```python
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
```
This processes the frame to detect hands and draws the hand landmarks if a hand is detected.

```python
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            h, w, _ = frame.shape
            current_x, current_y = int(index_tip.x * w), int(index_tip.y * h)
```
This extracts the positions of the thumb, index, and middle finger tips, and converts the index finger tip position to pixel coordinates.

```python
            if distance(thumb_tip, index_tip) < 0.05:
                drawing = True
                erasing = False
            elif distance(index_tip, middle_tip) < 0.05:
                drawing = False
                erasing = True
            else:
                drawing = False
                erasing = False
```
This checks the distances between fingers to determine if the user is drawing or erasing.

```python
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
```
This section handles the actual drawing or erasing on the canvas.

```python
    result = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    
    cv2.rectangle(result, (10, 10), (30, 30), colors[color_index], -1)
    
    cv2.imshow("Air Canvas", result)
```
This combines the original frame with the canvas and displays the result. It also shows a small rectangle indicating the current color.

```python
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros(frame.shape, dtype=np.uint8)
    elif key == ord('n'):
        color_index = (color_index + 1) % len(colors)
        print(f"Color changed to {colors[color_index]}")  # Debug print
```
This handles key presses: 'q' to quit, 'c' to clear the canvas, and 'n' to change colors.

```python
cap.release()
cv2.destroyAllWindows()
```
This releases the webcam and closes all windows when the program ends.

This code creates an interactive air canvas where you can draw by bringing your thumb and index finger together, erase by bringing your index and middle finger together, and change colors or clear the canvas using keyboard inputs.
