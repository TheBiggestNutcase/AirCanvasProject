import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

drawing = False
prev_x, prev_y = 0, 0
thickness = 5
color = (0, 255, 0) 

erasing = False
eraser_radius = 20  
eraser_speed = 5  

canvas_created = False

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if not canvas_created:
        canvas = frame.copy()
        canvas_created = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance_thumb_index = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
            
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            distance_index_middle = math.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
            
            if distance_thumb_index < 0.05:
                drawing = True
                erasing = False
            elif distance_index_middle < 0.05:
                drawing = False
                erasing = True
            else:
                drawing = False
                erasing = False
            
            if drawing:
                current_x, current_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = current_x, current_y
                cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), color, thickness)
                prev_x, prev_y = current_x, current_y
            elif erasing:
                current_x, current_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = current_x, current_y
                for _ in range(eraser_speed):
                    cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), (0, 0, 0), thickness*5)
                    prev_x, prev_y = current_x, current_y
            else:
                prev_x, prev_y = 0, 0  
    
    frame = cv2.add(frame, canvas)
    
    cv2.imshow('HandTracker', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord('c'):  
        canvas = frame.copy()
        canvas_created = True

cap.release()
cv2.destroyAllWindows()
