import cv2
import numpy as np
from hand_processor import HandProcessor
from utils import distance

class AirCanvas:
    def __init__(self, cam_index=1, max_hands=1, detection_confidence=0.5):
        # Initialize the HandProcessor
        self.hand_processor = HandProcessor(max_hands, detection_confidence)

        # Initialize the webcam
        self.cap = cv2.VideoCapture(cam_index)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to grab initial frame")

        # Create a black canvas with the same dimensions as the frame
        self.canvas = np.zeros(frame.shape, dtype=np.uint8)
        
        # Drawing settings
        self.drawing = False
        self.prev_x, self.prev_y = 0, 0
        self.thickness = 5
        self.colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
        self.color_index = 0

        # Eraser settings
        self.erasing = False
        self.eraser_thickness = self.thickness * 5
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with HandProcessor
        results = self.hand_processor.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.hand_processor.draw_landmarks(frame, hand_landmarks)
                
                # Get landmarks
                thumb_tip = hand_landmarks.landmark[self.hand_processor.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.hand_processor.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[self.hand_processor.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Convert to pixel coordinates
                h, w, _ = frame.shape
                current_x, current_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Check if index and thumb are in contact (for drawing)
                if distance(thumb_tip, index_tip) < 0.05:
                    self.drawing = True
                    self.erasing = False
                # Check if index and middle are in contact (for erasing)
                elif distance(index_tip, middle_tip) < 0.05:
                    self.drawing = False
                    self.erasing = True
                else:
                    self.drawing = False
                    self.erasing = False
                
                if self.drawing:
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = current_x, current_y
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (current_x, current_y), self.colors[self.color_index], self.thickness)
                elif self.erasing:
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = current_x, current_y
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (current_x, current_y), (0, 0, 0), self.eraser_thickness)
                
                self.prev_x, self.prev_y = current_x, current_y
        else:
            self.prev_x, self.prev_y = 0, 0
        
        # Combine the canvas and the frame
        result = cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)
        
        # Display the current color
        cv2.rectangle(result, (10, 10), (30, 30), self.colors[self.color_index], -1)
        
        return result
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            result = self.process_frame(frame)
            
            cv2.imshow("Air Canvas", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros(frame.shape, dtype=np.uint8)
            elif key == ord('n'):
                self.color_index = (self.color_index + 1) % len(self.colors)
                print(f"Color changed to {self.colors[self.color_index]}")  # Debug print
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas(cam_index=1)
    air_canvas.run()
