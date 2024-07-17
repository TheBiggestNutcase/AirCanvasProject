import mediapipe as mp

class HandProcessor:
    def __init__(self, max_hands=1, detection_confidence=0.5):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=max_hands, 
            min_detection_confidence=detection_confidence
        )
    
    def process(self, image):
        return self.hands.process(image)
    
    def draw_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
