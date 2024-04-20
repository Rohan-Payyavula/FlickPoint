# gesture_recognition/hand_gesture_recognizer.py

import cv2
import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self):
        # MediaPipe hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = self.hands.process(image)

        # Draw hand landmarks on the original image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image, results

    def recognize_gestures(self, results):
        # Placeholder for gesture recognition logic
        # Here you could analyze `results.multi_hand_landmarks` to classify gestures
        gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Example: Check if thumb is open
                thumb_is_open = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < \
                                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y
                if thumb_is_open:
                    gestures.append("Thumb Open")
                else:
                    gestures.append("Thumb Closed")
        
        return gestures

    def release_resources(self):
        # Clean up
        self.hands.close()

# Example usage
if __name__ == "__main__":
    # Open video capture
    cap = cv2.VideoCapture(0)

    recognizer = HandGestureRecognizer()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            processed_frame, results = recognizer.process_frame(frame)
            gestures = recognizer.recognize_gestures(results)
            print("Detected Gestures:", gestures)

            # Display the processed image
            cv2.imshow('Hand Gestures', processed_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        recognizer.release_resources()
        cap.release()
        cv2.destroyAllWindows()