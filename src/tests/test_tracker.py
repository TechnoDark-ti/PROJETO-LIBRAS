# testesx/test_tracker.py

import cv2 
from core.hand_tracker import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        data, img = tracker.process_frame(frame)

        print(data)

        cv2.imshow("Hand Tracker Test", img)
        if cv2.waitKey(1) & 0xFF == 27:
             break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()