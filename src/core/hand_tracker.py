"""
Entrada: frame (OpenCV)
Processo: detecção + landmarks
Sa´da: Estrutura de dados padroniz#da
"""

import cv2
from cvzone.HandTrackingModule import HandDetector

class HandTracker:
    def __init__(
            self,
            static_mode = False,
            max_hands = 2,
            detection_confidence = 0.7,
            tracking_confidence = 0.6
            ):
        
        """Inicalizado o dedtect de mãos usando CVZONE (MediaPipe)"""
        self.dedector = HandDetector(
            staticMode=static_mode,
            maxHands=max_hands,
            detectionCon=detection_confidence,
            minTrackCon=tracking_confidence,
        )

    def process(self, frame):
        
        if frame is None:
            return [], None
        
        try:
            hands, annotated_frame = self.dedector.findDistance(
                frame, draw=True
            )
            return hands, annotated_frame
        except Exception:
            return [], frame
        


    def extract_landmarks(self, hand):
        if not hand or "lmList" not in hand:
            return None
        
        return hand["lmList"]