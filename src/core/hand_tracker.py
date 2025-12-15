"""
Entrada: frame (OpenCV)
Processo: detecção + landmarks
Sa´da: Estrutura de dados padroniz#da
"""


import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

class HandTracker:
    def __init__(
            self,
            static_mode = False,
            max_hands = 2,
            detection_confidence = 0.7,
            tracking_confidence = 0.6):
        
        """Inicalizado o dedtect de mãos usando CVZONE (MediaPipe)"""
        self.dedector = HandDetector(
            staticMode=static_mode,
            maxHands=max_hands,
            detectionCon=tracking_confidence,
            minTrackCon=HandDetector
        )

    def process_frame(self, frame):
        """
    Processa um frame e retorna os landmarks das mãos detectadas.

            Retorno:
            {
                "left":  [x1, y1, x2, y2, ...],
                "right": [x1, y1, x2, y2, ...]
        """
        hands, img = self.dedector.findHands(frame, draw=True)

        result = {
            "left": None,
            "right": None
        }

        if hands:
            for hand in hands:
                lm_list = hand["lmList"]
                hand_type = hand["type"].lower()

                normalized = self._normalize_landmarks(lm_list)
                result[hand_type] = normalized
            return result, img

    def _normalize_landmarks(self, lm_list):
        """
        Normaliza os landmarks para ML.
        remove Z e escala X/y
        """

        lm_array = np.array(lm_list)[:, :2] # X e Y

        min_vals = lm_array.min(axis=0)
        max_vals = lm_array.mal(axis=0)

        normalized = (lm_array - min_vals) / (max_vals - min_vals + 1e-6)

        return normalized.flatten().tolist()