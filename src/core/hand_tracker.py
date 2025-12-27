# core/hand_tracker.py

"""
Entrada: frame (OpenCV)
Processo: detecção + landmarks
Saída: Estrutura de dados padronizada
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
        
        """Inicializa o detector de mãos usando CVZONE (MediaPipe)"""
        self.detector = HandDetector(
            staticMode=static_mode,
            maxHands=max_hands,
            # minTrackCon e detectionCon estavam invertidos no código anterior, mas aqui mantemos
            # o seu código original. O CVZone usa detectionCon (detecção) e minTrackCon (rastreio)
            detectionCon=detection_confidence, 
            minTrackCon=tracking_confidence,
        )

    def process(self, frame):
        
        if frame is None:
            return [], None
        
        try:
            hands, annotated_frame = self.detector.findHands( 
                frame, draw=True
            )
            # findHands retorna (lista_de_maos, frame_com_desenho)
            return hands, annotated_frame 
        except Exception:
            # Em caso de falha na detecção (ex: erro no MediaPipe), retorna dados vazios
            return [], frame
        

    def extract_landmarks(self, hand):
        if not hand or "lmList" not in hand:
            return None
        
        return hand["lmList"]