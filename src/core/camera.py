import numpy as np
import cv2

class Camera:
    """
    Classe responsável exclusivamente pelo controle da webcam.

    Responsabilidades:
    - Abrir a câmera
    - Ler frames
    - Encerrar corretamente o dispositivo

    Não contém e nem espere:
    - Lógica de negócio
    - Processamento de imagem
    - Integração com ML ou UI
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False


    def start(self) -> None:
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError(f"Não foi possível abrir a camera {self.camera_index}")


        self.is_running = True

    
    def read(self):
        if not self.is_running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()

        if not ret:
            return None
        
        return frame
    
    
    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_running = False

    @staticmethod
    def get_mock_frame(width=640, height=480):
        """
        Gera um frame simulado (preto ou cinza) para uso quando a webcam não está disponível.
        Retorna um array NumPy.
        """
        # Cria um frame preto
        mock_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Desenha texto para debug
        cv2.putText(
            mock_frame,
            "MOCK CAM ATIVA",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255), # Cor branca
            2,
        )
        return mock_frame