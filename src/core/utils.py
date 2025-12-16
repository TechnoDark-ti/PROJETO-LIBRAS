import cv2
import io
import numpy as np

from PIL import Image

def cv2_to_flet_image(frame: np.ndarray) -> bytes:
    """
    Converte um frame OpenCV (NumPy array) para bytes JPEG.
    :param frame: O frame anotado pelo HandTracker.
    :return: Bytes do frame no formato JPEG.
    """
    if frame is None:
        return b''
    
    # O OpenCV usa BGR; Flet/PIL/JPEG prefere RGB.
    # Esta conversão é opcional, mas garante cores corretas.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Converte o array NumPy para uma imagem PIL
    img_pil = Image.fromarray(rgb_frame)
    
    # Cria um buffer de bytes
    byte_io = io.BytesIO()
    
    # Salva a imagem no buffer como JPEG
    img_pil.save(byte_io, format='jpeg')
    
    # Retorna o conteúdo do buffer
    return byte_io.getvalue()