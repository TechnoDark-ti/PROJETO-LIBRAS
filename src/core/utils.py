import cv2
import io
import numpy as np
import base64
import os
import csv

from PIL import Image

def cv2_to_flet_image(frame: np.ndarray) -> str:
    """
    Converte um frame OpenCV (NumPy array) para uma string Base64.
    """
    if frame is None:
        return ""
    
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
    #return byte_io.getvalue()

    #5. Codifica os bytes JPEG para Base64
    base64_bytes = base64.b64encode(byte_io.getvalue())
    
    # 6. Decodifica para string e retorna.
    return base64_bytes.decode('utf-8')

def save_landmarks_to_csv(landmarks, label):
    folder = "data/raw_samples"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, f"{label}.csv")

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(landmarks)