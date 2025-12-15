#python 3.10.12

from core.camera import Camera
from core.hand_tracker import HandTracker
from core.translator import LibrasTranslator
from core.signal_buffer import SignalBuffer
import config
import time


def main():
    print("Iniciando Sistema de tradução Libras")


    camera = Camera()
    tracker = HandTracker()
    translator = LibrasTranslator()
    buffer = SignalBuffer(
        size=config.BUFFER_SIZE,
        min_confidence=config.MIN_CONFIDENCE
    )

    if not config.USE_CAMERA:
        print("Modo silmulado ativado")


        for sign in config.SIMULATED_SIGNS:
            print(f"[DEBUG] Detectado: {sign}")

            confirmed = buffer.update(sign)

            if confirmed:
                print(f"Sinal Confirmado: {confirmed}")
                time.sleep(0.5)
        print("simulação finalizada")
        return
    

    #Modo com webcam (se estiver on)
    while True:
        frame = camera.get_frame()
        landmarks = tracker.dedector(frame)
        signal = translator.translate(landmarks)

        confirmed = buffer.update(signal)

        if confirmed:
            print("Sinal Confirmado: ", confirmed)

if __name__ == "__main__":
    main()