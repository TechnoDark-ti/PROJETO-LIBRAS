#python 3.10.12

from core.camera import Camera
from core.hand_tracker import HandTracker
from core.translator import Translator
from core.signal_buffer import SignalBuffer
import config


def main():
    """
    Ponto de entrada do sistema.
    Apenas inicializa e conecta os módulos.
    """

    # Inicialização dos módulos
    camera = Camera()
    hand_tracker = HandTracker()
    translator = Translator(model_path="models/libras_model.pt")
    signal_buffer = SignalBuffer(
        size=config.BUFFER_SIZE,
        min_confidence=config.MIN_CONFIDENCE
    )

    # Loop principal (inativo sem webcam)
    while False:
        frame = camera.get_frame()
        landmarks = hand_tracker.process(frame)
        signal = translator.translate(landmarks)
        confirmed_signal = signal_buffer.update(signal)

        if confirmed_signal:
            print("Sinal confirmado:", confirmed_signal)


if __name__ == "__main__":
    main()
