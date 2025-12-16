"""
@Author: Márcio Moda
This code is Licensed by GPL V.3
"""

# main.py

import time
import config
import os
import sys

# Chamando todos os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Importação dos 5 módulos do CORE
from core.camera import Camera
from core.hand_tracker import HandTracker
from core.signal_classifier import SignalClassifier # NOVO MÓDULO INCLUÍDO
from core.signal_buffer import SignalBuffer
from core.translator import Translator


def main():
    print("Iniciando Sistema de Tradução Libras")
    
    # 1. Inicialização dos módulos (agora com 5 classes no Core)
    camera = Camera()
    hand_tracker = HandTracker()
    
    # CORREÇÃO 1: Instanciando o SignalClassifier (que usará o PyTorch/Regras)
    signal_classifier = SignalClassifier(model_path="models/libras_model.pt") 
    
    signal_buffer = SignalBuffer(
        size=config.BUFFER_SIZE,
        min_confidence=config.MIN_CONFIDENCE
    )
    
    # CORREÇÃO 2: O Translator não precisa de argumentos de modelo (.pt)
    translator = Translator() 

    
    # 2. Modo Simulado (Ativado via config.py)
    if not config.USE_CAMERA:
        print("Modo Simulado Ativado: Usando lista de sinais para teste do Pipeline.")
        
        for sign in config.SIMULATED_SIGNS: # Correção de SIMULATED_SIGN para SIMULATED_SIGNS
            print(f"[DEBUG] Sinal Bruto Detectado: {sign}")

            # O Buffer atualiza o sinal, garantindo estabilidade
            confirmed_signal = signal_buffer.update(sign) 

            if confirmed_signal:
                # O Translator traduz o sinal estável para texto
                translated_text = translator.translate(confirmed_signal) 
                print(f"Sinal Confirmado (Estável): {confirmed_signal} -> Tradução: {translated_text}")
            
            time.sleep(0.05) # Pequeno delay para simular frames
        
        print("Simulação finalizada com sucesso.")
        return


    # 3. Loop Principal (Ativado apenas se config.USE_CAMERA = True)
    
    try:
        camera.start()
        print("Câmera iniciada com sucesso.")
        
        while True:
            frame = camera.read()
            if frame is None:
                continue

            # PIPELINE DE 5 CAMADAS:
            
            # 1. HAND TRACKING
            hands, annotated_frame = hand_tracker.process(frame)
            
            # 2. CLASSIFICATION (Somente se houver mãos)
            signal = "Nenhum"
            if hands:
                landmarks = hand_tracker.extract_landmarks(hands[0])
                signal = signal_classifier.classify(landmarks)
            
            # 3. BUFFER
            confirmed_signal = signal_buffer.update(signal)

            # 4. TRANSLATION & OUTPUT
            if confirmed_signal:
                translated_text = translator.translate(confirmed_signal)
                # print(f"Output: {translated_text}") # Aqui será enviada a UI
            
            # (FUTURO) Exibir annotated_frame via Flet ou OpenCV
            
    except RuntimeError as e:
        print(f"ERRO FATAL: {e}")
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
    finally:
        camera.stop()
        print("Sistema encerrado.")


if __name__ == "__main__":
    main()