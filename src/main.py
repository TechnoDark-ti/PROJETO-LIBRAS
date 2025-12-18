"""
@Author: Márcio Moda
This code is Licensed by GPL V.3
"""

import time
import config
import os
import sys
import flet as ft
import threading
import numpy as np

# Chamando todos os módulos (gambiarra do python, c/java/c++ fazem isso nativamente, python é uma linguaguem porca)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Importação dos 5 módulos do CORE
from core.camera import Camera
from core.hand_tracker import HandTracker
from core.signal_classifier import SignalClassifier
from core.signal_buffer import SignalBuffer
from core.translator import Translator
from core.utils import cv2_to_flet_image

# Importação da GUI
from ui.app import MainApp, start_app # Launcher da UI


# ------------------------------------------------------------------------
# 1. FUNÇÃO DE PROCESSAMENTO PESADO (Executada em um thread separado)
# ------------------------------------------------------------------------

def processing_loop(camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback):
    """
    Loop principal que executa o pipeline de visão computacional.
    Roda em um thread separado para não bloquear a UI.
    """
    local_history = [] 
    
    # ------------------------------------------------------------------------
    # MODO SIMULADO (Webcam OFF) - CORRIGIDO
    # ------------------------------------------------------------------------

    if not config.USE_CAMERA:
        print("Thread de Processamento: Modo Simulado Ativado")

        # 1. Definindo a sequência de simulação (para teste de buffer visual)
        simulated_signs = config.SIMULATED_SIGNS + ["Nenhum"] * 5
        
        for sign in simulated_signs:
            # Assumimos que o 'sign' é o sinal bruto detectado na simulação
            current_signal = sign
            
            # Processamento do Core (Buffer e Tradução)
            confirmed_signal = signal_buffer.update(current_signal)
            translated_text = translator.translate(confirmed_signal)
            
            if confirmed_signal:
                local_history.append(confirmed_signal)
            
            # NOVO: Gera e converte o frame Mock para visualização no Flet
            mock_frame = Camera.get_mock_frame() 
            frame_bytes = cv2_to_flet_image(mock_frame)
            
            # SINCRONIZAÇÃO: Envia os dados e o FRAME para a UI
            ui_callback(
                current_signal=current_signal,
                #confirmed_signal=confirmed_signal, # Adicionado para debug
                translated_text=translated_text,
                history=local_history.copy(),
                frame_bytes=frame_bytes
            )
            
            time.sleep(0.5) 
            
        print("Simulação finalizada.")
        return # Finaliza a thread após a simulação
        

    # ------------------------------------------------------------------------
    # MODO REAL (Webcam ON) - REESTRUTURADO
    # ------------------------------------------------------------------------

    try:
        camera.start()
        print("Câmera iniciada com sucesso.")
        
        while True:
            frame = camera.read()
            if frame is None:
                continue

            # 1. HAND TRACKING
            hands, annotated_frame = hand_tracker.process(frame)
            current_signal = "Nenhum"
            
            # 2. CLASSIFICATION (Obter landmarks e classificar)
            if hands:
                landmarks = hand_tracker.extract_landmarks(hands[0])
                current_signal = signal_classifier.classify(landmarks)
            
            # 3. BUFFER
            confirmed_signal = signal_buffer.update(current_signal)

            # --- NOVO: CÁLCULO DE CONFIANÇA ---
            # Verificamos quantas vezes o sinal atual aparece no buffer
            count = list(signal_buffer.buffer).count(current_signal)
            confidence_val = count / signal_buffer.buffer.maxlen if signal_buffer.buffer.maxlen > 0 else 0
            # ------------------------------------------------

            # 5. TRANSLATION & OUTPUT
            translated_text = translator.translate(confirmed_signal)
            
            if confirmed_signal:
                local_history.append(confirmed_signal)

            # 6. CONVERSÃO E SINCRONIZAÇÃO
            frame_bytes = cv2_to_flet_image(annotated_frame) 

            ui_callback(
                current_signal=current_signal,
                #confirmed_signal=confirmed_signal,
                translated_text=translated_text,
                history=local_history.copy(),
                frame_bytes=frame_bytes,
                confidence=confidence_val
            )
            
            # time.sleep(1/30) # Opcional: controle de FPS
                
    except RuntimeError as e:
        print(f"ERRO FATAL (Core): {e}")
    except Exception as e:
        print(f"Erro inesperado no thread de processamento: {e}")
    finally:
        camera.stop()
        print("Thread de processamento encerrado.")

def reset_buffer_handler(e, signal_buffer_instance):
    """ Handler para o botão 'RESET BUFFER' (Refazer). """
    result = signal_buffer_instance.reset()
    print(f"[AÇÃO] Buffer resetado: {result}")

def main(page: ft.Page):
    print("Flet UI Iniciada.")
    
    # 1. Inicialização dos módulos do core (continuam aqui!)
    camera = Camera()
    hand_tracker = HandTracker()
    signal_classifier = SignalClassifier(model_path="models/libras_model.pt") 
    signal_buffer = SignalBuffer(size=config.BUFFER_SIZE, min_confidence=config.MIN_CONFIDENCE)
    translator = Translator() 
    
    # 2. Inicializa a UI (Cria a instância da MainApp)
    app = MainApp(page)
    
    # 3. Define a função de callback que o thread vai usar para se comunicar
    ui_callback_func = app.update_ui_with_data

    def start_handler(e):
        # [FUTURO]: Lógica para parar/iniciar o thread de processamento
        print("[AÇÃO] Botão INICIAR/PAUSAR clicado.")
    
    reset_handler_with_args = lambda e: reset_buffer_handler(e, signal_buffer)
    app.set_handlers(start_handler, reset_handler_with_args) # NOVO MÉTODO CHAMADO
    
    # 4. Inicia o thread de processamento (passando os módulos e a função de callback)
    processing_thread = threading.Thread(
        target=processing_loop,
        args=(camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback_func),
        daemon=True # Garante que o thread morra se o app principal fechar
    )
    processing_thread.start()

if __name__ == "__main__":
    start_app(target_main=main)