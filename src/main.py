"""
@Author: Márcio Moda
This code is Licensed by GPL V.3
"""

# main.py

import time
import config
import os
import sys
import flet as ft
import threading

# Chamando todos os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Importação dos 5 módulos do CORE
from core.camera import Camera
from core.hand_tracker import HandTracker
from core.signal_classifier import SignalClassifier # NOVO MÓDULO INCLUÍDO
from core.signal_buffer import SignalBuffer
from core.translator import Translator

# Importação da GUI
from ui.app import MainApp, start_app # Launcher da UI


# ------------------------------------------------------------------------
# 1. FUNÇÃO DE PROCESSAMENTO PESADO (Executada em um thread separado)
# ------------------------------------------------------------------------

def processing_loop(
    camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback
):
    """
    Loop principal que executa o pipeline de visão computacional.
    Roda em um thread separado para não bloquear a UI.
    """
    
    # Histórico Local: O thread deve manter o histórico
    local_history = [] 

    # Se a câmera real não for usada, entramos em modo de simulação
    if not config.USE_CAMERA:
        print("Thread de Processamento: Modo Simulado Ativado")
        
        # Simulação de frames e sinais para teste de pipeline visual e lógico
        simulated_signs = config.SIMULATED_SIGNS + ["Nenhum"] * 5
        
        for sign in simulated_signs:
            
            # --- SIMULAÇÃO DA DETECÇÃO ---
            current_signal = sign
            
            # (No modo real, o classifier nos daria o current_signal)
            
            # --- PROCESSAMENTO DO CORE ---
            confirmed_signal = signal_buffer.update(current_signal)
            translated_text = translator.translate(confirmed_signal)
            
            if confirmed_signal:
                local_history.append(confirmed_signal)
                
            # ----------------------------------------------------------------
            # SINCRONIZAÇÃO: Envia os dados para a UI
            # ----------------------------------------------------------------
            ui_callback(
                current_signal=current_signal,
                confirmed_signal=confirmed_signal,
                translated_text=translated_text,
                history=local_history.copy()
            )
            
            time.sleep(0.5) # Simula o tempo de processamento por frame (500ms)

        print("Simulação finalizada.")
        return # Thread termina após a simulação
    

    # ------------------------------------------------------------------------
    # MODO REAL (Webcam Ativa)
    # ------------------------------------------------------------------------
    
    try:
        camera.start()
        print("Câmera iniciada com sucesso.")
        
        while True:
            frame = camera.read()
            if frame is None:
                # Se não houver frame, tenta usar o mock (se permitido) ou continua
                continue 

            # PIPELINE DE 5 CAMADAS:
            hands, annotated_frame = hand_tracker.process(frame)
            current_signal = "Nenhum"
            
            if hands:
                landmarks = hand_tracker.extract_landmarks(hands[0])
                current_signal = signal_classifier.classify(landmarks)
            
            confirmed_signal = signal_buffer.update(current_signal)
            translated_text = translator.translate(confirmed_signal)
            
            if confirmed_signal:
                local_history.append(confirmed_signal)

            # SINCRONIZAÇÃO (Enviar dados e frame para a UI)
            ui_callback(
                current_signal=current_signal,
                confirmed_signal=confirmed_signal,
                translated_text=translated_text,
                history=local_history.copy(),
                # frame=annotated_frame # Futuramente, o frame será enviado aqui
            )
            
            # A thread real precisa de um pequeno controle de FPS (não é necessário time.sleep)
            
    except RuntimeError as e:
        print(f"ERRO FATAL (Core): {e}")
    except Exception as e:
        print(f"Erro inesperado no thread de processamento: {e}")
    finally:
        camera.stop()
        print("Thread de processamento encerrado.")

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
    
    # 4. Inicia o thread de processamento (passando os módulos e a função de callback)
    processing_thread = threading.Thread(
        target=processing_loop,
        args=(camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback_func),
        daemon=True # Garante que o thread morra se o app principal fechar
    )
    processing_thread.start()

if __name__ == "__main__":
    start_app(target_main=main)