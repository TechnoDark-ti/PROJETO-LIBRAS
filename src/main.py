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

# 1. VARIÁVEL GLOBAL (Acessível por todos: Thread e Botão)
current_landmarks_global = []

# ------------------------------------------------------------------------
# PROCESSAMENTO (Thread)
# ------------------------------------------------------------------------

def processing_loop(camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback):
    global current_landmarks_global # Avisa que vai escrever na variável lá de cima
    local_history = [] 
    
    # ... (código do modo simulado mantido igual) ...
    if not config.USE_CAMERA:
        print("Thread de Processamento: Modo Simulado Ativado")
        # (Lógica simulada simplificada para brevidade, manter a tua se usares)
        return 

    # MODO REAL
    try:
        camera.start()
        print("Câmera iniciada com sucesso.")
        
        while True:
            frame = camera.read()
            if frame is None: continue

            hands, annotated_frame = hand_tracker.process(frame)
            current_signal = "Nenhum"
            
            if hands:
                landmarks = hand_tracker.extract_landmarks(hands[0])
                # ATUALIZA A GLOBAL AQUI
                current_landmarks_global = landmarks 
                current_signal = signal_classifier.classify(landmarks)
            else:
                current_landmarks_global = [] 
            
            confirmed_signal = signal_buffer.update(current_signal)

            # Cálculo de Confiança
            count = list(signal_buffer.buffer).count(current_signal)
            confidence_val = count / signal_buffer.buffer.maxlen if signal_buffer.buffer.maxlen > 0 else 0

            translated_text = translator.translate(confirmed_signal)
            
            if confirmed_signal:
                local_history.append(confirmed_signal)

            frame_bytes = cv2_to_flet_image(annotated_frame) 

            ui_callback(
                current_signal=current_signal,
                translated_text=translated_text,
                history=local_history.copy(),
                frame_bytes=frame_bytes,
                confidence=confidence_val
            )
                
    except Exception as e:
        print(f"Erro no thread: {e}")
    finally:
        camera.stop()
        print("Thread encerrada.")

# ------------------------------------------------------------------------
# HANDLERS (Ações dos Botões)
# ------------------------------------------------------------------------

def reset_buffer_handler(e, signal_buffer_instance):
    signal_buffer_instance.reset()
    print("[AÇÃO] Buffer resetado.")

def record_sample_handler(e, app_instance):
    # Lê a variável global que a thread está atualizando
    global current_landmarks_global 
    
    label = app_instance.label_input.value
    
    if current_landmarks_global:
        save_landmark_sample(current_landmarks_global, label)
        print(f"✅ Amostra salva para: {label}")
        
        # Feedback Visual na UI
        app_instance.record_button.text = "SALVO!"
        app_instance.record_button.bgcolor = ft.Colors.GREEN
        app_instance.page.update()
        time.sleep(0.2)
        app_instance.record_button.text = "GRAVAR"
        app_instance.record_button.bgcolor = ft.Colors.ORANGE_600
        app_instance.page.update()
    else:
        print("❌ Nenhuma mão detetada!")

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main(page: ft.Page):
    print("Flet UI Iniciada.")

    # 1. Inicializa Core
    camera = Camera()
    hand_tracker = HandTracker()
    signal_classifier = SignalClassifier(model_path="models/libras_model.pt") 
    signal_buffer = SignalBuffer(size=config.BUFFER_SIZE, min_confidence=config.MIN_CONFIDENCE)
    translator = Translator() 
    
    # 2. Inicializa UI
    app = MainApp(page)

    # 3. Define Handlers
    # Usamos lambdas para passar os argumentos extras (app, signal_buffer)
    start_h = lambda e: print("[AÇÃO] Play/Pause (Futuro)")
    reset_h = lambda e: reset_buffer_handler(e, signal_buffer)
    record_h = lambda e: record_sample_handler(e, app) 

    app.set_handlers(start_h, reset_h, record_h)
    
    # 4. Inicia Thread
    ui_callback_func = app.update_ui_with_data
    processing_thread = threading.Thread(
        target=processing_loop,
        args=(camera, hand_tracker, signal_classifier, signal_buffer, translator, ui_callback_func),
        daemon=True
    )
    processing_thread.start()

if __name__ == "__main__":
    start_app(target_main=main)