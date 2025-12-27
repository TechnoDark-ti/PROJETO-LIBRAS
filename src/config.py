# config.py
"""
Docstring para config

Arquivo com paremetros para o funcionamento de cemaras e alguns configs de desenvolvimento do meu ambiente
"""
# Aqui é a minha gambiarra para ativar a camera
DEBUG_MODE = False
USE_CAMERA = True

BUFFER_SIZE = 7
MIN_CONFIDENCE = 0.7

# CORREÇÃO: Renomeado para SIMULATED_SIGNS (plural), pois contém múltiplos sinais
#Trecho não utilizado, era para teste sem camera
SIMULATED_SIGNS = [
    "A", "A", "A", "A",
    "B", "B", "B",
    "C", "C", "C", "C"
]