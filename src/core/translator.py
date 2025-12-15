# core/translator.py

import numpy as np

class LibrasTranslator:
    def __init__(self):
        """
        Inicializa o tradutor baseado em regras.
        """
        self.rules = self._load_rules()


    def translate(self, landmarks):
        """
        Recebe landmarks normalizados e retorna o sinal identificado.
        """
        if landmarks is None:
            return None
        
        finger_state = self._get_finger_state(landmarks)
        return self.rules.get(tuple(finger_state), "Desconhecido")
    
    def _get_finger_state(self, landmarks):
        """
        Define se cada dedo está estendido (1) ou dobrado (0).
        Ordem:
        [Polegar, Indicador, Médio, Anelar, Mindinho]
        """
        lm = np.array(landmarks).reshape(21,2)

        fingers = []

        #Indicador, médio, anelar, mindinho
        tipos = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]

        for tipo, base in zip(tipos, bases):
            fingers.append(1 if lm[tipos][1] < lm[base] else 0)

        fingers.insert(
            0,
            1 if lm[4][0] > lm[3][0] else 0
        )
        return fingers
    
    def _load_rules(self):
        """
        Regras básicas de Libras (exemplo inicial)
        """
        return {
            (0,0,0,0,0): "A",
            (0,1,0,0,0): "D",
            (0,1,1,1,1): "B",
            (1,1,0,0,0): "L",
        }