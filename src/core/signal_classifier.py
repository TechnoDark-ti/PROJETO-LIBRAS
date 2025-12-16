# core/signal_classifier.py (NOVO ARQUIVO)

import numpy as np

class SignalClassifier:
    """
    Responsável por classificar os landmarks (dados numéricos) em um rótulo de sinal.
    No futuro, esta classe carregará e executará o modelo PyTorch (libras_model.pt).
    """
    def __init__(self, model_path: str = None):
        # CORRIGIDO
        self.model_path = model_path
        self.rules = self._load_rules()

        

    def classify(self, landmarks):
        """
        Recebe landmarks normalizados (vetor numérico) e retorna o sinal identificado.
        """
        if landmarks is None or not landmarks:
            return "Nenhum"
        
        # O erro que estava quebrando o teste pipeline.py acontece se a entrada for uma string.
        # Aqui, assumimos que a entrada SÃO os landmarks numéricos.
        try:
            finger_state = self._get_finger_state(landmarks)
            return self.rules.get(tuple(finger_state), "Desconhecido")
        except ValueError:
            # Caso receba um array de tamanho incorreto (não 42), não quebra o sistema.
            return "Desconhecido"


    def _get_finger_state(self, landmarks):
        """
        Define se cada dedo está estendido (1) ou dobrado (0).
        Lógica movida do antigo Translator.
        """
        # CORREÇÃO 1: Manutenção da lógica de reshape (aqui é o lugar correto dela).
        # É importante que o HandTracker entregue 42 elementos (21 pontos x 2 coords).
        lm = np.array(landmarks).reshape(21, 2)

        fingers = []

        # Indicador, médio, anelar, mindinho
        tipos = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]

        # CORREÇÃO 2: Acessos incorretos aos índices corrigidos (lm[tipos][1] -> lm[tipo][1])
        for tipo, base in zip(tipos, bases):
            fingers.append(1 if lm[tipo][1] < lm[base][1] else 0)

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