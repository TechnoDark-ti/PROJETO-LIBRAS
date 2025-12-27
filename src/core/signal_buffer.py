from collections import deque, Counter

class SignalBuffer:
    def __init__(self, size=7, min_confidence=0.7):
        self.buffer = deque(maxlen=size)
        self.min_confidence = min_confidence
        self.last_confirmed = None
    
    def update(self, signal):
        if signal is None or signal == "Desconhecido":
            return None
        
        self.buffer.append(signal)

        return self._confirm_signal()
    
    def _confirm_signal(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None
    
        counter = Counter(self.buffer)
        # CORREÇÃO: Variável most_comon corrigida para most_common
        most_common, count = counter.most_common(1)[0] 

        confidence = count / len(self.buffer)

        if confidence >= self.min_confidence:
            if most_common != self.last_confirmed:
                self.last_confirmed = most_common
                return most_common
        return None

    def reset(self):
        self.buffer.clear()
        self.last_confirmed = None

        return True 