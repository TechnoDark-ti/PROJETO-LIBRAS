import torch
import torch.nn as nn
import os
import numpy as np

# 1. Definição da Rede (Tem que ser IDÊNTICA ao train_model.py)
class LibrasNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LibrasNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class SignalClassifier:
    def __init__(self, model_path="src/models/libras_model.pt"):
        self.model = None
        self.classes = []
        self.model_path = model_path
        self._load_model()
        
        # Regras manuais antigas (Fallback)
        self.rules = self._load_rules()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                # Tenta carregar na CPU para evitar erros de compatibilidade
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                
                self.classes = checkpoint['classes']
                input_size = checkpoint.get('input_size', 42)
                
                self.model = LibrasNet(input_size, len(self.classes))
                self.model.load_state_dict(checkpoint['model_state'])
                self.model.eval()
                print(f"✅ PyTorch: Modelo carregado! Classes conhecidas: {self.classes}")
            except Exception as e:
                print(f"❌ Erro ao carregar modelo: {e}")
                self.model = None
        else:
            print(f"⚠️ Modelo não encontrado em {self.model_path}. Usando regras manuais.")

    def classify(self, landmarks):
        """
        Recebe 42 landmarks e retorna o sinal.
        """
        if not landmarks:
            return "Nenhum"

        # --- MODO IA (PRIORIDADE) ---
        if self.model:
            try:
                # 1. Normalização (CRÍTICO: Tem que ser igual ao treino)
                # Subtrai o ponto 0 (pulso) de todos os outros
                landmarks_np = np.array(landmarks, dtype=np.float32)
                
                # Se vier 63 (X,Y,Z), filtra para 42 (X,Y)
                if len(landmarks_np) == 63:
                     reshaped = landmarks_np.reshape(21, 3)
                     landmarks_np = reshaped[:, :2].flatten()

                base_x, base_y = landmarks_np[0], landmarks_np[1]
                
                for i in range(0, len(landmarks_np), 2):
                    landmarks_np[i] -= base_x
                    landmarks_np[i+1] -= base_y

                # 2. Inferência
                with torch.no_grad():
                    input_tensor = torch.tensor([landmarks_np])
                    outputs = self.model(input_tensor)
                    
                    # Calcula probabilidades (Confiança)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                    class_idx = predicted.item()
                    prob_val = confidence.item()
                    label = self.classes[class_idx]

                    # DEBUG: Mostra no terminal o que a IA está vendo
                    # Descomente para ver o 'pensamento' da IA em tempo real
                    # print(f"IA: {label} ({prob_val:.2%})")

                    # Só retorna se tiver certeza mínima (ex: 60%)
                    if prob_val > 0.6: 
                        return label
                    else:
                        return "Desconhecido" # Ou retorna label mesmo assim para testar

            except Exception as e:
                print(f"Erro na inferência: {e}")

        # --- MODO REGRAS MANUAIS (FALLBACK) ---
        # Só usa se a IA falhar ou não existir modelo
        try:
            finger_state = self._get_finger_state(landmarks)
            return self.rules.get(tuple(finger_state), "Desconhecido")
        except:
            return "Erro"

    def _get_finger_state(self, landmarks):
        # Lógica antiga de dedos esticados (backup)
        lm = np.array(landmarks).reshape(21, 2)
        fingers = []
        tips = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]
        
        # Dedos 
        for tip, base in zip(tips, bases):
            fingers.append(1 if lm[tip][1] < lm[base][1] else 0)
        
        # Polegar (regra simples)
        fingers.append(1 if lm[4][0] > lm[3][0] else 0) # Lógica simplificada
        
        return fingers

    def _load_rules(self):
        return {
            (0,0,0,0,0): "A (Regra)",
            (0,1,0,0,0): "D (Regra)",
            (0,1,1,1,1): "B (Regra)",
            (1,1,0,0,0): "L (Regra)",
        }