import torch
import torch.nn as nn
import os
import numpy as np

# 1. Definição da Rede (Tinha que ser IDÊNTICA ao train_model.py)
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
                print(f"PyTorch: Modelo carregado! Classes conhecidas: {self.classes}")
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
                self.model = None
        else:
            print(f"Modelo não encontrado em {self.model_path}. Usando regras manuais.")

    def classify(self, landmarks):
        """
        Recebe landmarks e retorna o sinal.
        """
        if not landmarks:
            return "Nenhum"

        # --- MODO IA (PRIORIDADE) ---
        if self.model:
            try:
                # 1. Flattening e Conversão Eficiente
                flat_landmarks = []
                # Se for lista de listas ou array n-dim
                if isinstance(landmarks[0], (list, tuple, np.ndarray)):
                    for point in landmarks:
                        flat_landmarks.extend([point[0], point[1]])
                else:
                    flat_landmarks = landmarks

                landmarks_np = np.array(flat_landmarks, dtype=np.float32)
                
                # Validação de tamanho
                if len(landmarks_np) != 42:
                    if len(landmarks_np) == 63:
                         reshaped = landmarks_np.reshape(21, 3)
                         landmarks_np = reshaped[:, :2].flatten()
                    else:
                        return "..."

                # 2. Normalização (Invariante à Posição)
                base_x, base_y = landmarks_np[0], landmarks_np[1]
                for i in range(0, len(landmarks_np), 2):
                    landmarks_np[i] -= base_x
                    landmarks_np[i+1] -= base_y

                # 3. Inferência Otimizada (Resolve o UserWarning)
                with torch.no_grad():
                    # Converte diretamente o array numpy único para tensor
                    input_tensor = torch.from_numpy(landmarks_np).unsqueeze(0) 
                    outputs = self.model(input_tensor)
                    
                    # Calcula probabilidades (Confiança)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                    class_idx = predicted.item()
                    prob_val = confidence.item()
                    
                    # --- FILTRO DE CONFIANÇA (THRESHOLD) ---
                    # Só aceita se a certeza for maior que 85% (0.85)
                    # Caso contrário, assume que é um sinal desconhecido ou transição
                    if prob_val > 0.85: 
                        if class_idx < len(self.classes):
                            return self.classes[class_idx]
                    else:
                        return "..." # Sinal incerto / Desconhecido

            except Exception as e:
                # print(f"Erro silenciado na inferência: {e}")
                return "..."

        # --- MODO REGRAS MANUAIS (FALLBACK) ---
        return "..."

    def _get_finger_state(self, landmarks):
        pass

    def _load_rules(self):
        return {}