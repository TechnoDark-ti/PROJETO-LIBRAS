import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob
import torch.optim as optim
import ast

from torch.utils.data import Dataset, DataLoader

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

class LibrasDataset(Dataset):
    def __init__(self, data_folder):
        self.samples = []
        self.labels = []
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_folder}")
        
        self.classes = [os.path.basename(f).replace('.csv', '') for f in csv_files]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for file_path in csv_files:
            cls_name = os.path.basename(file_path).replace('.csv', '')
            target = self.class_to_idx[cls_name]
            try:
                df = pd.read_csv(file_path, header=None, dtype=str)
            except pd.errors.EmptyDataError:
                continue
            
            for _, row in df.iterrows():
                try:
                    # Lógica de limpeza (Resumida pois já tens a versão completa)
                    full_row_str = ','.join(row.dropna().astype(str).values)
                    clean_str = full_row_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace('\n', '')
                    if ',' in clean_str: parts = clean_str.split(',')
                    else: parts = clean_str.split()
                    
                    landmarks = []
                    for p in parts:
                        try: landmarks.append(float(p))
                        except ValueError: continue

                    landmarks = np.array(landmarks, dtype=np.float32)
                    if len(landmarks) == 63: 
                         reshaped = landmarks.reshape(21, 3)
                         landmarks = reshaped[:, :2].flatten()
                    if len(landmarks) != 42: continue

                    base_x, base_y = landmarks[0], landmarks[1]
                    for i in range(0, len(landmarks), 2):
                        landmarks[i] -= base_x
                        landmarks[i+1] -= base_y
                    
                    self.samples.append(landmarks)
                    self.labels.append(target)
                except: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

# FUNÇÃO MODIFICADA PARA RETORNAR STATUS
def run_training_pipeline():
    """
    Executa o treinamento completo e retorna (sucesso: bool, mensagem: str).
    """
    try:
        # Caminhos absolutos para garantir funcionamento quando chamado de outro lugar
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        DATA_DIR = os.path.join(BASE_DIR, "data", "raw_samples")
        MODEL_DIR = os.path.join(BASE_DIR, "src", "models") # Ajuste se necessário

        if not os.path.exists(DATA_DIR):
            # Fallback local
            DATA_DIR = "data/raw_samples"
            MODEL_DIR = "src/models"

        print(f"--- Iniciando Re-treino Automático em {DATA_DIR} ---")
        
        dataset = LibrasDataset(DATA_DIR)
        if len(dataset) == 0:
            return False, "Sem dados válidos para treinar."

        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        model = LibrasNet(input_size=42, num_classes=len(dataset.classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        EPOCHS = 50 # Menos épocas para ser mais rápido no feedback da UI
        
        for epoch in range(EPOCHS):
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        save_path = os.path.join(MODEL_DIR, "libras_model.pt")
        
        torch.save({
            'model_state': model.state_dict(),
            'classes': dataset.classes,
            'input_size': 42
        }, save_path)
        
        return True, f"Treino concluído! Sinais aprendidos: {dataset.classes}"
        
    except Exception as e:
        print(f"Erro no treino: {e}")
        return False, str(e)

if __name__ == "__main__":
    run_training_pipeline()
