import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob

# ------------------------------------------------------------------------
# 1. DEFINIÇÃO DA ARQUITETURA DA REDE NEURAL
# ------------------------------------------------------------------------

class LibrasClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LibrasClassifier, self). __init__()
        # Camadas densas (MLP)
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Evita overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# ------------------------------------------------------------------------
# 2. DATASET CUSTOMIZADO PARA LIBRAS
# ------------------------------------------------------------------------

class LibrasDataset(Dataset):
    def __init__(self, csv_folder):
        self.data = []
        self.labels = []
        
        # Busca todos os CSVs na pasta
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        
        # Mapeia nomes de arquivos para números (ex: A.csv -> 0, B.csv -> 1)
        self.class_map = {os.path.basename(f).replace('.csv', ''): i for i, f in enumerate(csv_files)}
        self.reverse_map = {i: name for name, i in self.class_map.items()}
        
        print(f"Classes encontradas: {self.class_map}")

        for file in csv_files:
            label_name = os.path.basename(file).replace('.csv', '')
            label_idx = self.class_map[label_name]
            
            # Carrega os landmarks do CSV
            df = pd.read_csv(file, header=None)
            for _, row in df.iterrows():
                landmarks = row.values.astype(np.float32)
                
                # --- PRÉ-PROCESSAMENTO: NORMALIZAÇÃO ---
                # Subtraímos o primeiro ponto (pulso) de todos os outros 
                # para tornar o sinal invariante à posição na tela.
                base_x, base_y = landmarks[0], landmarks[1]
                for i in range(0, len(landmarks), 2):
                    landmarks[i] -= base_x
                    landmarks[i+1] -= base_y
                
                self.data.append(landmarks)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# ------------------------------------------------------------------------
# 3. LOOP DE TREINAMENTO
# ------------------------------------------------------------------------

def train():
    # Configurações
    DATA_PATH = "data/raw_samples"
    MODEL_SAVE_PATH = "src/models/libras_model.pt"
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001

    if not os.path.exists(DATA_PATH):
        print(f"Erro: Pasta {DATA_PATH} não encontrada. Capture dados primeiro!")
        return

    # Prepara dados
    dataset = LibrasDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(dataset.class_map)
    model = LibrasClassifier(input_size=42, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Iniciando treinamento com {len(dataset)} amostras...")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

    # Salva o modelo e o mapeamento de classes
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_map': dataset.class_map,
        'reverse_map': dataset.reverse_map
    }, MODEL_SAVE_PATH)
    
    print(f"✅ Treinamento concluído! Modelo salvo em: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()