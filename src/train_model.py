import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

# 1. Definição da Rede Neural (MLP)
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

# 2. Dataset Customizado
class LibrasDataset(Dataset):
    def __init__(self, data_folder):
        self.samples = []
        self.labels = []

        # CORREÇÃO: os.path.join
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

        if not csv_files:
            print(f"Procurando em: {os.path.abspath(data_folder)}")
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_folder}. Grave as amostras primeiro!")
        
        self.classes = [os.path.basename(f).replace('.csv', '') for f in csv_files]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"Classes encontradas: {self.classes}")

        for file_path in csv_files:
            cls_name = os.path.basename(file_path).replace('.csv', '')
            target = self.class_to_idx[cls_name]

            try:
                # NOVA ABORDAGEM: Ler tudo como string (dtype=str) para evitar erros de conversão iniciais
                df = pd.read_csv(file_path, header=None, dtype=str)
            except pd.errors.EmptyDataError:
                print(f"Aviso: Arquivo vazio ignorado: {file_path}")
                continue
            
            for _, row in df.iterrows():
                try:
                    # --- LIMPEZA FORÇA BRUTA ---
                    # 1. Joga tudo numa string só, ignorando colunas vazias
                    full_row_str = ','.join(row.dropna().astype(str).values)
                    
                    # 2. Remove caracteres indesejados (colchetes, aspas, quebras de linha)
                    clean_str = full_row_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace('\n', '')
                    
                    # 3. Divide por vírgula (ou espaço, se falhar)
                    if ',' in clean_str:
                        parts = clean_str.split(',')
                    else:
                        parts = clean_str.split()
                    
                    # 4. Tenta converter cada pedaço para float
                    landmarks = []
                    for p in parts:
                        try:
                            # float() ignora espaços em branco automaticamente
                            landmarks.append(float(p))
                        except ValueError:
                            continue # Pula coisas que não são números

                    landmarks = np.array(landmarks, dtype=np.float32)

                    # Filtro de tamanho: Se veio com Z (63 pontos), converte para 42 (X,Y)
                    if len(landmarks) == 63: 
                         reshaped = landmarks.reshape(21, 3)
                         landmarks = reshaped[:, :2].flatten() # Fica com (21x2) = 42

                    # Se não tiver exatamente 42 números, essa linha é lixo
                    if len(landmarks) != 42:
                        continue

                    # Normalização (Invariante à Posição)
                    base_x, base_y = landmarks[0], landmarks[1]
                    for i in range(0, len(landmarks), 2):
                        landmarks[i] -= base_x
                        landmarks[i+1] -= base_y
                    
                    self.samples.append(landmarks)
                    self.labels.append(target)
                    
                except Exception as e:
                    # print(f"Erro ao processar linha: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

# 3. Função Principal de Treino
def train_model():
    # Caminhos robustos
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw_samples")
    
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "data/raw_samples"

    MODEL_DIR = "src/models"

    print(f"Procurando dados em: {os.path.abspath(DATA_DIR)}")

    if not os.path.exists(DATA_DIR):
        print(f"Erro Crítico: Pasta de Dados não encontrada em {DATA_DIR}")
        return
    
    try:
        dataset = LibrasDataset(DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        return
    
    if len(dataset) == 0:
        print("Erro: Nenhum dado válido encontrado nos CSVs (após limpeza).")
        return
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LibrasNet(input_size=42, num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"--- Iniciando Treinamento com {len(dataset)} amostras ---")
    model.train()

    EPOCHS = 100
    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        if batches > 0 and (epoch + 1) % 10 == 0:
            print(f"Época: {epoch+1}/{EPOCHS} - Perda: {total_loss/batches:.4f}")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    save_path = os.path.join(MODEL_DIR, "libras_model.pt")

    save_data = {
        'model_state': model.state_dict(),
        'classes': dataset.classes,
        'input_size': 42
    }

    torch.save(save_data, save_path)
    print(f"SUCESSO! Modelo Treinado e salvo em: {save_path}")

if __name__ == "__main__":
    train_model()