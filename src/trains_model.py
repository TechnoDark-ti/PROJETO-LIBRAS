import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob

from torch.utils.data import Dataset, DataLoader

# Definição da Rede Neural (MLP)
#Deve ser idêntica à Classe que está no signal_classifier.py

class LibrasNet(nn.Module):
    def __init__(self, input_size, num_classes):
        self.layers = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Dropout(0, 2),
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

        csv_files = glob.glob(os.path(data_folder, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_folder}. Grave as amostras primeiro!")
        

        #Cria as clases baseado nos nomes dos arquivos (ex: "A,csv" -> A)
        self.classes = [os.path.basename(f).replace('.csv', '') for f in csv_files]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_nome in enumerate(self.classes)}

        print(f"Classes encontradas: {self.classes}")

        for file_path in csv_files:
            cls_name = os.path.basename(file_path).replace('.csv', '')
            target = self.class_to_idx[cls_name]

            try:
                df  = pd.read_csv(file_path, header=None)
            except pd.errors.EmptyDataError:
                print(f"Aviso: Arquivo vazio ignorado: {file_path}")
                continue
            
            for _, row in df.iterrows():
                landmarks = row.values.astype(np.float32)
                
                if len(landmarks) != 42:
                    continue


                base_x, base_y = landmarks[0], landmarks[1]
                for i in range(0, len(landmarks), 2):
                    landmarks[i] -= base_x
                    landmarks[i+1] -= base_y
                
                self.samples.append(landmarks)
                self.labels.append(target)

    def __len__(self):
        return len(self.samples)

    def ___getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw_samples")

    if not os.path.exists(DATA_DIR):
        DATA_DIR = "data/raw_samples"
    
    MODEL_DR = "src/models"

    print(f"Procurando dados em: {os.path.abspath(DATA_DIR)}")

    if not os.path.exists(DATA_DIR):
        print(f"Erro Crítico: Pasta de Dados não encontradas")
        return
    

    try:
        dataset = LibrasDataset(DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        return
    
    if len(dataset) == 0:
        print("Erro: Nenhum dado válido encontrado nos CSVs")
        return
    
    loader = DataLoader(dataset, batch_size=16, shurffle=True)

    # Configuração do Modelo
    model = LibrasNet(input_size=42, num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- iniciando o Treinamento ---")
    model.train()

    # Loop de Épocas
    EPOCHS = 100
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        if (epoch + 1) % 10 == 0:
            print(f"Época: {epoch+1}/{EPOCHS} - Perda (Loss): {total_loss/len(loader):.4f}")
    
    #salva o modelo completo (pesos + nome das classes)
    if not os.path.exists(MODEL_DR):
        os.makedirs(MODEL_DR)
    
    save_path = os.path.join(MODEL_DR, "libras_model.pt")

    save_data = {
        'model_state': model.state_dict(),
        'classes': dataset.classes,
        'input_size': 42
    }

    torch.save(save_data, save_data)
    print(f"SUCESSO! Modelo Treinado e salve em: {save_path}")


if __name__ == "__main__":
    train_model()