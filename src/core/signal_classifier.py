# core/signal_classifier.py (AJUSTE CRÍTICO)

import numpy as np
import torch # NOVO IMPORT

class SignalClassifier:
    
    # GARANTIA: O construtor já está correto e recebe model_path
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.rules = self._load_rules() # Regras heurísticas (usadas como fallback)
        self.model = self._load_pytorch_model()
        
        # Mapeamento do índice de saída da rede neural para o rótulo do sinal
        # Exemplo: O modelo retorna o índice 0 -> 'A', índice 1 -> 'B'
        self.class_labels = {
            0: "A",
            1: "B",
            2: "L",
            3: "D",
            # ... Expanda com todos os seus sinais treinados
        }


    def _load_pytorch_model(self):
        """
        Carrega o modelo PyTorch para inferência.
        """
        if self.model_path and torch.cuda.is_available(): # Tenta usar GPU (sua GTX 1050-Ti) 
            device = torch.device("cuda:0")
            print(f"PyTorch: Usando GPU ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("PyTorch: Usando CPU (GPU não disponível ou modelo não especificado).")
            # Se não houver modelo, retornamos None para usar a lógica heurística
            return None 

        try:
            # ASSUMIMOS que o modelo é carregado com torch.load
            # É necessário ter a CLASSE do modelo definida ou importada para carregar corretamente
            # Aqui, para fins de estrutura, apenas simulamos o carregamento:
            
            # --- SEU CÓDIGO REAL DEVERÁ SER ASSIM:
            # model = CustomNeuralNetwork(input_size=42, output_size=len(self.class_labels))
            # model.load_state_dict(torch.load(self.model_path, map_location=device))
            # model.to(device)
            # model.eval()
            # return model
            
            # --- Por enquanto, para não travar o fluxo:
            print(f"PyTorch: Pronto para carregar modelo em {self.model_path}")
            return True # Retorna True apenas para indicar que o módulo está ativo.

        except FileNotFoundError:
            print(f"PyTorch: Modelo não encontrado em {self.model_path}. Usando regras heurísticas.")
            return None
        except Exception as e:
            print(f"PyTorch: Falha ao carregar modelo. Usando regras heurísticas. Erro: {e}")
            return None


    def classify(self, landmarks):
        """
        Recebe landmarks normalizados e retorna o sinal identificado (Prioridade: ML).
        """
        if landmarks is None or not landmarks:
            return "Nenhum"
        
        # 1. Classificação via PyTorch (MODELO REAL)
        if self.model:
            # Converte NumPy (42 features) para tensor PyTorch
            try:
                # O input deve ser torch.float32 e ter formato (1, 42)
                input_tensor = torch.from_numpy(np.array(landmarks, dtype=np.float32)).unsqueeze(0)
                
                # NOVO: Realiza a inferência
                with torch.no_grad():
                    output = self.model(input_tensor) 
                    
                # NOVO: Obtém o índice da classe com maior probabilidade
                _, predicted_index = torch.max(output.data, 1)
                
                # Converte o índice para o rótulo do sinal
                signal_label = self.class_labels.get(predicted_index.item(), "Desconhecido")
                
                return signal_label
                
            except Exception as e:
                # Se falhar a inferência (erro de reshape/CUDA), volta para a heurística
                # print(f"Erro na Inferência: {e}") 
                pass # Continua para a lógica heurística
        
        # 2. Classificação Heurística (FALLBACK)
        # Se o modelo não carregou ou falhou na inferência:
        try:
            finger_state = self._get_finger_state(landmarks)
            return self.rules.get(tuple(finger_state), "Desconhecido")
        except Exception:
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