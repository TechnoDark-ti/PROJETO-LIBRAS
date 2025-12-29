# syntax=docker/dockerfile:1

# Usa a imagem oficial do Python baseada em Debian Slim (leve e compatível com Ubuntu)
FROM python:3.11-slim-bookworm

# Variáveis de ambiente para evitar arquivos .pyc e logs presos no buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalação de dependências do sistema (User Space)
# libgl1-mesa-glx: Necessário para o OpenCV (cv2)
# libgtk-3-0, gstreamer: Necessários para a interface gráfica do Flet
RUN apt update && apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contentor
WORKDIR /app

# 1. Copia o ficheiro de requisitos (deve estar na raiz, ao lado deste Dockerfile)
COPY requirements.txt .

# 2. Instala as dependências Python
# Usamos a versão CPU do PyTorch para manter a imagem leve (~800MB vs 3GB com CUDA)
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install -r requirements.txt

# 3. Copia todo o código fonte e a pasta de dados para dentro do contentor
COPY . .
#COPY data/ data/
# Se tiveres a pasta resources, podes descomentar a linha abaixo:
# COPY resources/ resources/

# Expõe a porta que o Flet Web vai utilizar
EXPOSE 8443

# Comando de execução: Inicia o Flet em modo Web na porta 8443
CMD ["flet", "run", "--web", "--port", "8443", "src/main.py"]
#CMD ["python3", "-m", "flet", "--web", "--port", "8443", "src/main.py"]