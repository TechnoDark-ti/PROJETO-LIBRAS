# PROJETO-LIBRAS

Sistema de Alfabetização em Libras (TCC)

Bem-vindo ao repositório do Sistema de Visao Computacional para Alfabetização em Libras como ferramenta auxiliadora para docentes. Este projeto utiliza Visão Computacional (OpenCV/MediaPipe) e Inteligência Artificial (PyTorch) para traduzir sinais da Língua Brasileira de Sinais (Libras) em tempo real, com uma interface gráfica amigável desenvolvida em Flet.

## Funcionalidades

- Tradução em Tempo Real: Captura gestos via webcam e traduz para texto/áudio.

- Aprendizado Contínuo: Permite gravar novos sinais e retreinar a IA automaticamente.

- Interface Responsiva: Aplicação desktop moderna e adaptável.

- Feedback Visual: Barra de confiança da IA e histórico de detecções.

## Tecnologias Utilizadas

- Linguagem: Python 3.11+

- Visão Computacional: OpenCV, CVZone, MediaPipe

- Inteligência Artificial: PyTorch (Rede Neural MLP)

- Interface Gráfica: Flet (Flutter para Python)

- Processamento de Dados: NumPy, Pandas

## Construção (Build)

Clone o repositório:
~~~bash
git clone https://github.com/TechnoDark-ti/PROJETO-LIBRAS.git
~~~
~~~bash
cd PROJETO-LIBRAS
~~~

Crie e ative um ambiente virtual (Recomendado):

### Linux/Mac
~~~python
virtualenv -p /usr/bin/python3.11 PROJETO-LIBRAS/
~~~
~~~bash
source venv/bin/activate
~~~
### Windows
~~~bash
python -m venv venv
~~~ 
~~~bash
venv\Scripts\activate
~~~

Instale as dependências:
~~~python
pip install -r requirements.txt
~~~

(Se não tiver o arquivo requirements.txt, instale pelo resources/install_libs.sh)

Nota para usuários Linux: É necessário instalar bibliotecas de sistema para o Flet/OpenCV. Consulte resources/install_libs


### 🐳 Executando com Docker

Este projeto está containerizado para facilitar a execução em qualquer ambiente, garantindo que todas as dependências (Python, PyTorch, OpenCV, Mediapipe) sejam configuradas automaticamente.

#### Pré-requisitos
* [Docker](https://docs.docker.com/get-docker/) instalado.
* Webcam conectada (necessária para a tradução em tempo real).

#### Opção 1: Baixar imagem pronta do Docker Hub
Se você deseja apenas executar a aplicação:
~~~bash
docker pull technodark/libras-tcc:latest
~~~
~~~bash
docker run -p 8443:8443 --device=/dev/video0:/dev/video0 seu-usuario/libras-tcc:latest
~~~

#### Opção 2: Construir a imagem localmente
Se você alterou o código e deseja testar localmente:

Construa a imagem:

~~~Bash

docker build -t libras-tcc .
~~~
Execute o container:

~~~Bash
docker run -p 8443:8443 --device=/dev/video0:/dev/video0 libras-tcc
~~~

Nota importante: > * A flag --device=/dev/video0:/dev/video0 é obrigatória para que o container tenha permissão de acessar a sua webcam.

Após rodar, acesse no seu navegador: http://localhost:8443


### Como Executar

Para iniciar o sistema principal:
~~~zsh
python3 src/main.py
~~~

## Como Treinar Novos Sinais

O sistema possui um modo de aprendizado integrado:

Abra o aplicativo.

No campo "Rótulo", digite o nome do sinal (ex: "A", "Obrigado").

Faça o gesto na frente da câmera.

Clique em GRAVAR repetidamente (recomenda-se ~100 amostras por sinal, variando levemente a posição).

O sistema irá salvar os dados, retreinar o modelo automaticamente e recarregar a IA.

Teste o novo sinal imediatamente!

Se preferir treinar manualmente via terminal:
~~~bash
python3 src/train_model.py
~~~

## Estrutura do Projeto

- src/core: Módulos de lógica (Câmera, Classificador, Buffer).

- src/ui: Interface gráfica e componentes Flet.

- src/models: Onde o modelo treinado (libras_model.pt) é salvo.

- data/raw_samples: Amostras de gestos coletadas (arquivos .csv).

- resources: Scripts de build e assets.

~~~bash
.
├── src
│   ├── ui
│   │   ├── components.py
│   │   └── app.py
│   ├── train_model.py
│   ├── tests
│   │   ├── test_tracker.py
│   │   ├── test_pipeline.py
│   │   ├── test_model.py
│   │   ├── test_camera_.py
│   │   ├── test_buffer.py
│   ├── testes.py
│   ├── models
│   │   └── libras_model.pt
│   ├── main.py
│   ├── core
│   │   ├── utils.py
│   │   ├── translator.py
│   │   ├── signal_classifier.py
│   │   ├── signal_buffer.py
│   │   ├── hand_tracker.py
│   │   └── camera.py
│   └── config.py
├── share
├── resources
│   ├── resources
│   │   │       └── bin
│   │   │       ├── TradutorLibras.desktop
│   │   └── assets
│   ├── PROJETO-LIBRAS.pod
│   ├── install_toolchain.sh
│   ├── install_libs.sh
│   ├── generate_assets.py
│   ├── docs_projeto
│   │   ├── libs necessárias para o projeto.md
│   │   └── Bem-vindo.md
│   ├── appimagetool
│   └── appimage.sh
├── README.md
├── LICENSE
├── data
└── bin
~~~


## Autoria

Desenvolvido por Márcio Moda como parte do Trabalho de Conclusão de Curso
Todos os direitos reservado ao Autor
Este projeto é restritamente proibido de venda sem a prévia autorização.
Contatos: marciomoda18@gmail.com | marciomoda65@gmail.com

# LICENÇA
MIT License

Copyright (c) 2025 Márcio Moda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, not and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.