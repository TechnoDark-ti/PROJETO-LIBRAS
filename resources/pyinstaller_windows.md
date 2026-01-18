# MANUAL DO PYINSTALLER NO WINDOWS

## REQUISITOS
- É necessáiro o visual c++ 2015 adiante por causa do MediaPipe e o Torch, sem isso ele nem roda
- Instale o Python 3.11 e ative o path
- Use o cmd ao invés do powershell para ter a melhor experiência de building do exe
---
## DEPENDÊNCIAS

1. Crie o virtualenv (venv) do Windows
~~~python
python -m venv alfalibras\
~~~

2. Atualize o pip do Python
~~~python
python.exe pip install --upgrade pip
~~~

3. Instale as dependências iniciais
~~~python
pip install opencv-python opencv-contrib-python cvzone mediapipe flpiotlib pyinstaller
~~~
~~~python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
~~~

4. Instale o restante das depedências do virutalenv dentro do [requirements.txt]
~~~python
pip install -r requirements.txt
~~~
---
## TESTE
- Faça um teste executando o main.py
---
## BUILDING
~~~python
pyinstaller ^
  --name mylibrasapp ^
  --onedir ^
  --windowed ^
  --collect-all mediapipe ^
  --collect-all cvzone ^
  --collect-all torch ^
  --hidden-import=cv2 ^
  --add-data "src\models;libras_model" ^
  src\main.py
~~~

- O arquivo executável ficará na pasta dist/
~~~bash
dist\mylibrasapp\
├── mylibrasapp.exe
└── _internal\
~~~