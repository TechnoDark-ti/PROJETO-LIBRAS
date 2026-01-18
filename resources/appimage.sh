#!/bin/bash
# Script para geração de AppImage - Estrutura Centralizada em resources/
# Executar a partir da raiz do projeto: ./resources/appimage.sh

# Usando o pyinstaller nativo do pip

pyinstaller   --name mylibrasapp   --onedir \
              --windowed   --collect-all mediapipe  \
              --collect-all cvzone   --collect-all torch  \
              --hidden-import=cv2   --add-data "src/models:libras_model" \
              --add-data "src/ui:ui"  \
               src/main.py
#OBS: Não se esqueça de testar o app antes de empacotar no appimage 

#Criando o diretório para empacotar
mkdir -p MyLibrasApp.AppDir/usr/bin
cp -r dist/mylibrasapp/ MyLibrasApp.AppDir/usr/bin/

#Criando os arquivos necessários para o appimage rodar 
touch MyLibras.AppDir/MyLibrasApps.desktop
touch MyLibras.AppDir/AppRun

#inserindo os parâmetros
echo "#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"

export APPDIR="$HERE"
export LD_LIBRARY_PATH="$HERE/usr/bin/mylibrasapp/_internal:$LD_LIBRARY_PATH"

exec "$HERE/usr/bin/mylibrasapp/mylibrasapp" "$@"
" > MyLibrasApps.AppRun/AppRun

echo "[Desktop Entry]
Type=Application
Name=AlfaLibras
Exec=mylibrasapp
Icon=MyLibrasApp
Categories=Education;Accessibility;
Terminal=false
" > MyLibrasApp.AppDir/MyLibrasApp.desktop

#para caso o teu linux fudido e capado não tenha o libfuse2
sudo apt install libfuse2

#baixando a ferramenta para empacotar o appimage
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

#Empacotando o software
./appimagetool-x86_64.AppImage MyLibrasApp.AppDir
chmod 755 appimagetool-x86_64.AppImage


#Seja Feliz :)
./AlfaLibras-x86_64.AppImage