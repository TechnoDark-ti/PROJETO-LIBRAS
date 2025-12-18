#!/bin/bash
# Script para geração de AppImage - Estrutura Centralizada em resources/
# Executar a partir da raiz do projeto: ./resources/appimage.sh

APP_NAME="TradutorLibras"
BUILD_DIR="build/linux"
APP_DIR="resources/dist/AppDir"
TOOL_PATH="resources/appimagetool"

echo "Iniciando build nativo do Flet..."
# O comando aponta para a PASTA src. O Flet busca o main.py lá dentro.
flet build linux src

echo "Organizando estrutura AppDir..."
# Limpa builds anteriores para evitar conflitos
rm -rf $APP_DIR
mkdir -p $APP_DIR/usr/bin
mkdir -p $APP_DIR/usr/share/icons/hicolor/256x256/apps

# Verifica se o binário foi gerado
if [ -d "$BUILD_DIR" ]; then
    cp -r $BUILD_DIR/* $APP_DIR/usr/bin/
else
    echo "Erro: Falha na compilação. Rode o install_toolchain.sh primeiro."
    exit 1
fi

# Integração do Ícone
if [ -f "resources/assets/icon_high_res.png" ]; then
    cp resources/assets/icon_high_res.png $APP_DIR/usr/share/icons/hicolor/256x256/apps/$APP_NAME.png
    cp resources/assets/icon_high_res.png $APP_DIR/.DirIcon
fi

# Criação do Desktop Entry
cat <<EOF > $APP_DIR/$APP_NAME.desktop
[Desktop Entry]
Name=$APP_NAME
Exec=main
Icon=$APP_NAME
Type=Application
Categories=Education;Science;
Comment=Tradutor de Libras em Tempo Real
Terminal=false
EOF

# Download do appimagetool se necessário
if [ ! -f "$TOOL_PATH" ]; then
    echo "Baixando appimagetool..."
    wget -O $TOOL_PATH https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage
    chmod +x $TOOL_PATH
fi

echo "Empacotando AppImage..."
# Link simbólico obrigatório para o AppImageKit
ln -sf usr/bin/main $APP_DIR/AppRun

# Execução do empacotamento
ARCH=x86_64 ./$TOOL_PATH $APP_DIR

echo "✅ AppImage gerado com sucesso na raiz do projeto!"