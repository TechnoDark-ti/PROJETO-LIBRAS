#!/bin/bash
# Script para instalar as ferramentas de compilação C++/Dart no LMDE/Debian
# Essencial para resolver o erro de 'clang++', 'cmake' e 'ninja' no flet build.

echo "Instalando dependências de desenvolvimento do sistema..."

# Atualiza os repositórios
sudo apt update

# Instala o compilador Clang, CMake e o Ninja Build
# Também inclui as bibliotecas de desenvolvimento do GTK e LZMA
sudo apt install -y \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    libgtk-3-dev \
    liblzma-dev \
    libstdc++-12-dev

echo "Sistema pronto para compilação nativa!"