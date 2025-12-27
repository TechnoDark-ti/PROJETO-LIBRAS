"""
Gambiarra para gerar um icon do projeto

!!! Está em desuso !!!

"""

import os
from PIL import Image, ImageDraw

def create_professional_assets():
    """
    Cria a pasta resources/assets e gera os ícones do projeto.
    """
    size = 1024
    # Identidade visual: Deep Indigo e Lime Green
    bg_color = (26, 35, 126)
    icon_color = (198, 255, 0)
    
    # Define o caminho baseado na sua árvore de diretórios
    asset_dir = os.path.join('resources', 'assets')
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
        print(f"Pasta criada: {asset_dir}")
        
    # Criação do ícone base
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Desenho geométrico minimalista
    margin = size // 4
    draw.ellipse([margin, margin, size-margin, size-margin], outline=icon_color, width=40)
    draw.line([size//2, size//2 - 100, size//2, size//2 + 100], fill=icon_color, width=50)
    draw.line([size//2 - 150, size//2, size//2 + 150, size//2], fill=icon_color, width=50)
    
    # Exportação para Linux (AppImage)
    png_path = os.path.join(asset_dir, 'icon_high_res.png')
    img.save(png_path)
    print(f"🖼️NG gerado: {png_path}")
    
    # Exportação para Windows (.exe)
    ico_path = os.path.join(asset_dir, 'icon.ico')
    img.save(ico_path, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    print(f"ICO gerado: {ico_path}")

if __name__ == "__main__":
    create_professional_assets()