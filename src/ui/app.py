# ui/app.py

import flet as ft
from ui.components import create_detection_history, create_output_panel

class MainApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Tradutor Libras-PT"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.START
        self.page.window_width = 1000
        self.page.window_height = 700
        self.page.bgcolor = ft.colors.WHITE
        
        # Estado inicial (Simulado)
        self.current_signal = "N/A"
        self.translated_text = "Nenhuma tradução."
        self.history = ["A", "A", "B", "L", "A", "L"] # Lista de exemplo

        self._build_ui()

    def _build_ui(self):
        # ----------------------------------------------------
        # 1. Painel da Câmera / Detecção
        # ----------------------------------------------------
        self.video_area = ft.Container(
            width=500,
            height=400,
            # Placeholder para a área de vídeo/captura (CVZone/OpenCV será injetado aqui)
            content=ft.Text("Área de Captura de Sinais (Webcam / Flet Frame)"), 
            alignment=ft.alignment.center,
            bgcolor=ft.colors.BLACK,
            border_radius=10,
            margin=ft.margin.only(right=15)
        )
        
        # ----------------------------------------------------
        # 2. Painel Lateral de Saída
        # ----------------------------------------------------
        self.output_column = ft.Column(
            controls=[
                ft.Text("OUTPUT DO SISTEMA", size=18, weight=ft.FontWeight.BOLD),
                
                # Tradução Final
                create_output_panel(
                    title="Tradução Final",
                    content_text=self.translated_text,
                    color=ft.colors.CYAN_100
                ),
                
                # Sinal Atual (em Buffer)
                create_output_panel(
                    title="Sinal em Classificação",
                    content_text=self.current_signal,
                    color=ft.colors.INDIGO_100
                ),
                
                ft.Divider(height=20),
                
                # Controles (Placeholder para botões de Start/Stop)
                ft.Row([
                    ft.ElevatedButton("INICIAR CÂMERA", bgcolor=ft.colors.GREEN_500, color=ft.colors.WHITE),
                    ft.ElevatedButton("RESET BUFFER", bgcolor=ft.colors.RED_500, color=ft.colors.WHITE),
                ]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=15
        )
        
        # ----------------------------------------------------
        # 3. Painel de Histórico (Rodapé)
        # ----------------------------------------------------
        self.history_panel = ft.Column(
            controls=[
                ft.Text("HISTÓRICO DE SINAIS ESTÁVEIS", size=16, weight=ft.FontWeight.BOLD),
                create_detection_history(self.history)
            ],
            spacing=10
        )
        
        
        # Estrutura da Página
        self.page.add(
            ft.Row(
                controls=[
                    self.video_area,
                    self.output_column,
                ],
                alignment=ft.MainAxisAlignment.START
            ),
            ft.Container(height=30), # Espaçamento
            self.history_panel
        )
        self.page.update()

def start_app(target_main):
    """
    Função wrapper para rodar o aplicativo Flet,
    evitando que o código do MainApp seja importado no Core.
    """
    ft.app(target=target_main)

if __name__ == "__main__":
    # Teste de execução direta da UI
    ft.app(target=lambda page: MainApp(page))