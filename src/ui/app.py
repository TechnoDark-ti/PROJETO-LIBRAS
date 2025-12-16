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
        self.page.bgcolor = ft.Colors.WHITE
        
        # Estado inicial (Simulado)
        self.current_signal = "N/A"
        self.translated_text = "Nenhuma tradução."
        self.history = ["A", "A", "B", "L", "A", "L"] # Lista de exemplo

        self._build_ui()

    def _build_ui(self):

        # 1. Painel da Câmera / Detecção
        # ---
        # NOVO: Inicializa o controle Image que será o feed de vídeo
        self.image_control = ft.Image(
            src_base64=None, # Começa sem imagem
            width=500,
            height=400,
            fit=ft.ImageFit.CONTAIN # Garante que a imagem se ajuste
        )
        
        self.video_area = ft.Container(
            width=500,
            height=400,
            # Placeholder para a área de vídeo/captura
            content=self.image_control, # AGORA CONTÉM O CONTROLE IMAGE
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.BLACK,
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
                    color=ft.Colors.CYAN_100
                ),
                
                # Sinal Atual (em Buffer)
                create_output_panel(
                    title="Sinal em Classificação",
                    content_text=self.current_signal,
                    color=ft.Colors.INDIGO_100
                ),
                
                ft.Divider(height=20),
                
                # Controles (Placeholder para botões de Start/Stop)
                ft.Row([
                    ft.ElevatedButton("INICIAR CÂMERA", bgcolor=ft.Colors.GREEN_500, color=ft.Colors.WHITE),
                    ft.ElevatedButton("RESET BUFFER", bgcolor=ft.Colors.RED_500, color=ft.Colors.WHITE),
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
    
    def update_ui_with_data(self, current_signal: str, translated_text: str, history: list, frame_bytes: bytes = None):
        """
        Método chamado pelo thread do Core para atualizar a interface.
        """
        # Atualizar o estado da UI com os dados do Core
        self.current_signal = current_signal
        self.translated_text = translated_text
        self.history = history

        self.output_column.controls[1].content.controls[1].value = translated_text
        # Atualizar a área de Sinal em Classificação
        self.output_column.controls[2].content.controls[1].value = current_signal
        
        # Recriar o painel de histórico (mais simples que atualizar a lista de controls)
        self.history_panel.controls[1] = create_detection_history(history)
        
        if frame_bytes:
            # Flet espera a imagem em base64, mas o Image control aceita bytes diretamente em src_base64
            self.image_control.src_base64 = frame_bytes
        
        # Forçar a atualização da UI
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