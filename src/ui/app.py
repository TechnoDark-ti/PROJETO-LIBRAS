import flet as ft
from ui.components import create_detection_history, create_output_panel, create_confidence_bar

class MainApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Tradutor Libras-PT"
        self.page.window_width = 1100
        self.page.window_height = 850
        self.page.bgcolor = ft.Colors.WHITE
        
        self.current_signal = "N/A"
        self.translated_text = "Nenhuma tradução."
        self.history = ["Nenh"]
        self.confidence_value = 0.0

        self._build_ui()

    def _build_ui(self):
        # 1. Área da Câmera
        self.image_control = ft.Image(src_base64=None, width=640, height=480, fit=ft.ImageFit.CONTAIN)
        self.video_area = ft.Container(
            content=self.image_control,
            bgcolor=ft.Colors.BLACK,
            border_radius=10,
            padding=5
        )

        # 2. Painel Lateral (Output e Ferramentas para o Professor)
        # Criamos primeiro a coluna
        self.output_column = ft.Column(
            controls=[
                ft.Text("TRADUÇÃO EM TEMPO REAL", size=18, weight=ft.FontWeight.BOLD),
                create_output_panel("Texto Final", self.translated_text, ft.Colors.CYAN_100),
                create_output_panel("Sinal Detectado", self.current_signal, ft.Colors.INDIGO_100),
                ft.Divider(height=20),
                ft.Text("FERRAMENTAS DO PROFESSOR", size=14, weight=ft.FontWeight.W_500),
            ],
            spacing=15
        )

        # Agora criamos os botões de controle
        self.start_button = ft.ElevatedButton("LIGAR CÂMERA", bgcolor=ft.Colors.GREEN_500, color=ft.Colors.WHITE)
        self.reset_button = ft.ElevatedButton("LIMPAR TEXTO", bgcolor=ft.Colors.RED_500, color=ft.Colors.WHITE)
        
        # Área de Coleta (Design para leigos: Campo de texto + Botão)
        self.label_input = ft.TextField(label="Nome da Letra/Sinal", width=150, value="A", dense=True)
        self.record_button = ft.ElevatedButton(
            "SALVAR EXEMPLO", 
            icon=ft.Icons.CAMERA_ALT, 
            bgcolor=ft.Colors.ORANGE_700, 
            color=ft.Colors.WHITE
        )

        # Adicionamos os controles à coluna já existente
        self.output_column.controls.append(ft.Row([self.start_button, self.reset_button]))
        self.output_column.controls.append(ft.Divider(height=10))
        self.output_column.controls.append(ft.Row([self.label_input, self.record_button]))

        # 3. Rodapé (Confiança e Histórico)
        self.confidence_bar = create_confidence_bar(self.confidence_value)
        self.history_panel = ft.Column(
            controls=[
                self.confidence_bar,
                ft.Text("SINAIS RECENTES", size=16, weight=ft.FontWeight.BOLD),
                create_detection_history(self.history)
            ],
            spacing=10
        )

        self.page.add(
            ft.Row([self.video_area, self.output_column], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Container(height=20),
            self.history_panel
        )
        self.page.update()

    def update_ui_with_data(self, current_signal, translated_text, history, frame_bytes, confidence):
        # Atualiza Barra
        if len(self.confidence_bar.controls) > 1:
            self.confidence_bar.controls[1].value = confidence

        # Atualiza Textos
        self.output_column.controls[1].content.controls[1].value = translated_text
        self.output_column.controls[2].content.controls[1].value = current_signal
        
        # Atualiza Histórico
        self.history_panel.controls[2] = create_detection_history(history)
        
        if frame_bytes:
            self.image_control.src_base64 = frame_bytes
        
        self.page.update()

    # NOVO: set_handlers agora inclui o botão de gravar
    def set_handlers(self, start_func, reset_func, record_func):
        self.start_button.on_click = start_func
        self.reset_button.on_click = reset_func
        self.record_button.on_click = record_func
        self.page.update()
    
def start_app(target_main):
    """
    Função wrapper para rodar o aplicativo Flet.
    Ela permite que o main.py inicie a UI passando a função principal.
    """
    ft.app(target=target_main)

if __name__ == "__main__":
    # Permite testar a UI isoladamente se rodar este arquivo diretamente
    ft.app(target=lambda page: MainApp(page))