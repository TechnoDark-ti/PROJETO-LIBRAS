import flet as ft
from ui.components import create_detection_history, create_output_panel, create_confidence_bar

class MainApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Sistema de Alfabetização em Libras"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        
        # Removido dimensões fixas da janela para permitir maximizar livremente
        # self.page.window_width = 1200 
        # self.page.window_height = 900
        
        self.page.padding = 20
        self.page.bgcolor = ft.Colors.GREY_50
        
        # Variáveis de Estado
        self.current_signal = "..."
        self.translated_text = "Aguardando..."
        self.history = []
        self.confidence_value = 0.0

        # Placeholders para os handlers
        self.start_handle = lambda e: None
        self.reset_handle = lambda e: None
        self.record_handle = lambda e: None

        self._build_ui()

    def _build_ui(self):
        # --- 1. CABEÇALHO ---
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FRONT_HAND, size=40, color=ft.Colors.INDIGO),
                    ft.Column([
                        ft.Text("Tradutor Libras", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.INDIGO_900),
                        ft.Text("Sistema de Acessibilidade Computacional", size=12, color=ft.Colors.GREY_600)
                    ], spacing=0)
                ],
                alignment=ft.MainAxisAlignment.START
            ),
            margin=ft.margin.only(bottom=20)
        )

        # --- 2. ÁREA DE VÍDEO (Esquerda - Responsiva) ---
        self.image_control = ft.Image(
            src_base64=None,
            width=None, # Largura automática
            height=None, # Altura automática
            fit=ft.ImageFit.CONTAIN, # Ajusta mantendo proporção
            border_radius=15,
            expand=True # Ocupa todo o espaço do container pai
        )
        
        # Container responsivo para o vídeo
        self.video_container = ft.Container(
            content=self.image_control,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.BLACK,
            border_radius=15,
            shadow=ft.BoxShadow(blur_radius=15, color=ft.Colors.GREY_400, offset=ft.Offset(0, 5)),
            expand=True, # IMPORTANTE: Permite crescer/encolher
            padding=10
        )

        # --- 3. PAINEL DE CONTROLE (Direita - Largura Fixa ou Flexível) ---
        
        # Cards de Informação
        self.card_traducao = create_output_panel("Tradução Atual", self.translated_text, ft.Colors.WHITE, ft.Icons.TRANSLATE)
        self.card_sinal = create_output_panel("Sinal Detectado", self.current_signal, ft.Colors.WHITE, ft.Icons.FINGERPRINT)
        
        # Controles
        self.start_button = ft.ElevatedButton(
            "Ligar Câmera", 
            icon=ft.Icons.VIDEOCAM, 
            bgcolor=ft.Colors.INDIGO, 
            color=ft.Colors.WHITE,
            height=50,
            # CORREÇÃO: Definir expand=1 aqui fará o botão crescer dentro da Row pai
            expand=1,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
        )
        
        self.reset_button = ft.OutlinedButton(
            "Limpar Buffer", 
            icon=ft.Icons.REFRESH,
            height=50,
            # CORREÇÃO: Definir expand=1 aqui fará o botão crescer dentro da Row pai
            expand=1,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
        )

        # Área de Treinamento
        self.label_input = ft.TextField(
            label="Rótulo (ex: A, B)", 
            height=50, 
            border_radius=10, 
            bgcolor=ft.Colors.WHITE,
            expand=True # Ocupa espaço disponível na linha
        )
        
        self.record_button = ft.ElevatedButton(
            "Gravar", 
            icon=ft.Icons.SAVE, 
            bgcolor=ft.Colors.ORANGE_600, 
            color=ft.Colors.WHITE,
            height=50,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
        )

        training_row = ft.Container(
            content=ft.Row([self.label_input, self.record_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=15,
            bgcolor=ft.Colors.ORANGE_50,
            border_radius=15,
            border=ft.border.all(1, ft.Colors.ORANGE_200)
        )

        # Coluna de controles (Sidebar)
        controls_column = ft.Column(
            [
                ft.Text("Status do Sistema", weight=ft.FontWeight.BOLD, size=16),
                self.card_traducao,
                self.card_sinal,
                ft.Divider(),
                ft.Text("Controles", weight=ft.FontWeight.BOLD, size=16),
                # CORREÇÃO: Removido ft.Expanded(), agora usamos a propriedade expand=1 nos botões
                ft.Row([self.start_button, self.reset_button], spacing=10), 
                ft.Divider(),
                ft.Text("Modo Treinamento", weight=ft.FontWeight.BOLD, size=16),
                training_row
            ],
            spacing=15,
            scroll=ft.ScrollMode.AUTO # Permite rolar se a tela for pequena verticalmente
        )

        sidebar_container = ft.Container(
            content=controls_column,
            padding=10,
            width=350, # Largura fixa para a barra lateral
            # Se quiser responsivo total, remova width e use expand=1 na Row principal
        )

        # --- 4. RODAPÉ (Confiança e Histórico) ---
        self.confidence_display = create_confidence_bar(self.confidence_value)
        self.history_display = create_detection_history(self.history)
        
        footer_section = ft.Container(
            content=ft.Column([
                self.confidence_display,
                ft.Container(height=10),
                ft.Text("Histórico Recente", weight=ft.FontWeight.BOLD),
                self.history_display
            ]),
            padding=20,
            bgcolor=ft.Colors.WHITE,
            border_radius=15,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.GREY_300)
        )

        # MONTAGEM FINAL DA PÁGINA (Layout Flexível)
        
        # Corpo principal: Vídeo (expande) + Sidebar (fixa)
        main_body = ft.Row(
            [
                self.video_container, # expand=True definido no container
                sidebar_container
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True # Ocupa toda a altura disponível
        )

        # Layout global vertical
        self.page.add(
            ft.Column(
                [
                    header,
                    main_body, # Corpo expande
                    ft.Container(height=20),
                    footer_section # Rodapé fixo
                ],
                expand=True # Coluna principal ocupa a tela toda
            )
        )
        self.page.update()

    def update_ui_with_data(self, current_signal, translated_text, history, frame_bytes, confidence):
        try:
            progress_bar = self.confidence_display.content.controls[1]
            text_percent = self.confidence_display.content.controls[0].controls[1]
            
            progress_bar.value = confidence
            color = ft.Colors.RED if confidence < 0.4 else (ft.Colors.ORANGE if confidence < 0.7 else ft.Colors.GREEN)
            progress_bar.color = color
            text_percent.value = f"{int(confidence * 100)}%"
            text_percent.color = color
        except:
            pass

        self.card_traducao.content.controls[1].controls[1].value = translated_text
        self.card_sinal.content.controls[1].controls[1].value = current_signal
        
        new_history_row = create_detection_history(history)
        self.history_display.controls = new_history_row.controls
        self.history_display.update()
        
        if frame_bytes:
            self.image_control.src_base64 = frame_bytes
        
        self.page.update()

    def set_handlers(self, start_func, reset_func, record_func):
        self.start_handle = start_func
        self.reset_handle = reset_func
        self.record_handle = record_func

        self.start_button.on_click = self.start_handle
        self.reset_button.on_click = self.reset_handle
        self.record_button.on_click = self.record_handle
        
        self.page.update()

def start_app(target_main):
    ft.app(target=target_main)

if __name__ == "__main__":
    ft.app(target=lambda page: MainApp(page))