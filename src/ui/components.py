# ui/components.py

import flet as ft

def create_detection_history(history_list: list):
    """
    Cria a área de exibição do histórico de sinais (roxo/verde).
    history_list: Lista de sinais confirmados ou em buffer.
    """
    
    # Renderiza os últimos 5 a 10 sinais, por exemplo
    display_limit = 8 
    
    history_widgets = []
    
    # Iterar sobre a lista de histórico (do mais antigo ao mais recente)
    for index, sign in enumerate(history_list[-display_limit:]):
        
        # Cor de destaque para o sinal mais recente (como o verde na sua prototipagem)
        # Assumimos que o último sinal da lista é o mais estável/recente
        is_latest = index == len(history_list[-display_limit:]) - 1
        
        # Corrigido: Usaremos cores que representam o feedback visual que você sugeriu
        
        # Se for o último sinal estável (GREEN/AZUL), ou se for um sinal em buffer (ROXO/CINZA)
        color = ft.Colors.TEAL_400 if is_latest else ft.Colors.BLUE_GREY_500 
        
        history_widgets.append(
            ft.Container(
                content=ft.Text(
                    value=sign if sign else "...", 
                    size=16, 
                    color=ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD
                ),
                alignment=ft.alignment.center,
                width=50,
                height=30,
                bgcolor=color,
                border_radius=5,
                margin=ft.margin.only(right=5),
            )
        )
        
    return ft.Row(
        controls=history_widgets,
        wrap=True,
        spacing=5,
        alignment=ft.MainAxisAlignment.START
    )


def create_output_panel(title: str, content_text: str = "", color=ft.Colors.BLUE_GREY_100):
    """
    Cria um painel padrão para output ou status.
    """
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(title, weight=ft.FontWeight.BOLD, size=14, color=ft.Colors.BLACK54),
                ft.Text(content_text, size=20, weight=ft.FontWeight.W_600),
            ],
            spacing=5
        ),
        padding=10,
        bgcolor=color,
        border_radius=8,
        width=200,
        height=100
    )