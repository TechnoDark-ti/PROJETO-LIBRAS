import flet as ft

def create_detection_history(history_list: list):
    """
    Cria uma grade de cartões visuais para o histórico.
    """
    history_widgets = []
    
    # Exibe os últimos 6 sinais
    display_items = history_list[-6:] if history_list else ["..."]
    
    for index, sign in enumerate(display_items):
        # O último item é o destaque (Verde/Ativo)
        is_latest = index == len(display_items) - 1
        
        bg_color = ft.Colors.GREEN_ACCENT_700 if is_latest else ft.Colors.GREY_800
        text_color = ft.Colors.WHITE
        scale = 1.1 if is_latest else 1.0

        card = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        value=sign if sign else "?", 
                        size=20 if is_latest else 16, 
                        weight=ft.FontWeight.BOLD,
                        color=text_color
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
            width=60,
            height=60,
            bgcolor=bg_color,
            border_radius=12,
            padding=5,
            # CORREÇÃO AQUI: ft.Animation direto, sem sub-módulo 'animation'
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            scale=scale,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=5,
                color=ft.Colors.BLACK54,
                offset=ft.Offset(0, 2),
            ) if is_latest else None
        )
        history_widgets.append(card)
        
    return ft.Row(
        controls=history_widgets,
        wrap=True,
        spacing=10,
        alignment=ft.MainAxisAlignment.CENTER
    )

def create_output_panel(title: str, content_text: str = "", color=ft.Colors.BLUE_GREY_100, icon=ft.Icons.INFO):
    """
    Cria um painel de informação estilizado com ícone.
    """
    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Icon(icon, size=30, color=ft.Colors.BLACK54),
                ft.Column(
                    controls=[
                        ft.Text(title, size=12, weight=ft.FontWeight.W_500, color=ft.Colors.BLACK45),
                        ft.Text(content_text, size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK87),
                    ],
                    spacing=2,
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=15,
        bgcolor=color,
        border_radius=15,
        width=280,
        # Sombra suave
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=10,
            color=ft.Colors.GREY_300,
            offset=ft.Offset(0, 4),
        )
    )

def create_confidence_bar(value: float):
    """
    Barra de progresso moderna com label de porcentagem.
    """
    percentage = int(value * 100)
    color = ft.Colors.RED if value < 0.4 else (ft.Colors.ORANGE if value < 0.7 else ft.Colors.GREEN)
    
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    [
                        ft.Text("Nível de Confiança da IA", size=12, weight=ft.FontWeight.BOLD),
                        ft.Text(f"{percentage}%", size=12, weight=ft.FontWeight.BOLD, color=color)
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                ft.ProgressBar(
                    value=value, 
                    width=None, # Ocupa largura total do container pai
                    bar_height=8,
                    color=color,
                    bgcolor=ft.Colors.GREY_200
                )
            ],
            spacing=5,
        ),
        padding=10,
        border_radius=10,
        bgcolor=ft.Colors.WHITE,
        shadow=ft.BoxShadow(blur_radius=5, color=ft.Colors.GREY_200)
    )