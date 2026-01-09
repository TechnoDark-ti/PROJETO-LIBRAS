import matplotlib.pyplot as plt

def save_loss_graph(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Perda de Treinamento (Loss)', color='#1A237E', linewidth=2)
    plt.title('Convergência do Modelo ALFALIBRAS')
    plt.xlabel('Épocas')
    plt.ylabel('Valor da Perda (Cross-Entropy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # Salva a imagem para o TCC
    plt.savefig('resources/curva_loss.png', dpi=300)
    plt.show()