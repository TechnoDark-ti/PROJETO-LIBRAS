from core.signal_buffer import SignalBuffer


def main():
    buffer = SignalBuffer(size=5)

    sinais = ["A", "A", "A", "A", "A"]

    for s in sinais:
        confirmado = buffer.update(s)

        if confirmado:
            print("Sinal Confirmado: ", confirmado)



if __name__ == "__main__":
    main()

