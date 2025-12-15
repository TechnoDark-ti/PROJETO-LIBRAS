# tests/test_model.py

from core.translator import LibrasTranslator

def main():
    translator = LibrasTranslator

    #simulação de landmarks (21 pontos x 2)
    fake_landmarks = [0.5] * 42

    result = translator.translate(fake_landmarks)
    print("Resultado: ", result)

if __name__ == "__main__":
    main()