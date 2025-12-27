"""
Docstring para testes

Este arquivo serviu apenas no commit inicial para testar a tecnologia na máquina local

Ele atualmente não tem muita utilidade, uma vez que ele serviu apenas para testar a GPU e a camera
"""

import cv2
import torch
import time

from cvzone.HandTrackingModule import HandDetector

def testar_camera():
    print("\n[TESTE] Inciando teste da camera...")
    cap = cv2.VideoCapture

    if not cap.isOpened():
        print("[ERRO] Nenhuma camera encontrada.")
        return
    
    print("[INFO] Pressione 'q' para encerrar o teste.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERRO] Falha ao acessar a camera.")
            break

        cv2.imshow("Teste da Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Teste da camera finalizando!")


def test_hand_tracking():
    print("\n [TESTE] Inciando o teste do rastreamento de maos..")
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    print("[INFO] Pressione 'q' para encerrar o teste")

    while True:
        sucess, img = cap.read()

        if not sucess:
            print("[ERRO] Falha na leitura da camera")
            break

        hands, img = detector.findHands(img)

        cv2.imshow("Teste Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Rastreamento de maos funcionando!")


def test_pytorch_gpu():
    print("\n [TESTE] Testando PyTorch + GPU ...")

    cuda_available = torch.cuda.is_available()
    print(f"[INFO] CUDA disponível: {cuda_available}")

    if cuda_available:
        print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("[ALERTA] CUDA não detectada. PyTorch está usando CPU.")
    
    #desconsiderar esse teste matemático
    print("[INFO] Executando teste rápido...")
    start = time.time()

    x = torch.rand(5000, 5000)
    
    if cuda_available:
        x = x.to("cuda")
    
    y = x @ x
    _ = y.cpu()

    end = time.time()

    print(f"[OK] Teste finalizado em: {end - start:.3f} segundos.")

def menu():
    print("\n==============================")
    print("     TESTES DO SISTEMA")
    print("==============================")
    print("1 - Testar câmera")
    print("2 - Testar hand tracking")
    print("3 - Testar PyTorch GPU")
    print("0 - Sair")

    choice = input("Escolha uma opçao:")
    return choice

def teste():
    while True:
        opcao = menu()

        if opcao == "1":
            testar_camera()    

        elif opcao == "2":
            test_hand_tracking()

        elif opcao == "3":
            test_pytorch_gpu()

        elif opcao == "0":
            print("Encerrando...")
            break

        else:
            print("Opção inválida, tente novamente.")