"""
Script para coletar imagens de rosto pela webcam para o dataset de treinamento.

Uso:
    python src/coletar_faces.py --nome "João" --amostras 100
"""

import argparse
import os
import cv2
from face_detector import FaceDetector


def coletar_faces(nome: str, n_amostras: int, diretorio_dataset: str, tamanho_face: tuple = (160, 160)):
    """
    Captura rostos da webcam e salva no diretório do dataset.

    Args:
        nome: Nome da pessoa.
        n_amostras: Número de imagens a coletar.
        diretorio_dataset: Caminho base do dataset.
        tamanho_face: Tamanho das imagens recortadas.
    """
    pasta_pessoa = os.path.join(diretorio_dataset, nome)
    os.makedirs(pasta_pessoa, exist_ok=True)

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível acessar a webcam.")

    print(f"\n[INFO] Coletando {n_amostras} amostras para '{nome}'.")
    print("[INFO] Pressione 'ESPAÇO' para capturar ou 'Q' para sair.\n")

    contagem = 0
    modo_auto = False

    while contagem < n_amostras:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        exibicao = detector.draw_faces(
            frame, faces,
            labels=[f"Detectado ({len(faces)})" if len(faces) else ""]
        )

        status = f"Amostras: {contagem}/{n_amostras} | [ESPAÇO] capturar | [A] auto | [Q] sair"
        cv2.putText(exibicao, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Coletor de Faces", exibicao)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Coleta interrompida pelo usuário.")
            break
        elif key == ord("a"):
            modo_auto = not modo_auto
            print(f"[INFO] Modo automático: {'ATIVADO' if modo_auto else 'DESATIVADO'}")

        capturar = (key == ord(" ")) or (modo_auto and len(faces) > 0)

        if capturar and len(faces) > 0:
            face_img = detector.extract_face(frame, faces[0], target_size=tamanho_face)
            caminho = os.path.join(pasta_pessoa, f"{nome}_{contagem:04d}.jpg")
            cv2.imwrite(caminho, face_img)
            contagem += 1
            print(f"[OK] Amostra {contagem}/{n_amostras} salva.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[CONCLUÍDO] {contagem} imagens salvas em: {pasta_pessoa}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coletor de faces para treinamento")
    parser.add_argument("--nome", type=str, required=True, help="Nome da pessoa")
    parser.add_argument("--amostras", type=int, default=100, help="Número de amostras (padrão: 100)")
    parser.add_argument("--dataset", type=str, default="dataset", help="Diretório do dataset (padrão: dataset/)")
    parser.add_argument("--tamanho", type=int, default=160, help="Tamanho da imagem facial (padrão: 160)")
    args = parser.parse_args()

    coletar_faces(
        nome=args.nome,
        n_amostras=args.amostras,
        diretorio_dataset=args.dataset,
        tamanho_face=(args.tamanho, args.tamanho),
    )
