"""
Reconhecimento facial em tempo real via webcam.

Uso:
    python src/reconhecer.py --modelo modelos/modelo_reconhecimento.keras --meta modelos/metadados.json
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from face_detector import FaceDetector
from model import L2Normalize  # registra camada customizada antes do load_model

IMG_SIZE = (160, 160)
LIMIAR_CONFIANCA = 0.70   # Confiança mínima para reconhecer (0–1)


def carregar_modelo(caminho_modelo: str, caminho_meta: str):
    """Carrega o modelo Keras e os metadados de classes."""
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")
    if not os.path.exists(caminho_meta):
        raise FileNotFoundError(f"Metadados não encontrados: {caminho_meta}")

    import keras
    keras.config.enable_unsafe_deserialization()
    modelo = tf.keras.models.load_model(caminho_modelo)

    with open(caminho_meta, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    classes = metadados["classes"]
    print(f"[INFO] Modelo carregado. Classes: {classes}")
    return modelo, classes


def prever_face(modelo, frame_face: np.ndarray, classes: list, limiar: float = LIMIAR_CONFIANCA):
    """
    Realiza a predição de identidade para um rosto.

    Returns:
        (nome, confiança) — nome é 'Desconhecido' se abaixo do limiar.
    """
    img = cv2.resize(frame_face, IMG_SIZE)
    img = img.astype(np.float32)
    batch = np.expand_dims(img, axis=0)

    predicoes = modelo.predict(batch, verbose=0)[0]
    idx = int(np.argmax(predicoes))
    confianca = float(predicoes[idx])

    nome = classes[idx] if confianca >= limiar else "Desconhecido"
    return nome, confianca


def reconhecer_tempo_real(modelo, classes: list, limiar: float = LIMIAR_CONFIANCA):
    """Loop principal de reconhecimento em tempo real."""
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível acessar a webcam.")

    print("\n[INFO] Reconhecimento em tempo real iniciado. Pressione 'Q' para sair.")

    fps_anterior = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular FPS
        agora = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(agora - fps_anterior, 1e-6))
        fps_anterior = agora

        faces = detector.detect(frame)
        labels = []

        for bbox in faces:
            face_img = detector.extract_face(frame, bbox, target_size=IMG_SIZE)
            nome, confianca = prever_face(modelo, face_img, classes, limiar)
            cor = (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255)
            labels.append((f"{nome} ({confianca:.0%})", cor))

        # Desenhar cada rosto com sua cor e label
        output = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            label, cor = labels[i] if i < len(labels) else ("", (200, 200, 200))
            cv2.rectangle(output, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        # HUD
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(output, f"Rostos: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(output, "Q: Sair", (10, output.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Reconhecimento Facial - TensorFlow", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Encerrado.")


def reconhecer_imagem(caminho_imagem: str, modelo, classes: list, limiar: float = LIMIAR_CONFIANCA):
    """Reconhece rostos em uma imagem estática e salva o resultado."""
    frame = cv2.imread(caminho_imagem)
    if frame is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    detector = FaceDetector()
    faces = detector.detect(frame)
    output = frame.copy()

    for bbox in faces:
        face_img = detector.extract_face(frame, bbox, target_size=IMG_SIZE)
        nome, confianca = prever_face(modelo, face_img, classes, limiar)
        cor = (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255)
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(output, f"{nome} ({confianca:.0%})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        print(f"  Rosto detectado: {nome} (confiança: {confianca:.2%})")

    saida = caminho_imagem.replace(".", "_resultado.")
    cv2.imwrite(saida, output)
    print(f"[INFO] Resultado salvo em: {saida}")
    cv2.imshow("Resultado", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconhecimento facial em tempo real")
    parser.add_argument(
        "--modelo", type=str, default="modelos/modelo_reconhecimento.keras",
        help="Caminho para o modelo .keras"
    )
    parser.add_argument(
        "--meta", type=str, default="modelos/metadados.json",
        help="Caminho para o arquivo metadados.json"
    )
    parser.add_argument(
        "--imagem", type=str, default=None,
        help="(Opcional) Caminho de imagem para testar no modo estático"
    )
    parser.add_argument(
        "--limiar", type=float, default=LIMIAR_CONFIANCA,
        help=f"Confiança mínima para reconhecer (padrão: {LIMIAR_CONFIANCA})"
    )
    args = parser.parse_args()

    modelo, classes = carregar_modelo(args.modelo, args.meta)

    if args.imagem:
        reconhecer_imagem(args.imagem, modelo, classes, args.limiar)
    else:
        reconhecer_tempo_real(modelo, classes, args.limiar)
