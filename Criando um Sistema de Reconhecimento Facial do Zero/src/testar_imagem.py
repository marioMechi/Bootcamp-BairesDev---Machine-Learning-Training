"""
Testa o reconhecimento facial em imagens estáticas e exibe os rostos detectados
anotados com o nome de cada personagem.

Uso:
    python src/testar_imagem.py --imagem caminho/foto.jpg
    python src/testar_imagem.py --pasta caminho/pasta/  (processa todas as imagens)
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from face_detector import FaceDetector
from model import L2Normalize  # registra a camada customizada antes de load_model

IMG_SIZE = (160, 160)
LIMIAR = 0.40   # limiar mais baixo pois o dataset é pequeno

# Paleta de cores por personagem
CORES = {
    "chandler": (255, 140, 0),
    "joey":     (0, 200, 255),
    "monica":   (255, 0, 200),
    "phoebe":   (0, 255, 150),
    "rachel":   (255, 80, 80),
    "ross":     (80, 180, 255),
}
COR_DESCONHECIDO = (120, 120, 120)


def carregar_modelo(caminho_modelo: str, caminho_meta: str):
    modelo = tf.keras.models.load_model(caminho_modelo)
    with open(caminho_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta["classes"]


def prever(modelo, face_img: np.ndarray, classes: list):
    img = cv2.resize(face_img, IMG_SIZE).astype(np.float32)
    probs = modelo.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    nome = classes[idx] if conf >= LIMIAR else "Desconhecido"
    return nome, conf, probs


def anotar_imagem(frame: np.ndarray, modelo, classes: list, detector: FaceDetector) -> np.ndarray:
    faces = detector.detect(frame)
    output = frame.copy()

    if len(faces) == 0:
        cv2.putText(output, "Nenhum rosto detectado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return output, []

    resultados = []
    for bbox in faces:
        face_img = detector.extract_face(frame, bbox, target_size=IMG_SIZE, margin=15)
        nome, conf, probs = prever(modelo, face_img, classes)

        cor = CORES.get(nome, COR_DESCONHECIDO)
        x, y, w, h = bbox

        # Retângulo
        cv2.rectangle(output, (x, y), (x + w, y + h), cor, 3)

        # Label com fundo
        label = f"{nome.capitalize()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(output, (x, y - th - 12), (x + tw + 8, y), cor, -1)
        cv2.putText(output, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resultados.append({"nome": nome, "confianca": conf, "bbox": bbox})

    # Mini legenda no canto
    for i, res in enumerate(resultados):
        txt = f"[{i+1}] {res['nome'].capitalize()} ({res['confianca']:.0%})"
        cor = CORES.get(res["nome"], COR_DESCONHECIDO)
        cv2.putText(output, txt, (10, output.shape[0] - 20 - i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

    return output, resultados


def processar_imagem(caminho: str, modelo, classes: list, detector: FaceDetector,
                     saida_dir: str) -> str:
    frame = cv2.imread(caminho)
    if frame is None:
        print(f"[ERRO] Não foi possível abrir: {caminho}")
        return None

    print(f"\n[INFO] Processando: {os.path.basename(caminho)}")
    output, resultados = anotar_imagem(frame, modelo, classes, detector)

    for r in resultados:
        print(f"  -> {r['nome'].capitalize():12s}  confiança: {r['confianca']:.1%}")

    os.makedirs(saida_dir, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho))[0]
    caminho_saida = os.path.join(saida_dir, f"{nome_base}_resultado.jpg")
    cv2.imwrite(caminho_saida, output)
    print(f"  Salvo em: {caminho_saida}")
    return caminho_saida


def mostrar_grade(imagens_saida: list):
    """Exibe uma janela com grade de todos os resultados."""
    if not imagens_saida:
        return

    frames = [cv2.imread(p) for p in imagens_saida if p and os.path.exists(p)]
    if not frames:
        return

    # Redimensiona para largura uniforme
    w_alvo = 640
    frames_r = []
    for f in frames:
        h, w = f.shape[:2]
        novo_h = int(h * w_alvo / w)
        frames_r.append(cv2.resize(f, (w_alvo, novo_h)))

    # Monta grade 2 colunas
    colunas = 2
    linhas = (len(frames_r) + 1) // colunas
    h_max = max(f.shape[0] for f in frames_r)

    grade_linhas = []
    for i in range(linhas):
        linha_frames = frames_r[i * colunas: (i + 1) * colunas]
        # Preenche com preto se número ímpar
        while len(linha_frames) < colunas:
            linha_frames.append(np.zeros((h_max, w_alvo, 3), dtype=np.uint8))
        # Padeia altura
        padded = []
        for f in linha_frames:
            if f.shape[0] < h_max:
                pad = np.zeros((h_max - f.shape[0], w_alvo, 3), dtype=np.uint8)
                f = np.vstack([f, pad])
            padded.append(f)
        grade_linhas.append(np.hstack(padded))

    grade = np.vstack(grade_linhas)
    cv2.imshow("Resultados - Reconhecimento Facial Friends", grade)
    print("\n[INFO] Pressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(args):
    print("[INFO] Carregando modelo...")
    modelo, classes = carregar_modelo(args.modelo, args.meta)
    detector = FaceDetector(scale_factor=1.1, min_neighbors=4, min_size=(40, 40))

    saida_dir = args.saida

    imagens_processadas = []

    if args.imagem:
        p = processar_imagem(args.imagem, modelo, classes, detector, saida_dir)
        imagens_processadas.append(p)

    elif args.pasta:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        arquivos = sorted([
            os.path.join(args.pasta, f)
            for f in os.listdir(args.pasta)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"[INFO] {len(arquivos)} imagens encontradas em {args.pasta}")
        for arq in arquivos[:args.max]:
            p = processar_imagem(arq, modelo, classes, detector, saida_dir)
            imagens_processadas.append(p)

    else:
        print("[ERRO] Passe --imagem ou --pasta como argumento.")
        parser.print_help()
        return

    print(f"\n[CONCLUÍDO] {len([p for p in imagens_processadas if p])} imagens salvas em: {saida_dir}")

    if not args.no_display:
        mostrar_grade(imagens_processadas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de reconhecimento facial em imagens")
    parser.add_argument("--imagem", type=str, default=None, help="Caminho para uma imagem")
    parser.add_argument("--pasta", type=str, default=None, help="Pasta com imagens para testar")
    parser.add_argument("--max", type=int, default=6, help="Máximo de imagens (padrão: 6)")
    parser.add_argument("--modelo", type=str, default="modelos/modelo_reconhecimento.keras")
    parser.add_argument("--meta", type=str, default="modelos/metadados.json")
    parser.add_argument("--saida", type=str, default="modelos/resultados_teste")
    parser.add_argument("--no-display", action="store_true", dest="no_display",
                        help="Não abre janela de visualização")
    args = parser.parse_args()
    main(args)
