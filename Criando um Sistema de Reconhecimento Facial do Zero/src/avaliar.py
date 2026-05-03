"""
Utilitários para avaliação e visualização do sistema de reconhecimento facial.

Uso:
    python src/avaliar.py --dataset dataset/ --modelo modelos/modelo_reconhecimento.keras --meta modelos/metadados.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))

IMG_SIZE = (160, 160)
BATCH_SIZE = 32


def carregar_dados_avaliacao(diretorio: str, classes: list) -> tuple:
    """Carrega imagens do dataset e as mapeia às classes do modelo."""
    imagens, labels = [], []

    for idx, classe in enumerate(classes):
        pasta = os.path.join(diretorio, classe)
        if not os.path.isdir(pasta):
            continue
        for arq in os.listdir(pasta):
            if not arq.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            caminho = os.path.join(pasta, arq)
            img = tf.keras.preprocessing.image.load_img(caminho, target_size=IMG_SIZE)
            arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
            imagens.append(arr)
            labels.append(idx)

    return np.array(imagens), np.array(labels)


def avaliar(args):
    # Carregar metadados e modelo
    with open(args.meta, "r", encoding="utf-8") as f:
        metadados = json.load(f)
    classes = metadados["classes"]

    modelo = tf.keras.models.load_model(args.modelo)
    print(f"[INFO] Modelo carregado. Classes: {classes}")

    # Carregar dados
    X, y_verdadeiro = carregar_dados_avaliacao(args.dataset, classes)
    if len(X) == 0:
        print("[ERRO] Nenhuma imagem encontrada.")
        return

    print(f"[INFO] {len(X)} imagens carregadas.")

    # Predições
    ds = tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    probabilidades = modelo.predict(ds, verbose=1)
    y_predito = np.argmax(probabilidades, axis=1)

    # Relatório de classificação
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)
    print(classification_report(y_verdadeiro, y_predito, target_names=classes))

    # Matriz de confusão
    cm = confusion_matrix(y_verdadeiro, y_predito)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), max(6, len(classes) * 1.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de Confusão — Reconhecimento Facial")
    plt.tight_layout()

    os.makedirs(args.saida, exist_ok=True)
    caminho_cm = os.path.join(args.saida, "matriz_confusao.png")
    plt.savefig(caminho_cm, dpi=150)
    plt.close()
    print(f"[INFO] Matriz de confusão salva em: {caminho_cm}")

    # Visualizar amostras com predições
    _visualizar_amostras(X, y_verdadeiro, y_predito, probabilidades, classes, args.saida)


def _visualizar_amostras(X, y_verdadeiro, y_predito, probabilidades, classes, diretorio_saida, n=16):
    """Salva uma grade com amostras e suas predições."""
    indices = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = X[idx].astype(np.uint8)
        verdadeiro = classes[y_verdadeiro[idx]]
        predito = classes[y_predito[idx]]
        confianca = probabilidades[idx][y_predito[idx]]
        cor = "green" if verdadeiro == predito else "red"

        axes[i].imshow(img[..., ::-1])  # BGR → RGB
        axes[i].set_title(
            f"Real: {verdadeiro}\nPred: {predito} ({confianca:.0%})",
            fontsize=8, color=cor
        )
        axes[i].axis("off")

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Amostras — Predições do Modelo", fontsize=12)
    plt.tight_layout()
    caminho = os.path.join(diretorio_saida, "amostras_predicoes.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[INFO] Visualização de amostras salva em: {caminho}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do modelo de reconhecimento facial")
    parser.add_argument("--dataset", type=str, default="dataset", help="Diretório do dataset")
    parser.add_argument(
        "--modelo", type=str, default="modelos/modelo_reconhecimento.keras",
        help="Caminho para o modelo .keras"
    )
    parser.add_argument(
        "--meta", type=str, default="modelos/metadados.json",
        help="Caminho para o arquivo metadados.json"
    )
    parser.add_argument("--saida", type=str, default="modelos", help="Diretório para salvar os gráficos")
    args = parser.parse_args()
    avaliar(args)
