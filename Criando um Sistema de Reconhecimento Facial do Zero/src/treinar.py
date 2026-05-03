"""
Script de treinamento do modelo de reconhecimento facial.

Uso:
    python src/treinar.py --dataset dataset/ --epocas 20
    python src/treinar.py --dataset dataset/ --epocas 20 --fine-tuning

    # Baixar dataset Friends do Kaggle e treinar em seguida:
    python src/treinar.py --friends --epocas 20
"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
# Resolve importações locais ao executar como script
sys.path.insert(0, os.path.dirname(__file__))
from model import criar_modelo, descongelar_base


def configurar_gpu():
    """
    Tenta configurar GPU. No Windows nativo com TF >= 2.11, GPU não é suportada
    nativamente — o treino usará CPU com oneDNN (aceleração Intel/AMD).
    Para usar GPU no Windows, utilize WSL2 + CUDA.
    """
    import os as _os
    # Habilita oneDNN para CPU (padrão, mas explicitado)
    _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    # Suprime logs verbose do TF
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] {len(gpus)} GPU(s) detectada(s) e configurada(s):")
            for gpu in gpus:
                detalhes = tf.config.experimental.get_device_details(gpu)
                nome = detalhes.get("device_name", gpu.name)
                print(f"      {nome}")
        except RuntimeError as e:
            print(f"[AVISO] Erro ao configurar GPU: {e}")
    else:
        print("[INFO] GPU não disponível no Windows nativo com TF >= 2.11.")
        print("[INFO] Treinando com CPU + oneDNN (para GPU use WSL2 + CUDA).")

    print(f"[INFO] Dispositivo: {tf.test.gpu_device_name() or 'CPU (oneDNN)'}\n")


IMG_SIZE = (160, 160)
# Batch maior na GPU aproveita o paralelismo; na CPU mantém conservador
BATCH_SIZE = 64 if tf.config.list_physical_devices("GPU") else 32
SEED = 42


def carregar_dataset(diretorio: str, tamanho_img: tuple = IMG_SIZE) -> tuple:
    """
    Carrega imagens e labels a partir de uma estrutura de pastas:
        dataset/
            pessoa_a/
                img_0001.jpg
            pessoa_b/
                img_0001.jpg

    Returns:
        (imagens, labels_codificados, encoder)
    """
    imagens, labels = [], []

    if not os.path.isdir(diretorio):
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio}")

    classes = sorted(
        [d for d in os.listdir(diretorio) if os.path.isdir(os.path.join(diretorio, d))]
    )

    if len(classes) < 2:
        raise ValueError("O dataset precisa ter pelo menos 2 pessoas (subpastas).")

    print(f"\n[INFO] Classes encontradas ({len(classes)}): {classes}")

    for classe in classes:
        pasta = os.path.join(diretorio, classe)
        arquivos = [
            f for f in os.listdir(pasta)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  {classe}: {len(arquivos)} imagens")

        for arq in arquivos:
            caminho = os.path.join(pasta, arq)
            img = tf.keras.preprocessing.image.load_img(caminho, target_size=tamanho_img)
            arr = tf.keras.preprocessing.image.img_to_array(img)
            imagens.append(arr)
            labels.append(classe)

    X = np.array(imagens, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    print(f"\n[INFO] Total de amostras: {len(X)}")
    return X, y, encoder


def augmentar_dados(X_treino: np.ndarray, y_treino: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """Cria pipeline de augmentação de dados para o conjunto de treino."""
    dataset = tf.data.Dataset.from_tensor_slices((X_treino, y_treino))
    dataset = dataset.shuffle(buffer_size=len(X_treino), seed=SEED)

    augmentar = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentacao")

    def aplicar_aug(img, label):
        img = augmentar(img, training=True)
        return img, label

    dataset = dataset.map(aplicar_aug, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def plotar_historico(historico: dict, diretorio_saida: str):
    """Salva gráficos de acurácia e perda do treinamento."""
    os.makedirs(diretorio_saida, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Acurácia
    axes[0].plot(historico["accuracy"], label="Treino")
    axes[0].plot(historico["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(True)

    # Perda
    axes[1].plot(historico["loss"], label="Treino")
    axes[1].plot(historico["val_loss"], label="Validação")
    axes[1].set_title("Perda por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Perda")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    caminho = os.path.join(diretorio_saida, "historico_treinamento.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico salvo em: {caminho}")


def treinar(args):
    configurar_gpu()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # 1. Carregar dados
    X, y, encoder = carregar_dataset(args.dataset)
    n_classes = len(encoder.classes_)

    # 2. Dividir em treino / validação / teste
    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    print(f"\n[INFO] Treino: {len(X_treino)} | Validação: {len(X_val)} | Teste: {len(X_teste)}")

    # 3. Pipelines
    ds_treino = augmentar_dados(X_treino, y_treino, BATCH_SIZE)
    ds_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 4. Criar modelo
    modelo = criar_modelo(n_classes=n_classes)
    modelo.summary()

    os.makedirs(args.saida, exist_ok=True)

    # 5. Fase 1 — treino com backbone congelado
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_fase1 = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(args.saida, "melhor_modelo_fase1.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        TensorBoard(log_dir=os.path.join(args.saida, "logs", "fase1")),
    ]

    print("\n[FASE 1] Treinando cabeça de classificação (backbone congelado)...")
    hist1 = modelo.fit(
        ds_treino,
        validation_data=ds_val,
        epochs=args.epocas,
        callbacks=callbacks_fase1,
    )

    historico_total = dict(hist1.history)

    # 6. Fase 2 — fine-tuning (opcional)
    if args.fine_tuning:
        print("\n[FASE 2] Fine-tuning das últimas camadas do backbone...")
        modelo = descongelar_base(modelo, n_camadas=30)
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_fase2 = [
            EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(args.saida, "melhor_modelo_fase2.keras"),
                monitor="val_accuracy",
                save_best_only=True,
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
            TensorBoard(log_dir=os.path.join(args.saida, "logs", "fase2")),
        ]

        hist2 = modelo.fit(
            ds_treino,
            validation_data=ds_val,
            epochs=args.epocas // 2,
            callbacks=callbacks_fase2,
        )

        for k, v in hist2.history.items():
            historico_total[k] = historico_total.get(k, []) + v

    # 7. Avaliação no conjunto de teste
    print("\n[INFO] Avaliando no conjunto de teste...")
    ds_teste = (
        tf.data.Dataset.from_tensor_slices((X_teste, y_teste))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    perda, acuracia = modelo.evaluate(ds_teste)
    print(f"[RESULTADO] Perda: {perda:.4f} | Acurácia: {acuracia:.4f} ({acuracia*100:.1f}%)")

    # 8. Salvar modelo final e metadados
    caminho_modelo = os.path.join(args.saida, "modelo_reconhecimento.keras")
    modelo.save(caminho_modelo)
    print(f"[INFO] Modelo salvo em: {caminho_modelo}")

    metadados = {
        "classes": encoder.classes_.tolist(),
        "n_classes": n_classes,
        "input_shape": list(IMG_SIZE) + [3],
        "acuracia_teste": float(acuracia),
    }
    caminho_meta = os.path.join(args.saida, "metadados.json")
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metadados salvos em: {caminho_meta}")

    # 9. Plotar histórico
    plotar_historico(historico_total, args.saida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento do modelo de reconhecimento facial")
    parser.add_argument("--dataset", type=str, default="dataset", help="Diretório do dataset")
    parser.add_argument("--epocas", type=int, default=20, help="Número de épocas (padrão: 20)")
    parser.add_argument("--saida", type=str, default="modelos", help="Diretório de saída dos modelos")
    parser.add_argument(
        "--fine-tuning",
        action="store_true",
        dest="fine_tuning",
        help="Realiza fine-tuning do backbone após o treino inicial",
    )
    parser.add_argument(
        "--friends",
        action="store_true",
        help="Baixa o dataset Friends do Kaggle antes de treinar",
    )
    args = parser.parse_args()

    if args.friends:
        from preparar_dataset_friends import baixar_dataset, organizar_dataset
        caminho_kaggle = baixar_dataset()
        organizar_dataset(caminho_kaggle, args.dataset)

    treinar(args)
