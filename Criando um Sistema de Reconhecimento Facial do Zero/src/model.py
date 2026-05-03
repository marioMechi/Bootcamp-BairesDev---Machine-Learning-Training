"""
Arquitetura do modelo de reconhecimento facial usando TensorFlow/Keras.

Utiliza MobileNetV2 pré-treinado como extrator de features (transfer learning),
com camadas densas para classificação das identidades.
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model


@keras.saving.register_keras_serializable(package="FaceRecognizer")
class L2Normalize(layers.Layer):
    """Normalização L2 dos embeddings — substitui Lambda para serialização segura."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape


def criar_modelo(n_classes: int, input_shape: tuple = (160, 160, 3), dropout_rate: float = 0.4) -> Model:
    """
    Cria o modelo de reconhecimento facial com MobileNetV2 como backbone.

    Args:
        n_classes: Número de identidades distintas.
        input_shape: Formato da imagem de entrada (H, W, C).
        dropout_rate: Taxa de dropout para regularização.

    Returns:
        Modelo Keras compilado.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # Congela as camadas base inicialmente (fine-tuning posterior)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_face")

    # Pré-processamento esperado pelo MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)

    # Camada de embedding L2-normalizado (útil para distâncias coseno)
    embeddings = layers.Dense(64, name="embeddings")(x)
    embeddings = L2Normalize(name="l2_norm")(embeddings)

    # Cabeça de classificação
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(embeddings)

    model = Model(inputs=inputs, outputs=outputs, name="FaceRecognizer")
    return model


def descongelar_base(model: Model, n_camadas: int = 30) -> Model:
    """
    Descongela as últimas `n_camadas` do backbone para fine-tuning.

    Args:
        model: Modelo já treinado com backbone congelado.
        n_camadas: Número de camadas a descongelar a partir do fim.

    Returns:
        Modelo com backbone parcialmente descongelado.
    """
    base_model = model.get_layer("mobilenetv2_1.00_160")
    base_model.trainable = True

    # Congela tudo exceto as últimas n_camadas
    for layer in base_model.layers[:-n_camadas]:
        layer.trainable = False

    return model


def criar_extrator_embeddings(model: Model) -> Model:
    """
    Cria um sub-modelo que retorna apenas os embeddings (sem classificação).
    Útil para comparação por distância coseno em produção.

    Args:
        model: Modelo de classificação treinado.

    Returns:
        Modelo que retorna vetores de embedding (64-d).
    """
    return Model(
        inputs=model.input,
        outputs=model.get_layer("l2_norm").output,
        name="EmbeddingExtractor",
    )
