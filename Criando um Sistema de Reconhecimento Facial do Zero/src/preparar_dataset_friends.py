"""
Baixa o dataset de faces dos personagens de Friends do Kaggle e
organiza as imagens na estrutura esperada pelo treinar.py:

    dataset/
        chandler/
        joey/
        monica/
        phoebe/
        rachel/
        ross/

Uso:
    python src/preparar_dataset_friends.py
    python src/preparar_dataset_friends.py --destino meu_dataset/
"""

import argparse
import os
import shutil

import kagglehub
from tqdm import tqdm


DATASET_ID = "amiralikalbasi/images-of-friends-character-for-face-recognition"

# Extensões de imagem aceitas
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def baixar_dataset() -> str:
    """Faz o download do dataset via kagglehub e retorna o caminho local."""
    print(f"[INFO] Baixando dataset '{DATASET_ID}'...")
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"[INFO] Dataset baixado em: {path}")
    return path


def _eh_imagem(nome_arquivo: str) -> bool:
    ext = os.path.splitext(nome_arquivo)[1].lower()
    return ext in EXTENSOES_VALIDAS


def organizar_dataset(origem: str, destino: str):
    """
    Organiza o dataset do Kaggle para a estrutura esperada pelo treinar.py.

    Estrutura real do dataset:
        <origem>/
            Friends/
                Train/
                    Chandler/  Joey/  Monica/  Phoebe/  Rachel/  Ross/
                Test/   <- imagens sem label, ignoradas

    Args:
        origem: Caminho raiz do dataset baixado pelo kagglehub.
        destino: Pasta de destino do projeto.
    """
    os.makedirs(destino, exist_ok=True)

    # Localiza a pasta Train com subpastas de personagens
    pasta_train = None
    for raiz, dirs, _ in os.walk(origem):
        for d in dirs:
            if d.lower() == "train":
                candidato = os.path.join(raiz, d)
                subpastas = [
                    sd for sd in os.listdir(candidato)
                    if os.path.isdir(os.path.join(candidato, sd))
                ]
                if subpastas:
                    pasta_train = candidato
                    break
        if pasta_train:
            break

    if not pasta_train:
        raise FileNotFoundError(
            "Pasta 'Train' com subpastas de personagens não encontrada no dataset."
        )

    print(f"[INFO] Pasta Train: {pasta_train}")

    classes = sorted([
        d for d in os.listdir(pasta_train)
        if os.path.isdir(os.path.join(pasta_train, d))
    ])
    print(f"[INFO] Personagens encontrados: {classes}")

    total_copiados = 0
    for classe in classes:
        pasta_origem_classe = os.path.join(pasta_train, classe)
        nome_classe = classe.lower().replace(" ", "_")
        pasta_destino_classe = os.path.join(destino, nome_classe)
        os.makedirs(pasta_destino_classe, exist_ok=True)

        imagens = [
            os.path.join(raiz, f)
            for raiz, _, arquivos in os.walk(pasta_origem_classe)
            for f in arquivos
            if _eh_imagem(f)
        ]

        print(f"  {nome_classe}: {len(imagens)} imagens")

        for i, src in enumerate(tqdm(imagens, desc=f"  Copiando {nome_classe}", leave=False)):
            ext = os.path.splitext(src)[1].lower()
            dst = os.path.join(pasta_destino_classe, f"{nome_classe}_{i:05d}{ext}")
            shutil.copy2(src, dst)
            total_copiados += 1

    print(f"\n[CONCLUÍDO] {total_copiados} imagens organizadas em: {destino}")
    print(f"[INFO] Estrutura pronta para treinar.py --dataset {destino}")


def main(args):
    # 1. Download
    caminho_kaggle = baixar_dataset()

    # 2. Organização
    organizar_dataset(caminho_kaggle, args.destino)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa e organiza o dataset Friends para reconhecimento facial"
    )
    parser.add_argument(
        "--destino",
        type=str,
        default="dataset",
        help="Pasta de destino do dataset organizado (padrão: dataset/)",
    )
    args = parser.parse_args()
    main(args)
