"""
Download das classes 'elephant' e 'giraffe' do COCO 2017
e converte as anotações para o formato YOLO (YOLOv5).

Estrutura gerada:
    coco_dataset/
        images/train/   <- imagens de treino
        images/val/     <- imagens de validação
        labels/train/   <- labels YOLO de treino
        labels/val/     <- labels YOLO de validação
    yolov5/data/coco_elefante_girafa.yaml  <- config de treino
"""

import os
import json
import zipfile
import requests
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuração ─────────────────────────────────────────────────────────────

# Classes de interesse (nomes em inglês conforme o COCO)
CLASSES = ['elephant', 'giraffe']

# IDs das categorias no COCO 2017
COCO_CAT_IDS = {
    'elephant': 22,
    'giraffe':  25,
}

# Mapeamento: categoria COCO → índice YOLO (0-based, ordem de CLASSES)
CAT_TO_IDX = {cat_id: CLASSES.index(name) for name, cat_id in COCO_CAT_IDS.items()}

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / 'coco_dataset'
ANN_DIR        = DATASET_DIR / 'annotations'
IMAGES_DIR     = DATASET_DIR / 'images'
LABELS_DIR     = DATASET_DIR / 'labels'

ANN_ZIP_URL    = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMG_URL_TRAIN  = 'http://images.cocodataset.org/train2017/{}'
IMG_URL_VAL    = 'http://images.cocodataset.org/val2017/{}'

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = 'Baixando') -> None:
    """Faz download com barra de progresso."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        chunk_size = 1024 * 64  # 64 KB
        if HAS_TQDM:
            bar = tqdm(total=total, unit='B', unit_scale=True, desc=desc)
        written = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if HAS_TQDM:
                    bar.update(len(chunk))
        if HAS_TQDM:
            bar.close()
        if not HAS_TQDM:
            mb = written / (1024 * 1024)
            print(f'  {desc}: {mb:.1f} MB baixados')


def coco_bbox_to_yolo(bbox, img_w: int, img_h: int):
    """Converte [x, y, w, h] COCO para [xc, yc, w, h] YOLO (normalizado 0-1)."""
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


def process_split(split: str, ann_file: Path, img_url_tpl: str) -> None:
    """Processa um split (train/val): filtra imagens, baixa e gera labels YOLO."""
    print(f'\n[{split.upper()}] Carregando anotações...')
    with open(ann_file, encoding='utf-8') as f:
        data = json.load(f)

    target_cat_ids = set(COCO_CAT_IDS.values())

    # Filtra anotações das classes desejadas
    filtered_anns = [a for a in data['annotations'] if a['category_id'] in target_cat_ids]
    img_ids_needed = {a['image_id'] for a in filtered_anns}
    id_to_info = {img['id']: img for img in data['images'] if img['id'] in img_ids_needed}

    # Agrupa anotações por imagem
    anns_by_img = defaultdict(list)
    for ann in filtered_anns:
        anns_by_img[ann['image_id']].append(ann)

    print(f'  {len(img_ids_needed)} imagens com elephant/giraffe encontradas.')

    images_out = IMAGES_DIR / split
    labels_out = LABELS_DIR / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(img_ids_needed, desc=f'Baixando {split}') if HAS_TQDM else img_ids_needed

    ok, skip, err = 0, 0, 0
    for img_id in iterator:
        info = id_to_info[img_id]
        filename = info['file_name']
        img_path   = images_out / filename
        label_path = labels_out / (Path(filename).stem + '.txt')

        # Baixa imagem somente se ainda não existe
        if not img_path.exists():
            url = img_url_tpl.format(filename)
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
                else:
                    err += 1
                    continue
            except Exception as e:
                err += 1
                if not HAS_TQDM:
                    print(f'  ERRO ao baixar {filename}: {e}')
                continue
        else:
            skip += 1

        # Gera label YOLO
        with open(label_path, 'w') as f:
            for ann in anns_by_img[img_id]:
                cls_idx = CAT_TO_IDX[ann['category_id']]
                xc, yc, w, h = coco_bbox_to_yolo(ann['bbox'], info['width'], info['height'])
                # Clamp para [0,1] por segurança
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                f.write(f'{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')
        ok += 1

    print(f'  Concluído: {ok} baixadas, {skip} já existiam, {err} erros.')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Download COCO 2017 – Elephant & Giraffe (YOLOv5)')
    print('=' * 60)

    # 1. Baixar e extrair anotações
    train_ann = ANN_DIR / 'instances_train2017.json'
    val_ann   = ANN_DIR / 'instances_val2017.json'

    if not train_ann.exists() or not val_ann.exists():
        ann_zip = DATASET_DIR / 'annotations_trainval2017.zip'
        if not ann_zip.exists():
            print('\nBaixando anotações COCO 2017 (~242 MB)...')
            download_file(ANN_ZIP_URL, ann_zip, 'Anotações COCO')
        print('Extraindo anotações...')
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(DATASET_DIR)
        ann_zip.unlink()  # remove o zip após extrair
        print('Anotações extraídas.')
    else:
        print('\nAnotações já existem, pulando download.')

    # 2. Processar cada split
    process_split('train', train_ann, IMG_URL_TRAIN)
    process_split('val',   val_ann,   IMG_URL_VAL)

    # 3. Criar YAML de configuração para o YOLOv5
    yaml_path = BASE_DIR / 'yolov5' / 'data' / 'coco_elefante_girafa.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = DATASET_DIR.as_posix()
    yaml_content = (
        f"# Dataset: COCO 2017 - Elephant & Giraffe\n"
        f"path: {dataset_path}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(CLASSES)}  # número de classes\n"
        f"names: {CLASSES}    # 0=elephant, 1=giraffe\n"
    )
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print('\n' + '=' * 60)
    print('  Dataset pronto!')
    print(f'  Imagens:  {IMAGES_DIR}')
    print(f'  Labels:   {LABELS_DIR}')
    print(f'  Config:   {yaml_path}')
    print('=' * 60)
    print('\nPara treinar com YOLOv5, execute:')
    print('  cd yolov5')
    print('  python train.py --data data/coco_elefante_girafa.yaml \\')
    print('                  --weights yolov5s.pt --epochs 50 --img 640')


if __name__ == '__main__':
    main()
