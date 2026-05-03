# Sistema de Reconhecimento Facial com TensorFlow

Sistema completo de reconhecimento facial construído do zero usando **TensorFlow/Keras** e **OpenCV**, com transfer learning sobre MobileNetV2.

---

## Arquitetura

```
dataset/
    pessoa_a/   ← imagens capturadas
    pessoa_b/
modelos/
    modelo_reconhecimento.keras
    metadados.json
    historico_treinamento.png
    matriz_confusao.png
src/
    face_detector.py   ← detecção de rostos (Haar Cascade)
    model.py           ← arquitetura MobileNetV2 + cabeça de classificação
    coletar_faces.py   ← captura de imagens pela webcam
    treinar.py         ← treinamento com augmentação e callbacks
    reconhecer.py      ← reconhecimento em tempo real (webcam) ou estático
    avaliar.py         ← matriz de confusão e relatório de classificação
```

### Pipeline do modelo

```
Imagem (160×160×3)
    → MobileNetV2 (backbone ImageNet, congelado)
    → GlobalAveragePooling2D
    → Dense(256) + BatchNorm + Dropout
    → Dense(128) + BatchNorm + Dropout
    → Embedding L2-normalizado (64-d)
    → Dense(n_classes) + Softmax
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Fluxo de uso

### 1. Coletar faces

Execute para **cada pessoa** que deseja reconhecer:

```bash
# Captura 100 fotos da webcam para "João"
python src/coletar_faces.py --nome "Joao" --amostras 100

# Captura 150 fotos para "Maria"
python src/coletar_faces.py --nome "Maria" --amostras 150
```

**Controles durante a coleta:**
- `ESPAÇO` — captura manualmente
- `A` — ativa/desativa captura automática
- `Q` — encerra

---

### 2. Treinar o modelo

```bash
# Treinamento básico (20 épocas)
python src/treinar.py --dataset dataset/ --epocas 20

# Com fine-tuning do backbone (recomendado para datasets maiores)
python src/treinar.py --dataset dataset/ --epocas 30 --fine-tuning
```

Saídas geradas em `modelos/`:
- `modelo_reconhecimento.keras`
- `metadados.json`
- `historico_treinamento.png`

---

### 3. Reconhecimento em tempo real

```bash
python src/reconhecer.py
```

Ou com caminhos customizados:

```bash
python src/reconhecer.py \
    --modelo modelos/modelo_reconhecimento.keras \
    --meta modelos/metadados.json \
    --limiar 0.75
```

**Modo imagem estática:**

```bash
python src/reconhecer.py --imagem foto.jpg
```

---

### 4. Avaliar o modelo

```bash
python src/avaliar.py --dataset dataset/ --modelo modelos/modelo_reconhecimento.keras --meta modelos/metadados.json
```

Gera em `modelos/`:
- `matriz_confusao.png`
- `amostras_predicoes.png`
- Relatório de classificação no terminal

---

## Parâmetros importantes

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `--amostras` | Imagens por pessoa | 100 |
| `--epocas` | Épocas de treino | 20 |
| `--fine-tuning` | Fine-tune do backbone | desativado |
| `--limiar` | Confiança mínima p/ reconhecer | 0.70 |

---

## Dicas

- **Mínimo recomendado**: 50–100 imagens por pessoa com variação de iluminação, ângulo e expressão.
- **Qualidade > quantidade**: imagens bem iluminadas e centralizadas produzem melhores resultados.
- Ajuste `--limiar` para reduzir falsos positivos (valor maior = mais restrito).
- Use `--fine-tuning` quando tiver 200+ imagens por classe.
