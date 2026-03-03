# KNN-OOD (ResNet-18 on CIFAR-10)

This repository provides a **reproducible experiment scaffold** for the paper:

> *Out-of-Distribution Detection with Deep Nearest Neighbors* (Sun et al., ICML 2022)

It includes:
- training scripts for **KNN** (CE-trained backbone) and **KNN+** (SupCon-trained backbone),
- config files for hyperparameters,
- OOD testing script over CIFAR-10 benchmark OOD datasets,
- feature visualization script.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset layout

The scripts auto-download:
- CIFAR-10
- SVHN

For the remaining OOD datasets, place extracted image folders under `data/`:

```text
data/
  lsun/
    class_x/*.jpg
  isun/
    class_x/*.jpg
  textures/
    class_x/*.jpg
  places365/
    class_x/*.jpg
```

Use this exact lowercase naming for compatibility with `scripts/test_ood.py`.

## 3) Train the backbone

### KNN (cross-entropy backbone)
```bash
python scripts/train.py --config configs/train_ce_cifar10.yaml --mode ce
```

### KNN+ (SupCon + CE backbone)
```bash
python scripts/train.py --config configs/train_supcon_cifar10.yaml --mode supcon
```

The best checkpoint is saved to `outputs/<run>/best.pt`.

## 4) Evaluate KNN OOD detection

```bash
python scripts/test_ood.py --config configs/eval_knn_cifar10.yaml
```

Output format:

```text
dataset,fpr95,auroc
svhn,...
lsun,...
isun,...
textures,...
places365,...
```

## 5) Visualize feature geometry (ID vs OOD)

```bash
python scripts/visualize_features.py \
  --checkpoint outputs/supcon/best.pt \
  --ood lsun \
  --output outputs/tsne_lsun.png
```

This generates a t-SNE scatter of normalized penultimate features, similar to the qualitative analysis in the paper.

## 6) Main implementation notes

- Backbone: **ResNet-18** with 512-D penultimate feature.
- OOD score: negative distance to the **k-th nearest training feature**.
- Feature normalization: enabled by default in evaluation config.
- Fast search: uses `faiss.IndexFlatL2` when `use_faiss: true`.

## 7) Important reproducibility tips

- Match paper-style hyperparameters through configs in `configs/`.
- For full fidelity to the paper, train SupCon for long schedules (500 epochs).
- Set `knn.k` via validation (paper used candidate sweep; default here is `k=50` for CIFAR-10).

## 8) File guide

- `scripts/train.py` – training entry point (CE or SupCon).
- `scripts/test_ood.py` – computes KNN OOD metrics on listed datasets.
- `scripts/visualize_features.py` – t-SNE plot for ID/OOD feature separation.
- `knn_ood/models.py` – ResNet-18 + projector definition.
- `knn_ood/losses.py` – supervised contrastive loss.
- `knn_ood/datasets.py` – CIFAR-10 / OOD loaders.
- `knn_ood/metrics.py` – FPR@95 and AUROC.

