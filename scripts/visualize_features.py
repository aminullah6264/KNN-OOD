#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from knn_ood.datasets import get_cifar10, get_ood_dataset, make_loader
from knn_ood.models import ResNet18Backbone, l2_normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ood", default="lsun")
    p.add_argument("--output", default="outputs/tsne.png")
    p.add_argument("--samples", type=int, default=1500)
    return p.parse_args()


def collect(model, loader, device, limit):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, f, _ = model(x, return_features=True)
            feats.append(f.cpu())
            labels.append(y)
            if sum(z.shape[0] for z in feats) >= limit:
                break
    f = torch.cat(feats, 0)[:limit]
    y = torch.cat(labels, 0)[:limit]
    return f, y


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Backbone(num_classes=10, proj_dim=128)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model = model.to(device).eval()

    id_loader = make_loader(get_cifar10(args.data_root, train=False), 256, 4)
    ood_loader = make_loader(get_ood_dataset(args.ood, args.data_root), 256, 4)

    id_f, id_y = collect(model, id_loader, device, args.samples)
    ood_f, _ = collect(model, ood_loader, device, args.samples)
    all_f = l2_normalize(torch.cat([id_f, ood_f], dim=0)).numpy()

    z2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_f)
    id_z, ood_z = z2[: len(id_f)], z2[len(id_f) :]

    plt.figure(figsize=(7, 6))
    plt.scatter(id_z[:, 0], id_z[:, 1], c=id_y.numpy(), s=6, cmap="tab10", alpha=0.7, label="CIFAR-10")
    plt.scatter(ood_z[:, 0], ood_z[:, 1], c="gray", s=6, alpha=0.5, label=args.ood)
    plt.legend()
    plt.title("Normalized Penultimate Features (t-SNE)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
