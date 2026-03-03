#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn_ood.datasets import get_cifar10, get_ood_dataset, make_loader
from knn_ood.models import ResNet18Backbone, infer_proj_dim, l2_normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ood", default="lsun")
    p.add_argument("--output", default="outputs/embed.png",
                   help="Base output path. For --method both, saves *_tsne.png and *_umap.png")
    p.add_argument("--samples", type=int, default=1500)

    p.add_argument("--method", choices=["tsne", "umap", "both"], default="tsne")
    # t-SNE params
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-seed", type=int, default=42)
    # UMAP params
    p.add_argument("--umap-n-neighbors", type=int, default=20)
    p.add_argument("--umap-min-dist", type=float, default=0.15)
    p.add_argument("--umap-seed", type=int, default=42)
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


def _save_scatter(z2, id_len, id_y, ood_name, title, out_path):
    id_z, ood_z = z2[:id_len], z2[id_len:]

    plt.figure(figsize=(7, 6))
    plt.scatter(id_z[:, 0], id_z[:, 1], c=id_y.numpy(), s=6, cmap="tab10", alpha=0.7, label="CIFAR-10")
    plt.scatter(ood_z[:, 0], ood_z[:, 1], c="gray", s=6, alpha=0.5, label=ood_name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved to {out_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    proj_dim = infer_proj_dim(ckpt["model"])
    model = ResNet18Backbone(num_classes=10, proj_dim=proj_dim)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    id_loader = make_loader(get_cifar10(args.data_root, train=False), 256, 4)
    ood_loader = make_loader(get_ood_dataset(args.ood, args.data_root), 256, 4)

    id_f, id_y = collect(model, id_loader, device, args.samples)
    ood_f, _ = collect(model, ood_loader, device, args.samples)
    all_f = l2_normalize(torch.cat([id_f, ood_f], dim=0)).numpy()

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.method in ("tsne", "both"):
        z_tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            random_state=args.tsne_seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(all_f)
        out_tsne = out_base.with_name(out_base.stem + "_tsne" + out_base.suffix)
        _save_scatter(
            z_tsne, len(id_f), id_y, args.ood,
            "Normalized Penultimate Features (t-SNE)",
            str(out_tsne),
        )

    if args.method in ("umap", "both"):
        try:
            import umap
        except ImportError as e:
            raise SystemExit("UMAP not installed. Run: pip install umap-learn") from e

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.umap_seed,
        )
        z_umap = reducer.fit_transform(all_f)
        out_umap = out_base.with_name(out_base.stem + "_umap" + out_base.suffix)
        _save_scatter(
            z_umap, len(id_f), id_y, args.ood,
            "Normalized Penultimate Features (UMAP)",
            str(out_umap),
        )


if __name__ == "__main__":
    main()