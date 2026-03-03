#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import torch

from knn_ood.datasets import get_cifar10, get_ood_dataset, make_loader
from knn_ood.metrics import auroc, fpr95
from knn_ood.models import ResNet18Backbone, l2_normalize
from knn_ood.utils import load_yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def extract_features(model, loader, device):
    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, f, _ = model(x, return_features=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def knn_scores(train_feats: np.ndarray, test_feats: np.ndarray, k: int, use_faiss: bool = True):
    if use_faiss:
        import faiss

        index = faiss.IndexFlatL2(train_feats.shape[1])
        index.add(train_feats.astype(np.float32))
        dists, _ = index.search(test_feats.astype(np.float32), k)
        return -dists[:, -1]
    d = ((test_feats[:, None, :] - train_feats[None, :, :]) ** 2).sum(-1)
    part = np.partition(d, kth=k - 1, axis=1)
    return -part[:, k - 1]


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = ResNet18Backbone(num_classes=10, proj_dim=128)
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    train_loader = make_loader(get_cifar10(cfg["data_root"], train=True), cfg["batch_size"], cfg["num_workers"])
    test_id_loader = make_loader(get_cifar10(cfg["data_root"], train=False), cfg["batch_size"], cfg["num_workers"])

    train_feats, _ = extract_features(model, train_loader, device)
    id_feats, _ = extract_features(model, test_id_loader, device)

    if cfg["knn"]["normalize"]:
        train_feats = l2_normalize(torch.from_numpy(train_feats)).numpy()
        id_feats = l2_normalize(torch.from_numpy(id_feats)).numpy()

    id_scores = knn_scores(train_feats, id_feats, cfg["knn"]["k"], cfg["knn"]["use_faiss"])

    print("dataset,fpr95,auroc")
    for dname in cfg["ood_datasets"]:
        ood_loader = make_loader(get_ood_dataset(dname, cfg["data_root"]), cfg["batch_size"], cfg["num_workers"])
        ood_feats, _ = extract_features(model, ood_loader, device)
        if cfg["knn"]["normalize"]:
            ood_feats = l2_normalize(torch.from_numpy(ood_feats)).numpy()
        ood_scores = knn_scores(train_feats, ood_feats, cfg["knn"]["k"], cfg["knn"]["use_faiss"])
        print(f"{dname},{fpr95(id_scores, ood_scores):.2f},{auroc(id_scores, ood_scores):.2f}")


if __name__ == "__main__":
    main()
