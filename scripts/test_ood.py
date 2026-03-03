#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn_ood.datasets import _cifar_transform, get_cifar10, get_ood_dataset, make_loader
from knn_ood.metrics import auroc, fpr95
from knn_ood.models import ResNet18Backbone, infer_proj_dim, l2_normalize
from knn_ood.utils import load_yaml

from torchvision import datasets, transforms
from torch.utils.data import Subset
from knn_ood.datasets import CIFAR_STATS

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


def fit_threshold_high_is_id(scores_id: np.ndarray, target_tpr: float) -> float:
    tpr = float(np.clip(target_tpr, 0.0, 1.0))
    return float(np.quantile(scores_id, 1.0 - tpr))


def make_drop_mask(dim: int, drop_percent: float, seed: int) -> np.ndarray:
    keep = max(1, int(round(dim * (1.0 - float(drop_percent) / 100.0))))
    rng = np.random.RandomState(int(seed))
    idx = rng.permutation(dim)[:keep]
    mask = np.zeros(dim, dtype=np.float32)
    mask[idx] = 1.0
    return mask


def knn_drop_scores(
    train_feats: np.ndarray,
    test_feats: np.ndarray,
    k: int,
    drop_percent: float,
    seed: int,
    renormalize: bool,
    use_faiss: bool = True,
):
    mask = make_drop_mask(train_feats.shape[1], drop_percent=drop_percent, seed=seed)
    train_drop = train_feats * mask[None, :]
    test_drop = test_feats * mask[None, :]
    if renormalize:
        train_norm = np.linalg.norm(train_drop, axis=1, keepdims=True) + 1e-12
        test_norm = np.linalg.norm(test_drop, axis=1, keepdims=True) + 1e-12
        train_drop = train_drop / train_norm
        test_drop = test_drop / test_norm
    return knn_scores(train_drop, test_drop, k=k, use_faiss=use_faiss)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg["checkpoint"], map_location="cpu")
    proj_dim = infer_proj_dim(ckpt["model"])
    model = ResNet18Backbone(num_classes=10, proj_dim=proj_dim)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    train_dataset_full = get_cifar10(cfg["data_root"], train=True)

    val_ratio = float(cfg["val"]["ratio"])
    seed = int(cfg["seed"])
    n_train = len(train_dataset_full)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_train)
    n_val = max(1, int(round(n_train * val_ratio)))
    val_idx = perm[:n_val]
    bank_idx = perm[n_val:]

    # train_dataset_eval = datasets.CIFAR10(
    #     root=cfg["data_root"],
    #     train=True,
    #     download=True,
    #     transform=_cifar_transform(False),
    # )

    # train_loader = make_loader(Subset(train_dataset_eval, bank_idx.tolist()), cfg["batch_size"], cfg["num_workers"])
    # val_loader = make_loader(Subset(train_dataset_eval, val_idx.tolist()), cfg["batch_size"], cfg["num_workers"])
    # test_id_loader = make_loader(get_cifar10(cfg["data_root"], train=False), cfg["batch_size"], cfg["num_workers"])


# Deterministic transform (same as your CIFAR test transform)
    mean, std = CIFAR_STATS
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Use TRAIN split but with eval transform for the feature bank + val split
    train_dataset_eval = datasets.CIFAR10(
        root=cfg["data_root"], train=True, download=True, transform=eval_tf
    )

    # Use TEST split with eval transform for ID test
    test_dataset_eval = datasets.CIFAR10(
        root=cfg["data_root"], train=False, download=True, transform=eval_tf
    )

    train_loader    = make_loader(Subset(train_dataset_eval, bank_idx.tolist()),
                                cfg["batch_size"], cfg["num_workers"], shuffle=False)
    val_loader      = make_loader(Subset(train_dataset_eval, val_idx.tolist()),
                                cfg["batch_size"], cfg["num_workers"], shuffle=False)
    test_id_loader  = make_loader(test_dataset_eval,
                                cfg["batch_size"], cfg["num_workers"], shuffle=False)


    train_feats, _ = extract_features(model, train_loader, device)
    val_feats, _ = extract_features(model, val_loader, device)
    id_feats, _ = extract_features(model, test_id_loader, device)

    if cfg["knn"]["normalize"]:
        train_feats = l2_normalize(torch.from_numpy(train_feats)).numpy()
        val_feats = l2_normalize(torch.from_numpy(val_feats)).numpy()
        id_feats = l2_normalize(torch.from_numpy(id_feats)).numpy()

    methods = cfg.get("ood_methods", ["knn"])

    method_specs = {}
    if "knn" in methods:
        val_scores_knn = knn_scores(train_feats, val_feats, cfg["knn"]["k"], cfg["knn"]["use_faiss"])
        tau_knn = fit_threshold_high_is_id(val_scores_knn, cfg["val"]["target_tpr"])
        id_scores_knn = knn_scores(train_feats, id_feats, cfg["knn"]["k"], cfg["knn"]["use_faiss"])
        method_specs["knn"] = {"id_scores": id_scores_knn, "tau": tau_knn}

    if "knn_drop" in methods:
        drop_cfg = cfg["knn_drop"]
        val_scores_drop = knn_drop_scores(
            train_feats,
            val_feats,
            k=cfg["knn"]["k"],
            drop_percent=drop_cfg["drop_percent"],
            seed=drop_cfg["seed"],
            renormalize=drop_cfg["renormalize"],
            use_faiss=cfg["knn"]["use_faiss"],
        )
        tau_drop = fit_threshold_high_is_id(val_scores_drop, cfg["val"]["target_tpr"])
        id_scores_drop = knn_drop_scores(
            train_feats,
            id_feats,
            k=cfg["knn"]["k"],
            drop_percent=drop_cfg["drop_percent"],
            seed=drop_cfg["seed"],
            renormalize=drop_cfg["renormalize"],
            use_faiss=cfg["knn"]["use_faiss"],
        )
        method_specs["knn_drop"] = {"id_scores": id_scores_drop, "tau": tau_drop}

    print("method,dataset,tau,id_accept_rate_val,fpr95,auroc")
    for dname in cfg["ood_datasets"]:
        ood_loader = make_loader(get_ood_dataset(dname, cfg["data_root"]), cfg["batch_size"], cfg["num_workers"])
        ood_feats, _ = extract_features(model, ood_loader, device)
        if cfg["knn"]["normalize"]:
            ood_feats = l2_normalize(torch.from_numpy(ood_feats)).numpy()

        for method, spec in method_specs.items():
            if method == "knn":
                ood_scores = knn_scores(train_feats, ood_feats, cfg["knn"]["k"], cfg["knn"]["use_faiss"])
            elif method == "knn_drop":
                ood_scores = knn_drop_scores(
                    train_feats,
                    ood_feats,
                    k=cfg["knn"]["k"],
                    drop_percent=cfg["knn_drop"]["drop_percent"],
                    seed=cfg["knn_drop"]["seed"],
                    renormalize=cfg["knn_drop"]["renormalize"],
                    use_faiss=cfg["knn"]["use_faiss"],
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            id_scores = spec["id_scores"]
            tau = spec["tau"]
            id_accept = float((id_scores >= tau).mean() * 100.0)
            print(f"{method},{dname},{tau:.4f},{id_accept:.2f},{fpr95(id_scores, ood_scores):.2f},{auroc(id_scores, ood_scores):.2f}")


if __name__ == "__main__":
    main()
