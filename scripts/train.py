#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knn_ood.datasets import get_cifar10, get_cifar10_two_crop, make_loader
from knn_ood.losses import SupConLoss
from knn_ood.models import ResNet18Backbone
from knn_ood.utils import load_yaml, make_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["ce", "supcon"], required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    make_dir(cfg["save_dir"])

    if args.mode == "supcon":
        train_set = get_cifar10_two_crop(cfg["data"]["root"])
    else:
        train_set = get_cifar10(cfg["data"]["root"], train=True)
    test_set = get_cifar10(cfg["data"]["root"], train=False)
    train_loader = make_loader(train_set, cfg["data"]["batch_size"], cfg["data"]["num_workers"], shuffle=True)
    test_loader = make_loader(test_set, cfg["data"]["batch_size"], cfg["data"]["num_workers"], shuffle=False)

    proj_dim = cfg["model"].get("proj_dim") if args.mode == "supcon" else None
    model = ResNet18Backbone(num_classes=cfg["model"]["num_classes"], proj_dim=proj_dim).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    if args.mode == "ce":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg["train"]["milestones"], gamma=cfg["train"]["gamma"]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ce_loss = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss(temperature=cfg["train"].get("temperature", 0.1))

    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['train']['epochs']}", leave=False):
            optimizer.zero_grad()
            if args.mode == "ce":
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x, return_features=True)
                loss = ce_loss(logits, y)
            else:
                x1, x2 = x
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                x_cat = torch.cat([x1, x2], dim=0)
                logits_cat, _, proj_cat = model(x_cat, return_features=True)
                bsz = y.size(0)
                logits1, logits2 = logits_cat[:bsz], logits_cat[bsz:]
                proj1, proj2 = proj_cat[:bsz], proj_cat[bsz:]
                ce = 0.5 * (ce_loss(logits1, y) + ce_loss(logits2, y))
                sup = supcon_loss(torch.cat([proj1, proj2], dim=0), torch.cat([y, y], dim=0))
                loss = ce + sup
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc, "epoch": epoch}, os.path.join(cfg["save_dir"], "best.pt"))
        print(f"Epoch {epoch+1}: acc={acc:.2f}, best={best_acc:.2f}")


if __name__ == "__main__":
    main()
