
# 三层锚点 bank 构建脚本：
# class anchor：先按图像级语义选类别
# sub-mode anchor：再在类内选最接近的子模式
# patch anchor：最后只在选中的少量 mode 上做 patch 级 prior 检索# 

# 新 bank 构造：
# 输出 class_centers / mode_centers / mode_proto_maps / mode_radius / mode_to_class
# 比原来的单层 prototype 池更贴近你论文里的类别锚点叙事

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from configs import get_cfg
from data import get_loader
from model import get_model
from util.net import init_training
from util.util import init_checkpoint, run_pre

try:
    from sklearn_extra.cluster import KMedoids
except Exception:
    KMedoids = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg_path", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cluster", choices=["kmeans", "gmm", "agglomerative", "kmedoids"], default="kmeans")
    # parser.add_argument("--save_path", default="runs/prototypes/mvtec_bank.pth")
    # parser.add_argument("--save_path", default="runs/prototypes/visa_bank.pth")
    # parser.add_argument("--save_path", default="runs/prototypes/mvtec_loco_bank.pth")
    parser.add_argument("--save_path", default="runs/prototypes/mpdd_bank.pth")
    # parser.add_argument("--save_path", default="runs/prototypes/btad_bank.pth")
    return parser.parse_args()


@torch.no_grad()
def extract_normal_features_per_class(model, loader):
    model.eval()
    feats_by_class = {}
    vecs_by_class = {}

    for data in tqdm(loader, desc="Extract normal features per class"):
        imgs = data["img"].cuda()
        cls_names = data["cls_name"]
        anomaly = data["anomaly"]
        mask = anomaly == 0
        if mask.sum() == 0:
            continue

        imgs = imgs[mask]
        cls_names = np.array(cls_names)[mask.cpu().numpy()]
        feats_t, _, _ = model(imgs)
        feats_map = feats_t
        feats_vec = F.normalize(feats_map.mean(dim=(2, 3)), dim=1)

        for fm, fv, cls in zip(feats_map, feats_vec, cls_names):
            feats_by_class.setdefault(cls, []).append(fm.cpu())
            vecs_by_class.setdefault(cls, []).append(fv.cpu())

    return feats_by_class, vecs_by_class


def cluster_vectors(vecs_np, cluster, k):
    n = vecs_np.shape[0]
    k_use = min(k, n)
    if k_use <= 1:
        return np.zeros(n, dtype=np.int64), 1

    if cluster == "kmeans":
        algo = KMeans(n_clusters=k_use, random_state=0, n_init="auto")
        labels = algo.fit_predict(vecs_np)
    elif cluster == "gmm":
        algo = GaussianMixture(n_components=k_use, covariance_type="full", random_state=0)
        labels = algo.fit_predict(vecs_np)
    elif cluster == "agglomerative":
        algo = AgglomerativeClustering(n_clusters=k_use)
        labels = algo.fit_predict(vecs_np)
    elif cluster == "kmedoids":
        if KMedoids is None:
            raise RuntimeError("sklearn-extra is required for kmedoids clustering.")
        algo = KMedoids(n_clusters=k_use, random_state=0, metric="euclidean")
        labels = algo.fit_predict(vecs_np)
    else:
        raise ValueError(cluster)
    return labels.astype(np.int64), k_use


def build_three_level_bank(feats_by_class, vecs_by_class, cluster, k):
    class_names = sorted(feats_by_class.keys())
    class_centers = []
    mode_centers = []
    mode_proto_maps = []
    mode_radius = []
    mode_counts = []
    mode_to_class = []

    for class_idx, cls_name in enumerate(class_names):
        feat_maps = torch.stack(feats_by_class[cls_name], dim=0)   # [N,C,H,W]
        feat_vecs = torch.stack(vecs_by_class[cls_name], dim=0)    # [N,C]
        feat_vecs = F.normalize(feat_vecs, dim=1)

        class_center = F.normalize(feat_vecs.mean(dim=0), dim=0)
        class_centers.append(class_center)

        labels, k_use = cluster_vectors(feat_vecs.numpy(), cluster, k)
        for mode_idx in range(k_use):
            idx = np.where(labels == mode_idx)[0]
            cluster_maps = feat_maps[idx]
            cluster_vecs = feat_vecs[idx]
            center = F.normalize(cluster_vecs.mean(dim=0), dim=0)

            dists = torch.norm(cluster_vecs - center.unsqueeze(0), dim=1)
            medoid_idx = torch.argmin(dists).item()
            proto_map = cluster_maps[medoid_idx]
            radius = dists.mean().item() + 1e-6

            mode_centers.append(center)
            mode_proto_maps.append(proto_map)
            mode_radius.append(radius)
            mode_counts.append(float(len(idx)))
            mode_to_class.append(class_idx)

    bank = {
        "class_names": class_names,
        "class_centers": torch.stack(class_centers, dim=0).cpu(),
        "mode_centers": torch.stack(mode_centers, dim=0).cpu(),
        "mode_proto_maps": torch.stack(mode_proto_maps, dim=0).cpu(),
        "mode_radius": torch.tensor(mode_radius, dtype=torch.float32),
        "mode_counts": torch.tensor(mode_counts, dtype=torch.float32),
        "mode_to_class": torch.tensor(mode_to_class, dtype=torch.long),
        "cluster": cluster,
        "k": int(k),
    }
    return bank


def main():
    args = parse_args()
    cfg_terminal = argparse.Namespace(
        cfg_path=args.cfg_path,
        mode="train",
        sleep=-1,
        memory=-1,
        dist_url="env://",
        logger_rank=0,
        opts=[],
    )
    cfg = get_cfg(cfg_terminal)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)

    model_cfg = Namespace()
    model_cfg.name = "uniad"
    model_cfg.kwargs = dict(
        pretrained=False,
        checkpoint_path="",
        strict=True,
        model_backbone=cfg.model_backbone,
        model_decoder=cfg.model_decoder,
    )
    model = get_model(model_cfg)
    model.cuda()
    model.eval()

    loaders = get_loader(cfg)
    train_loader = loaders[0] if isinstance(loaders, tuple) else loaders["train"]
    feats_by_class, vecs_by_class = extract_normal_features_per_class(model, train_loader)
    bank = build_three_level_bank(feats_by_class, vecs_by_class, args.cluster, args.k)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, save_path)
    print(f">>> Saved CPR bank to {save_path}")
    print(
        f">>> classes={len(bank['class_names'])}, modes={bank['mode_centers'].shape[0]}, "
        f"cluster={bank['cluster']}, k={bank['k']}"
    )


if __name__ == "__main__":
    main()

