import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from configs import get_cfg
from data import get_loader
from model import get_model
from util.net import init_training
from util.util import run_pre


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight CPR prototype-matching analysis for ablation sections."
    )
    parser.add_argument(
        "-c",
        "--cfg_path",
        default="configs/uniad/uniad_mvtec_cpr.py",
        help="Config path used to build loader/model.",
    )
    parser.add_argument("--checkpoint_path", default="", help="Optional model checkpoint override.")
    parser.add_argument("--bank_path", default="", help="Optional bank checkpoint override.")
    parser.add_argument(
        "--save_dir",
        default="visualizations/cpr_matching",
        help="Directory for summary table and figures.",
    )
    parser.add_argument("--max_eval_batches", type=int, default=-1, help="Limit test batches for quick checks.")
    parser.add_argument("--tsne_max_samples", type=int, default=240, help="Max sample-level features in t-SNE.")
    parser.add_argument("--case_rows", type=int, default=4, help="How many normal cases to visualize.")
    return parser.parse_args()


def build_cfg(args):
    cfg_terminal = argparse.Namespace(
        cfg_path=args.cfg_path,
        mode="test",
        sleep=-1,
        memory=-1,
        dist_url="env://",
        logger_rank=0,
        opts=[],
    )
    cfg = get_cfg(cfg_terminal)
    if args.checkpoint_path:
        cfg.model.kwargs["checkpoint_path"] = args.checkpoint_path
    if args.bank_path:
        cfg.model.kwargs["bank_path"] = args.bank_path
    cfg.vis = False
    cfg.trainer.data.drop_last = False
    return cfg


def denormalize_image(img_tensor):
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return img


def compute_global_patch_assignment(feat_align, mode_proto_maps, mode_to_class):
    b, c, h, w = feat_align.shape
    proto = mode_proto_maps.to(feat_align.device)
    feat_flat = F.normalize(feat_align.view(b, c, h * w), dim=1)
    proto_flat = F.normalize(proto.view(proto.shape[0], c, h * w), dim=1)
    sim = torch.einsum("bch,kch->bkh", feat_flat, proto_flat)
    top_mode = torch.argmax(sim, dim=1).view(b, h, w)
    top_class = mode_to_class.to(feat_align.device)[top_mode]
    return top_mode, top_class


def sample_tsne_indices(labels, max_samples):
    labels = np.asarray(labels)
    if len(labels) <= max_samples:
        return np.arange(len(labels))

    rng = np.random.default_rng(42)
    keep = []
    unique_labels = np.unique(labels)
    per_class = max(1, max_samples // max(1, len(unique_labels)))
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        take = min(per_class, len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep = np.array(sorted(set(keep)))
    if len(keep) < max_samples:
        remain = np.setdiff1d(np.arange(len(labels)), keep)
        extra = rng.choice(remain, size=min(len(remain), max_samples - len(keep)), replace=False)
        keep = np.sort(np.concatenate([keep, extra]))
    return keep[:max_samples]


def save_summary_csv(path, metrics):
    rows = [
        ("sample_center_acc_all", metrics["sample_center_acc_all"]),
        ("sample_center_acc_normal", metrics["sample_center_acc_normal"]),
        ("normal_patch_purity", metrics["normal_patch_purity"]),
        ("normal_cross_class_ratio", metrics["normal_cross_class_ratio"]),
        ("abnormal_region_cross_class_ratio", metrics["abnormal_region_cross_class_ratio"]),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in rows:
            writer.writerow([name, f"{value:.6f}"])


def save_tsne_figure(path, sample_feats, sample_labels, class_centers, class_names):
    if len(sample_feats) < 5:
        return

    data = np.concatenate([sample_feats, class_centers], axis=0)
    perplexity = min(30, max(5, len(sample_feats) // 6))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
    embed = tsne.fit_transform(data)

    sample_embed = embed[: len(sample_feats)]
    center_embed = embed[len(sample_feats):]

    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab20", len(class_names))
    for idx, cls_name in enumerate(class_names):
        mask = np.asarray(sample_labels) == idx
        if mask.sum() == 0:
            continue
        plt.scatter(
            sample_embed[mask, 0],
            sample_embed[mask, 1],
            s=18,
            alpha=0.65,
            color=cmap(idx),
            label=cls_name,
        )
    plt.scatter(center_embed[:, 0], center_embed[:, 1], c="black", marker="*", s=180, label="class center")
    for idx, cls_name in enumerate(class_names):
        plt.text(center_embed[idx, 0], center_embed[idx, 1], cls_name, fontsize=9, ha="left", va="bottom")
    plt.title("CPR Sample-Level Feature t-SNE")
    plt.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_case_figure(path, cases, class_names):
    if not cases:
        return

    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    cmap = plt.get_cmap("tab20", len(class_names))

    for row, case in enumerate(cases):
        axes[row, 0].imshow(case["image"])
        axes[row, 0].set_title(f"Image\n{case['cls_name']}")
        axes[row, 1].imshow(case["class_map"], cmap=cmap, vmin=0, vmax=max(1, len(class_names) - 1))
        axes[row, 1].set_title(
            f"Retrieved Class Map\ncenter={case['pred_center']} wrong={case['wrong_ratio']:.2%}"
        )
        axes[row, 2].imshow(case["wrong_map"], cmap="Reds", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("Cross-Class Mask")
        axes[row, 3].imshow(case["anomaly_map"], cmap="jet")
        axes[row, 3].set_title("Anomaly Map")
        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle("CPR Prototype Matching Cases (normal samples)", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    run_pre(cfg)
    init_training(cfg)

    loaders = get_loader(cfg)
    test_loader = loaders[1] if isinstance(loaders, tuple) else loaders["test"]

    model = get_model(cfg.model)
    model = model.cuda().eval()
    net = model.module if hasattr(model, "module") else model

    class_names = list(net.bank.class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    sample_total = 0
    sample_correct = 0
    normal_sample_total = 0
    normal_sample_correct = 0
    normal_patch_total = 0
    normal_patch_correct = 0
    abnormal_patch_total = 0
    abnormal_patch_wrong = 0

    tsne_feats = []
    tsne_labels = []
    case_pool = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break

            imgs = data["img"].cuda()
            masks = data["img_mask"].cuda()
            cls_names = data["cls_name"]
            anomaly = data["anomaly"].cuda()

            feats_backbone = net.net_backbone(imgs)
            feat_align = net.net_merge(feats_backbone)
            g_code = F.normalize(feat_align.mean(dim=(2, 3)), dim=1)

            class_centers = F.normalize(net.bank.class_centers.cuda(), dim=1)
            class_scores = torch.matmul(g_code, class_centers.t())
            pred_class_idx = torch.argmax(class_scores, dim=1)
            true_class_idx = torch.tensor([class_to_idx[name] for name in cls_names], device=imgs.device)

            top_mode_idx, top_mode_class = compute_global_patch_assignment(
                feat_align.detach(),
                net.bank.mode_proto_maps,
                net.bank.mode_to_class,
            )

            _, _, pred, extra = net(imgs)
            anomaly_map = pred.detach().cpu().numpy()[:, 0]

            sample_total += imgs.shape[0]
            sample_correct += (pred_class_idx == true_class_idx).sum().item()

            for i in range(imgs.shape[0]):
                is_normal = int(anomaly[i].item()) == 0
                patch_correct = top_mode_class[i] == true_class_idx[i]
                wrong_ratio = 1.0 - patch_correct.float().mean().item()

                if is_normal:
                    normal_sample_total += 1
                    normal_sample_correct += int(pred_class_idx[i].item() == true_class_idx[i].item())
                    normal_patch_total += patch_correct.numel()
                    normal_patch_correct += patch_correct.sum().item()
                    case_pool.append(
                        {
                            "wrong_ratio": wrong_ratio,
                            "cls_name": cls_names[i],
                            "pred_center": class_names[pred_class_idx[i].item()],
                            "image": denormalize_image(imgs[i]),
                            "class_map": top_mode_class[i].cpu().numpy(),
                            "wrong_map": (~patch_correct).float().cpu().numpy(),
                            "anomaly_map": anomaly_map[i],
                        }
                    )
                    tsne_feats.append(g_code[i].cpu().numpy())
                    tsne_labels.append(true_class_idx[i].item())
                else:
                    mask = F.interpolate(
                        masks[i : i + 1].float(),
                        size=top_mode_class.shape[-2:],
                        mode="nearest",
                    )[0, 0] > 0.5
                    if mask.any():
                        abnormal_patch_total += mask.sum().item()
                        abnormal_patch_wrong += (top_mode_class[i][mask] != true_class_idx[i]).sum().item()

    metrics = {
        "sample_center_acc_all": sample_correct / max(sample_total, 1),
        "sample_center_acc_normal": normal_sample_correct / max(normal_sample_total, 1),
        "normal_patch_purity": normal_patch_correct / max(normal_patch_total, 1),
        "normal_cross_class_ratio": 1.0 - (normal_patch_correct / max(normal_patch_total, 1)),
        "abnormal_region_cross_class_ratio": abnormal_patch_wrong / max(abnormal_patch_total, 1),
    }

    save_summary_csv(save_dir / "matching_summary.csv", metrics)

    if tsne_feats:
        tsne_feats = np.stack(tsne_feats, axis=0)
        tsne_labels = np.asarray(tsne_labels)
        keep = sample_tsne_indices(tsne_labels, args.tsne_max_samples)
        save_tsne_figure(
            save_dir / "sample_center_tsne.png",
            tsne_feats[keep],
            tsne_labels[keep],
            F.normalize(net.bank.class_centers, dim=1).cpu().numpy(),
            class_names,
        )

    case_pool.sort(key=lambda x: x["wrong_ratio"], reverse=True)
    save_case_figure(save_dir / "retrieval_cases.png", case_pool[: args.case_rows], class_names)

    print("Saved CPR matching analysis to:", save_dir)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
