# 新 CPR：
# 不让原型主导重建
# 用 conf_map + gate 对原始输入特征做“轻度校正”
# 公式上是输入主导、原型辅助、可信度调制

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_model, MODEL
from model.uniad import MFCN, UniAD_decoder, Namespace


def _minmax_norm(x, eps=1e-6):
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


class DiscHeadV2(nn.Module):
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
        )
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        res = x
        x = self.block(x)
        x = self.act(x + res)
        return self.head(x)


class ThreeLevelAnchorBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_names = []
        self.class_to_idx = {}

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        bank = cls()

        class_names = ckpt["class_names"]
        if isinstance(class_names, (list, tuple)):
            bank.class_names = [str(x) for x in class_names]
        else:
            bank.class_names = [str(x) for x in class_names.tolist()]
        bank.class_to_idx = {name: idx for idx, name in enumerate(bank.class_names)}

        bank.register_buffer("class_centers", ckpt["class_centers"].float())
        bank.register_buffer("mode_centers", ckpt["mode_centers"].float())
        bank.register_buffer("mode_proto_maps", ckpt["mode_proto_maps"].float())
        bank.register_buffer("mode_radius", ckpt["mode_radius"].float())
        bank.register_buffer("mode_counts", ckpt["mode_counts"].float())
        bank.register_buffer("mode_to_class", ckpt["mode_to_class"].long())
        return bank

    @torch.no_grad()
    def retrieve_three_level(self, feat_align, g_code, top_modes=2, tau=0.07, use_cosine=True):
        device = feat_align.device
        b, c, h, w = feat_align.shape
        k_total = self.mode_proto_maps.shape[0]

        g = F.normalize(g_code, dim=1)
        class_centers = F.normalize(self.class_centers.to(device), dim=1)
        class_scores = torch.matmul(g, class_centers.t())
        class_idx = torch.argmax(class_scores, dim=1)

        prior_list = []
        dist_list = []
        conf_list = []
        mode_idx_list = []
        mode_score_list = []

        feat_flat = feat_align.view(b, c, h * w)

        for i in range(b):
            cls_idx = class_idx[i]
            cls_mask = self.mode_to_class.to(device) == cls_idx
            candidate_idx = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            if candidate_idx.numel() == 0:
                candidate_idx = torch.arange(k_total, device=device)

            cand_centers = F.normalize(self.mode_centers[candidate_idx].to(device), dim=1)
            mode_scores = torch.matmul(g[i : i + 1], cand_centers.t()).squeeze(0)
            topk = min(int(top_modes), int(candidate_idx.numel()))
            top_scores, top_pos = torch.topk(mode_scores, k=topk, dim=0)
            chosen_idx = candidate_idx[top_pos]

            proto_maps = self.mode_proto_maps[chosen_idx].to(device)
            proto_flat = proto_maps.view(topk, c, h * w)
            cur_feat = feat_flat[i : i + 1]

            if use_cosine:
                f_n = F.normalize(cur_feat, dim=1)
                p_n = F.normalize(proto_flat, dim=1)
                sim = torch.einsum("bch,mch->bmh", f_n, p_n)
                weights = torch.softmax(sim / tau, dim=1)
                dist = (1.0 - sim).clamp(min=0.0)
            else:
                f2 = (cur_feat ** 2).sum(dim=1, keepdim=True)
                p2 = (proto_flat ** 2).sum(dim=1).unsqueeze(0)
                fp = torch.einsum("bch,mch->bmh", cur_feat, proto_flat)
                dist = (f2 + p2 - 2 * fp).clamp(min=0.0).sqrt()
                weights = torch.softmax(-dist / tau, dim=1)

            prior = torch.einsum("bmh,mch->bch", weights, proto_flat).view(1, c, h, w)
            dist_min = dist.min(dim=1)[0].view(1, 1, h, w)

            chosen_radius = self.mode_radius[chosen_idx].to(device)
            radius_patch = torch.einsum("bmh,m->bh", weights, chosen_radius).view(1, 1, h, w)
            conf_patch = torch.exp(-dist_min / (radius_patch + 1e-6))
            global_conf = torch.sigmoid(top_scores.mean()).view(1, 1, 1, 1)
            conf = conf_patch * global_conf

            prior_list.append(prior)
            dist_list.append(dist_min)
            conf_list.append(conf)
            mode_idx_list.append(chosen_idx.detach().cpu())
            mode_score_list.append(top_scores.detach().cpu())

        return {
            "prior_align": torch.cat(prior_list, dim=0),
            "dist_min": torch.cat(dist_list, dim=0),
            "conf_map": torch.cat(conf_list, dim=0).clamp(0.0, 1.0),
            "class_idx": class_idx,
            "class_scores": class_scores,
            "mode_indices": mode_idx_list,
            "mode_scores": mode_score_list,
        }


class WeakPriorGate(nn.Module):
    def __init__(self, channels, hidden_ratio=0.5):
        super().__init__()
        hidden = max(32, int(channels * hidden_ratio))
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, prior, conf):
        residual_mag = torch.abs(feat - prior)
        return self.net(torch.cat([feat, prior, residual_mag, conf], dim=1))


class UniAD_CPR(nn.Module):
    def __init__(
        self,
        model_backbone,
        model_decoder,
        bank_path,
        tau=0.07,
        use_cosine=True,
        disc_mid=128,
        lambda_disc=0.20,
        fuse_mode="mul",
        top_modes=2,
        correction_strength=0.25,
        gate_hidden_ratio=0.5,
        conf_power=1.0,
        conf_floor=0.05,
    ):
        super().__init__()

        self.tau = float(tau)
        self.use_cosine = bool(use_cosine)
        self.lambda_disc = float(lambda_disc)
        self.fuse_mode = fuse_mode
        self.top_modes = int(top_modes)
        self.correction_strength = float(correction_strength)
        self.conf_power = float(conf_power)
        self.conf_floor = float(conf_floor)

        self.net_backbone = get_model(model_backbone)
        self.net_merge = MFCN(
            inplanes=model_decoder["inplanes"],
            outplanes=model_decoder["outplanes"],
            instrides=[2, 4, 8, 16],
            outstrides=[16],
        )
        self.net_ad = UniAD_decoder(
            inplanes=model_decoder["outplanes"],
            instrides=model_decoder["instrides"],
            feature_size=model_decoder["feature_size"],
            feature_jitter=Namespace(**{"scale": 20.0, "prob": 1.0}),
            neighbor_mask=Namespace(
                **{"neighbor_size": model_decoder["neighbor_size"], "mask": [True, True, True]}
            ),
            hidden_dim=256,
            pos_embed_type="learned",
            save_recon=Namespace(**{"save_dir": "result_recon"}),
            initializer={"method": "xavier_uniform"},
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )

        self.bank = ThreeLevelAnchorBank.load(bank_path).cuda()
        channels = model_decoder["outplanes"][0]
        self.prior_gate = WeakPriorGate(channels, hidden_ratio=gate_hidden_ratio)
        self.disc_head = DiscHeadV2(channels, mid=disc_mid)

        self.frozen_layers = ["net_backbone"]
        self.lambda_disc_max = float(lambda_disc)
        self.lambda_disc_start_epoch = 150
        self.lambda_disc_warmup = 100

    def train(self, mode=True):
        self.training = mode
        for name, m in self.named_children():
            if name in self.frozen_layers:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            else:
                m.train(mode)
        return self

    def _current_lambda_disc(self):
        e = self.current_epoch if hasattr(self, "current_epoch") else 0
        if e < self.lambda_disc_start_epoch:
            return e, 0.0
        t = min((e - self.lambda_disc_start_epoch) / float(self.lambda_disc_warmup), 1.0)
        return e, self.lambda_disc_max * t

    def forward(self, imgs):
        feats_backbone = self.net_backbone(imgs)
        feat_align = self.net_merge(feats_backbone)
        feat_detached = feat_align.detach()
        g_code = feat_detached.mean(dim=(2, 3))

        bank_out = self.bank.retrieve_three_level(
            feat_align=feat_detached,
            g_code=g_code,
            top_modes=self.top_modes,
            tau=self.tau,
            use_cosine=self.use_cosine,
        )
        prior_align = bank_out["prior_align"]
        proto_dist = F.interpolate(
            bank_out["dist_min"],
            size=(feat_detached.shape[-2], feat_detached.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        conf_map = bank_out["conf_map"].clamp(self.conf_floor, 1.0).pow(self.conf_power)

        gate = self.prior_gate(feat_detached, prior_align, conf_map)
        feat_cpr = feat_detached + self.correction_strength * conf_map * gate * (prior_align - feat_detached)

        _, feature_rec, _ = self.net_ad(feat_cpr)
        pred_rec = torch.sqrt(torch.sum((feature_rec - feat_detached) ** 2, dim=1, keepdim=True))
        pred_rec = self.net_ad.upsample(pred_rec)

        score_disc = self.disc_head(feat_align)
        score_disc = F.interpolate(
            score_disc,
            size=pred_rec.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        proto_dist_up = F.interpolate(
            proto_dist,
            size=pred_rec.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        conf_up = F.interpolate(conf_map, size=pred_rec.shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            extra = {
                "score_disc": score_disc,
                "proto_dist": proto_dist_up,
                "feat_align": feat_align,
                "prior_align": prior_align,
                "conf_map": conf_up,
            }
            return feat_detached, feature_rec, pred_rec, extra

        epoch, lambda_disc = self._current_lambda_disc()
        if (not torch.jit.is_tracing()) and (
            not hasattr(self, "_last_lambda_disc") or abs(lambda_disc - self._last_lambda_disc) > 1e-6
        ):
            print(f"[CPR] epoch={epoch:4d} | lambda_disc={lambda_disc:.4f}")
            self._last_lambda_disc = lambda_disc

        score_n = _minmax_norm(score_disc) * conf_up
        if self.fuse_mode == "mul":
            pred_fuse = pred_rec * (1.0 + lambda_disc * score_n)
        else:
            pred_fuse = pred_rec + lambda_disc * score_n

        extra = {
            "pred_rec": pred_rec,
            "pred_fuse": pred_fuse,
            "score_disc": score_disc,
            "proto_dist": proto_dist_up,
            "prior_align": prior_align,
            "conf_map": conf_up,
        }
        return feat_detached, feature_rec, pred_fuse, extra


@MODEL.register_module
def uniad_cpr(pretrained=False, **kwargs):
    return UniAD_CPR(**kwargs)
