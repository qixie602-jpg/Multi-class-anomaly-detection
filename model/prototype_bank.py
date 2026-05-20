# ============================================================
# PrototypeBank (Hard Routing Version)
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """
    Prototype Bank for CPR / PNPT-style usage

    - Support HARD routing (argmax, one-hot)  ✅【主用】
    - Keep SOFT routing for ablation (optional)

    Buffers:
        p_mean  : [K, C, H, W]   prior feature maps
        p_m2    : [K, C, H, W]   second moment (optional, for var)
        g_codes : [K, g_dim]     global codes for retrieval
        counts  : [K]            (kept for compatibility)
    """

    def __init__(
        self,
        K,
        g_dim,
        C,
        H,
        W,
        momentum=0.99,
        eps=1e-6,
        tau=0.1,   # only used by soft routing
    ):
        super().__init__()

        self.K = int(K)
        self.g_dim = int(g_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.tau = float(tau)

        self.register_buffer("p_mean", torch.zeros(K, C, H, W))
        self.register_buffer("p_m2", torch.zeros(K, C, H, W))
        self.register_buffer("g_codes", torch.zeros(K, g_dim))
        self.register_buffer("counts", torch.zeros(K))

    # ============================================================
    # Init
    # ============================================================
    def init_random(self, std=0.02):
        nn.init.normal_(self.p_mean, mean=0.0, std=std)
        nn.init.normal_(self.p_m2, mean=std ** 2, std=std)
        nn.init.normal_(self.g_codes, mean=0.0, std=std)
        self.counts.zero_()

    # ============================================================
    # HARD Routing  ★ 推荐使用
    # ============================================================
    @torch.no_grad()
    def route(self, g, return_sim=True):
        """
        Hard routing by cosine similarity (argmax).

        Args:
            g : [B, g_dim]
        Returns:
            weights : [B, K]   one-hot assignment
            sim     : [B, K]   cosine similarity
        """
        g = F.normalize(g, dim=1)
        proto = F.normalize(self.g_codes, dim=1)

        sim = torch.matmul(g, proto.t())      # [B, K]
        idx = torch.argmax(sim, dim=1)        # [B]

        weights = torch.zeros_like(sim)
        weights.scatter_(1, idx[:, None], 1.0)

        return (weights, sim) if return_sim else weights

    # ============================================================
    # Prior access
    # ============================================================
    def get_prior_mean_soft(self, weights):
        """
        Args:
            weights : [B, K]  (one-hot or soft)
        Returns:
            mean : [B, C, H, W]
        """
        return torch.einsum("bk,kchw->bchw", weights, self.p_mean)

    def get_prior_var_soft(self, weights):
        mean = self.get_prior_mean_soft(weights)
        m2 = torch.einsum("bk,kchw->bchw", weights, self.p_m2)
        var = m2 - mean ** 2
        return torch.clamp(var, min=self.eps)

    # ============================================================
    # EMA Update  ❌【PNPT-style / 聚类先验场景：不推荐使用】
    # ============================================================
    @torch.no_grad()
    def update(self, weights, g, feat_align):
        """
        保留接口，仅用于兼容旧实验。
        在你当前设定（离线聚类 + 硬检索）下，
        训练/推理阶段应【禁止调用】此函数。
        """
        B, K = weights.shape
        g = F.normalize(g, dim=1)

        w_sum = weights.sum(dim=0)

        for k in range(K):
            wk = w_sum[k]
            if wk < 1e-6:
                continue

            w = weights[:, k].view(B, 1, 1, 1)
            mean_k = (w * feat_align).sum(dim=0) / wk
            m2_k = (w * feat_align.pow(2)).sum(dim=0) / wk
            g_k = (weights[:, k].view(B, 1) * g).sum(dim=0) / wk

            if self.counts[k] < 1:
                self.p_mean[k].copy_(mean_k)
                self.p_m2[k].copy_(m2_k)
                self.g_codes[k].copy_(g_k)
            else:
                m = self.momentum
                self.p_mean[k].mul_(m).add_(mean_k, alpha=1.0 - m)
                self.p_m2[k].mul_(m).add_(m2_k, alpha=1.0 - m)
                self.g_codes[k].mul_(m).add_(g_k, alpha=1.0 - m)

            self.counts[k] += wk

    # ============================================================
    # Save / Load
    # ============================================================
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "K": self.K,
                "g_dim": self.g_dim,
                "momentum": self.momentum,
                "eps": self.eps,
                "tau": self.tau,
                "p_mean": self.p_mean.cpu(),
                "p_m2": self.p_m2.cpu(),
                "g_codes": self.g_codes.cpu(),
                "counts": self.counts.cpu(),
            },
            path,
        )

    @staticmethod
    def load(path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)

        # ===== 新增：兼容 extract_cluster_features.py 生成的 bank =====
        if "p_align" in ckpt:
            p_mean = ckpt["p_align"]            # [K,C,H,W]
            K, C, H, W = p_mean.shape
            g_codes = ckpt["g_codes"]

            bank = PrototypeBank(
                K=K,
                g_dim=g_codes.shape[1],
                C=C, H=H, W=W,
                momentum=ckpt.get("momentum", 0.99),
            )

            bank.p_mean.copy_(p_mean)
            bank.p_m2.zero_()                   # 没有二阶矩，置零即可
            bank.g_codes.copy_(g_codes)
            bank.counts.copy_(ckpt.get("counts", torch.ones(K)))

            print(f">>> Loaded PrototypeBank (from p_align) | K={K}, C={C}, H={H}, W={W}")
            return bank

        # ===== 原生格式 =====
        if "p_mean" in ckpt:
            K, C, H, W = ckpt["p_mean"].shape
            bank = PrototypeBank(
                K=K,
                g_dim=ckpt["g_dim"],
                C=C, H=H, W=W,
                momentum=ckpt.get("momentum", 0.99),
                eps=ckpt.get("eps", 1e-6),
                tau=ckpt.get("tau", 0.1),
            )
            bank.p_mean.copy_(ckpt["p_mean"])
            bank.p_m2.copy_(ckpt["p_m2"])
            bank.g_codes.copy_(ckpt["g_codes"])
            bank.counts.copy_(ckpt["counts"])
            return bank

        raise ValueError(f"Unrecognized PrototypeBank checkpoint format: {ckpt.keys()}")

    # 从 route(g) → retrieve_patchwise(feat_align)。prototype 变成“patch 级”的，不再是整图一个模板。
    @torch.no_grad()
    def retrieve_patchwise(self, feat_align, tau=0.07, use_cosine=True):
        """
        Patch-wise prototype retrieval.

        Args:
            feat_align: [B, C, H, W]  (e.g., 16x16)
        Returns:
            prior_align: [B, C, H, W]      patch-wise weighted prototype prior
            w_patch:     [B, K, H, W]      soft assignment weights
            dist_min:    [B, 1, H, W]      min distance to prototypes per patch (anomaly cue)
            sim:         [B, K, H, W]      similarity map (optional debug)
        """
        B, C, H, W = feat_align.shape
        K = self.K

        # --- flatten ---
        f = feat_align.view(B, C, H * W)                  # [B,C,HW]
        p = self.p_mean.view(K, C, H * W)                 # [K,C,HW]

        if use_cosine:
            f_n = F.normalize(f, dim=1)                   # [B,C,HW]
            p_n = F.normalize(p, dim=1)                   # [K,C,HW]
            # sim[b,k,hw] = dot(f[b,:,hw], p[k,:,hw])
            sim_bkhw = torch.einsum("bch,kch->bkh", f_n, p_n)  # [B,K,HW]
            sim = sim_bkhw.view(B, K, H, W)
            # weights across K at each (h,w)
            w_patch = torch.softmax(sim_bkhw / tau, dim=1).view(B, K, H, W)
            # distance proxy
            dist_bkhw = (1.0 - sim_bkhw).clamp(min=0.0)   # [B,K,HW]
        else:
            # L2 distance per position
            f2 = (f ** 2).sum(dim=1, keepdim=True)        # [B,1,HW]
            p2 = (p ** 2).sum(dim=1).unsqueeze(0)         # [1,K,HW]
            fp = torch.einsum("bch,kch->bkh", f, p)       # [B,K,HW]
            dist_bkhw = (f2 + p2 - 2 * fp).clamp(min=0.0).sqrt()
            w_patch = torch.softmax(-dist_bkhw / tau, dim=1).view(B, K, H, W)
            sim = None

        # --- build patch-wise prior ---
        # prior[b,c,hw] = sum_k w[b,k,hw] * p_mean[k,c,hw]
        w = w_patch.view(B, K, H * W)                     # [B,K,HW]
        prior_bchw = torch.einsum("bkh,kch->bch", w, p)    # [B,C,HW]
        prior_align = prior_bchw.view(B, C, H, W)          # [B,C,H,W]

        # --- min distance per patch (anomaly cue) ---
        dist_min = dist_bkhw.min(dim=1)[0].view(B, 1, H, W)  # [B,1,H,W]

        return prior_align, w_patch, dist_min, sim

    @staticmethod
    def entropy_loss(w_patch, eps=1e-8):
        """
        Encourage confident assignment (low entropy) per patch.
        w_patch: [B,K,H,W]
        """
        w = w_patch.clamp(min=eps)
        ent = -(w * w.log()).sum(dim=1, keepdim=False)     # [B,H,W]
        return ent.mean()

    def diversity_loss(self, w_patch):
        """
        Prevent collapse: encourage using multiple prototypes across batch.
        w_patch: [B,K,H,W]
        """
        B, K, H, W = w_patch.shape
        w_avg = w_patch.mean(dim=(0, 2, 3))                # [K]
        target = torch.full_like(w_avg, 1.0 / K)
        return ((w_avg - target) ** 2).mean()
