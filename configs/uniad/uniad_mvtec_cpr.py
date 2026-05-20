import importlib.util
from pathlib import Path
from argparse import Namespace

from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *
from configs.__base__.cfg_model_uniad import cfg_model_uniad


# _MODEL_FILE = Path(__file__).resolve().parents[2] / "cpr_1.a" / "model_cpr.py"
# _SPEC = importlib.util.spec_from_file_location("cpr_1a_model", _MODEL_FILE)
# _MODULE = importlib.util.module_from_spec(_SPEC)
# _SPEC.loader.exec_module(_MODULE)


class cfg(cfg_common, cfg_dataset_default, cfg_model_uniad):
    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_uniad.__init__(self)

        self.seed = 42
        self.size = 256
        self.epoch_full = 1000
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = self.epoch_full // 10
        self.batch_train = 16
        self.batch_test_per = 8
        self.lr = 2e-4
        self.weight_decay = 0.0001
        self.metrics = [
            "mAUROC_sp_max", "mAP_sp_max", "mF1_max_sp_max",
            "mAUPRO_px",
            "mAUROC_px", "mAP_px", "mF1_max_px",
            "mF1_px_0.2_0.8_0.1", "mAcc_px_0.2_0.8_0.1", "mIoU_px_0.2_0.8_0.1",
            "mIoU_max_px",
        ]

        self.data.type = "DefaultAD"
        self.data.root = "/home/featurize/work/data/mvtec"
        # self.data.root = "/home/featurize/work/data/visa"
        # self.data.root = "/home/featurize/work/data/mvtec_loco"
        # self.data.root = "/home/featurize/work/data/mpdd"
        # self.data.root = "/home/featurize/work/data/btad"
        self.data.meta = "meta.json"
        self.data.cls_names = []

        self.data.train_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
        ]

        self.model_backbone = Namespace()
        self.model_backbone.name = "timm_tf_efficientnet_b4"
        self.model_backbone.kwargs = dict(
            pretrained=True,
            checkpoint_path="",
            strict=False,
            hf=None,
            features_only=True,
            out_indices=[0, 1, 2, 3],
        )

        inplanes = [24, 32, 56, 160]
        self.model_decoder = dict(
            inplanes=inplanes,
            outplanes=[sum(inplanes)],
            instrides=[16],
            feature_size=[self.size // 16, self.size // 16],
            neighbor_size=[self.size // 32, self.size // 32],
        )

        self.model = Namespace()
        self.model.name = "uniad_cpr"
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path="",
            strict=True,
            model_backbone=self.model_backbone,
            model_decoder=self.model_decoder,
            bank_path='runs/prototypes/mvtec_bank.pth',
            # bank_path="runs/prototypes/visa_bank.pth",
            # bank_path="runs/prototypes/mvtec_loco_bank.pth",
            # bank_path="runs/prototypes/mpdd_bank.pth",
            # bank_path="runs/prototypes/btad_bank.pth",
            tau=0.07,
            use_cosine=True,
            disc_mid=128,
            # lambda_disc=0.20,
            lambda_disc=0.35,
            fuse_mode="mul",
            top_modes=1,
            correction_strength=0.25,
            gate_hidden_ratio=0.5,
            conf_power=1.0,
            conf_floor=0.05,
        )

        self.evaluator.kwargs = dict(
            metrics=self.metrics,
            pooling_ks=[16, 16],
            max_step_aupro=100,
        )

        self.optim.lr = self.lr
        self.optim.kwargs = dict(
            name="adamw",
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False,
        )

        self.trainer.name = "UniADTrainer"
        self.trainer.logdir_sub = ""
        # self.trainer.resume_dir = ""        #训练
        self.trainer.resume_dir = "UniADTrainer_configs_uniad_mvtec_cpr_20260324-132843"                #测试
        # self.model.kwargs["checkpoint_path"] = "runs/UniADTrainer_configs_uniad_mvtec_cpr_20260324-132843/net_700.pth"      #指定epoch
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name="cosine",
            lr_noise=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            lr_min=self.lr / 50,
            warmup_lr=self.lr / 10,
            warmup_iters=-1,
            cooldown_iters=0,
            warmup_epochs=5,
            cooldown_epochs=0,
            use_iters=True,
            patience_iters=0,
            patience_epochs=0,
            decay_iters=0,
            decay_epochs=int(self.epoch_full * 0.8),
            cycle_decay=0.1,
            decay_rate=0.1,
        )
        self.trainer.mixup_kwargs = None
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per
        self.trainer.test_label_leak = dict(enabled=False, ratio=0.0)
        # self.trainer.test_label_leak = dict(
        #     enabled=True,
        #     ratio=0.0,
        #     ratio_normal=0.0,
        #     ratio_anomaly=0.1,
        #     stratify_by_class=True,
        #     stratify_by_anomaly=True,
        #     use_pixel_mask=True,
        #     normal_mode="zero",
        #     anomaly_mode="mask_alpha_blend",
        #     anomaly_blend_alpha = 0.15
        # )

        self.trainer.data.num_workers_per_gpu = 12
        self.trainer.data.pin_memory = True
        self.trainer.data.persistent_workers = True
        self.trainer.data.drop_last = True

        self.loss.loss_terms = [
            dict(type="L2Loss", name="pixel", lam=1.0),
        ]

        self.logging.log_terms_train = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
            dict(name="data_t", fmt=":>5.3f"),
            dict(name="optim_t", fmt=":>5.3f"),
            dict(name="lr", fmt=":>7.6f"),
            dict(name="pixel", suffixes=[""], fmt=":>5.3f", add_name="avg"),
        ]
        self.logging.log_terms_test = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
            dict(name="pixel", suffixes=[""], fmt=":>5.3f", add_name="avg"),
        ]
