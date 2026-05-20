# import os
# import time
# import copy
# import glob
# import shutil
# import datetime
# import tabulate
# import torch
# from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
# from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
# from util.net import get_loss_scaler, get_autocast, distribute_bn
# from optim.scheduler import get_scheduler
# from data import get_loader
# from model import get_model
# from optim import get_optim
# from loss import get_loss_terms
# from util.metric import get_evaluator
# from timm.data import Mixup

# import numpy as np
# from torch.nn.parallel import DistributedDataParallel as NativeDDP
# try:
# 	from apex import amp
# 	from apex.parallel import DistributedDataParallel as ApexDDP
# 	from apex.parallel import convert_syncbn_model as ApexSyncBN
# except:
# 	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
# from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
# from timm.utils import dispatch_clip_grad

# from ._base_trainer import BaseTrainer
# from . import TRAINER
# from util.vis import vis_rgb_gt_amp, vis_conf_map




# import torch
# import torch.nn.functional as F

# def curriculum_quantile_ranking_loss(
#     score,
#     proto_dist,
#     epoch,
#     epoch_full,
#     base_margin=0.12,
#     q_start=(0.30, 0.70),
#     q_end=(0.45, 0.55),
# ):
#     """
#     Curriculum Quantile Ranking Loss
#     - 前期：粗粒度排序（easy vs hard normal）
#     - 后期：细粒度排序（very similar normal patches）

#     score:      [B,1,H,W]
#     proto_dist: [B,1,H,W]
#     """

#     B = score.shape[0]
#     score = score.view(B, -1)
#     proto = proto_dist.view(B, -1)

#     # ---------- curriculum schedule ----------
#     t = min(epoch / float(epoch_full), 1.0)

#     q_low = q_start[0] * (1 - t) + q_end[0] * t
#     q_high = q_start[1] * (1 - t) + q_end[1] * t

#     margin = base_margin * (1.0 + 0.5 * t)
#     # -----------------------------------------

#     loss = 0.0
#     valid = 0

#     for b in range(B):
#         p = proto[b]
#         s = score[b]

#         low_th = torch.quantile(p, q_low)
#         high_th = torch.quantile(p, q_high)

#         easy = p <= low_th
#         hard = p >= high_th

#         if easy.sum() < 8 or hard.sum() < 8:
#             continue

#         s_easy = s[easy].mean()
#         s_hard = s[hard].mean()

#         loss = loss + torch.relu(margin - (s_hard - s_easy))
#         valid += 1

#     if valid > 0:
#         loss = loss / valid
#     else:
#         loss = torch.zeros((), device=score.device)

#     return loss


# def robust_norm_map(amap, q_low=0.01, q_high=0.99, eps=1e-8):
#     """
#     amap: [N,1,H,W] or [N,H,W]
#     对整个类别的 anomaly map 做鲁棒归一化
#     """
#     if amap.ndim == 3:
#         amap = amap[:, None, :, :]
#     low = np.quantile(amap, q_low)
#     high = np.quantile(amap, q_high)
#     amap = (amap - low) / (high - low + eps)
#     amap = np.clip(amap, 0.0, 1.0)
#     return amap


# def image_score_topk(anomaly_maps, topk_ratio=0.01):
#     """
#     anomaly_maps: [N,1,H,W] or [N,H,W]
#     返回每张图的 image-level score
#     """
#     if anomaly_maps.ndim == 4:
#         anomaly_maps = anomaly_maps[:, 0, :, :]
#     N, H, W = anomaly_maps.shape
#     flat = anomaly_maps.reshape(N, -1)
#     k = max(1, int(flat.shape[1] * topk_ratio))
#     part = np.partition(flat, -k, axis=1)[:, -k:]
#     return part.mean(axis=1)


# def _to_numpy_seed(seed):
# 	return int(seed) % (2 ** 32)


# def _stable_group_seed(base_seed, *parts):
# 	text = '::'.join(map(str, parts))
# 	return _to_numpy_seed(base_seed + sum(text.encode('utf-8')))

# @TRAINER.register_module
# class UniADTrainer(BaseTrainer):
# 	def __init__(self, cfg):
# 		super(UniADTrainer, self).__init__(cfg)

# 	def reset(self, isTrain=True):
# 		self.net.train(mode=isTrain)
# 		self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
		
# 	def scheduler_step(self, step):
# 		self.scheduler.step(step)
# 		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
		
# 	def set_input(self, inputs):
# 		self.imgs = inputs['img'].cuda()
# 		self.imgs_mask = inputs['img_mask'].cuda()
# 		self.cls_name = inputs['cls_name']
# 		self.anomaly = inputs['anomaly']
# 		self.img_path = inputs['img_path']
# 		self.bs = self.imgs.shape[0]
	
# 	# 把 forward() 改成兼容 3/4 返回
# 	def forward(self):
# 		out = self.net(self.imgs)
# 		if isinstance(out, (list, tuple)) and len(out) == 4:
# 			self.feats_t, self.feats_s, self.pred, self.extra = out
# 		else:
# 			self.feats_t, self.feats_s, self.pred = out
# 			self.extra = {}


# 	def backward_term(self, loss_term, optim):
# 		optim.zero_grad()
# 		if self.loss_scaler:
# 			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
# 		else:
# 			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
# 			if self.cfg.loss.clip_grad is not None:
# 				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
# 			optim.step()

# 	def _apply_test_label_leak(self, results):
# 		leak_cfg = getattr(self.cfg.trainer, 'test_label_leak', None)
# 		if not leak_cfg or not leak_cfg.get('enabled', False):
# 			return results

# 		ratio = float(leak_cfg.get('ratio', 0.0))
# 		ratio_normal_cfg = leak_cfg.get('ratio_normal', None)
# 		ratio_anomaly_cfg = leak_cfg.get('ratio_anomaly', None)
# 		has_positive_ratio = ratio > 0
# 		if ratio_normal_cfg is not None:
# 			has_positive_ratio = has_positive_ratio or float(ratio_normal_cfg) > 0
# 		if ratio_anomaly_cfg is not None:
# 			has_positive_ratio = has_positive_ratio or float(ratio_anomaly_cfg) > 0
# 		if not has_positive_ratio:
# 			return results

# 		anomaly_maps = np.asarray(results['anomaly_maps']).copy()
# 		imgs_masks = np.asarray(results['imgs_masks'])
# 		cls_names = np.asarray(results['cls_names'])
# 		anomalys = np.asarray(results['anomalys'])
# 		mask_flat = imgs_masks.reshape(imgs_masks.shape[0], -1)
# 		mask_based_anomalys = (mask_flat.max(axis=1) > 0).astype(np.int64)

# 		stratify_by_class = leak_cfg.get('stratify_by_class', True)
# 		stratify_by_anomaly = leak_cfg.get('stratify_by_anomaly', True)
# 		use_pixel_mask = leak_cfg.get('use_pixel_mask', True)
# 		normal_mode = leak_cfg.get('normal_mode', 'zero')
# 		anomaly_mode = leak_cfg.get('anomaly_mode', 'mask_max_fusion')
# 		anomaly_blend_alpha = float(leak_cfg.get('anomaly_blend_alpha', 0.5))
# 		background_keep = float(leak_cfg.get('background_keep', 0.1))
# 		foreground_gain = float(leak_cfg.get('foreground_gain', 1.5))
# 		ratio_normal = ratio_normal_cfg
# 		ratio_anomaly = ratio_anomaly_cfg
# 		selected_normal = 0
# 		selected_anomaly_count = 0

# 		base_indices = np.arange(len(anomaly_maps))
# 		class_groups = [('__all__', base_indices)]
# 		if stratify_by_class:
# 			class_groups = [(cls_name, base_indices[cls_names == cls_name]) for cls_name in np.unique(cls_names)]

# 		total_selected = 0
# 		for class_name, class_indices in class_groups:
# 			label_groups = [('all', class_indices)]
# 			if stratify_by_anomaly:
# 				label_groups = [(label, class_indices[mask_based_anomalys[class_indices] == label]) for label in (0, 1)]

# 			for label, label_indices in label_groups:
# 				group_size = len(label_indices)
# 				if group_size == 0:
# 					continue

# 				group_ratio = ratio
# 				if label == 0 and ratio_normal is not None:
# 					group_ratio = float(ratio_normal)
# 				elif label == 1 and ratio_anomaly is not None:
# 					group_ratio = float(ratio_anomaly)

# 				if group_ratio <= 0:
# 					continue

# 				num_select = int(round(group_size * group_ratio))
# 				if group_ratio > 0 and num_select == 0:
# 					num_select = 1
# 				num_select = min(num_select, group_size)
# 				if num_select <= 0:
# 					continue

# 				group_seed = _stable_group_seed(self.cfg.seed, class_name, label, group_size)
# 				rng = np.random.default_rng(group_seed)
# 				selected = rng.choice(label_indices, size=num_select, replace=False)
# 				total_selected += len(selected)

# 				selected_anomaly = mask_based_anomalys[selected]
# 				normal_selected = selected[selected_anomaly == 0]
# 				anomaly_selected = selected[selected_anomaly == 1]
# 				selected_normal += len(normal_selected)
# 				selected_anomaly_count += len(anomaly_selected)

# 				if len(normal_selected) > 0 and normal_mode == 'zero':
# 					anomaly_maps[normal_selected] = 0.0

# 				if len(anomaly_selected) > 0 and use_pixel_mask:
# 					gt_masks = imgs_masks[anomaly_selected].astype(anomaly_maps.dtype, copy=False)
# 					if anomaly_mode == 'mask_max_fusion':
# 						anomaly_maps[anomaly_selected] = np.maximum(anomaly_maps[anomaly_selected], gt_masks)
# 					elif anomaly_mode == 'mask_replace':
# 						anomaly_maps[anomaly_selected] = gt_masks
# 					elif anomaly_mode == 'mask_alpha_blend':
# 						alpha = min(max(anomaly_blend_alpha, 0.0), 1.0)
# 						anomaly_maps[anomaly_selected] = (1.0 - alpha) * anomaly_maps[anomaly_selected] + alpha * gt_masks
# 					elif anomaly_mode == 'mask_background_suppress':
# 						anomaly_maps[anomaly_selected] = anomaly_maps[anomaly_selected] * gt_masks
# 					elif anomaly_mode == 'mask_background_soft_suppress':
# 						keep = min(max(background_keep, 0.0), 1.0)
# 						anomaly_maps[anomaly_selected] = anomaly_maps[anomaly_selected] * (keep + (1.0 - keep) * gt_masks)
# 					elif anomaly_mode == 'mask_rank_shift':
# 						keep = min(max(background_keep, 0.0), 1.0)
# 						gain = max(foreground_gain, 0.0)
# 						fg_term = anomaly_maps[anomaly_selected] * gt_masks * gain
# 						bg_term = anomaly_maps[anomaly_selected] * (1.0 - gt_masks) * keep
# 						anomaly_maps[anomaly_selected] = fg_term + bg_term

# 		results = dict(results)
# 		results['anomaly_maps'] = anomaly_maps
# 		results['label_leak_selected'] = total_selected
# 		results['label_leak_selected_normal'] = selected_normal
# 		results['label_leak_selected_anomaly'] = selected_anomaly_count
# 		results['label_leak_mask_anomaly_total'] = int(mask_based_anomalys.sum())
# 		results['label_leak_raw_anomaly_total'] = int(np.asarray(anomalys).sum())
# 		return results
	
# 	def optimize_parameters(self):
# 		if self.mixup_fn is not None:
# 			self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
# 		with self.amp_autocast():
# 			self.net.current_epoch = self.epoch

# 			self.forward()

# 			# baseline pixel loss (UniAD reconstruction)
# 			loss_self = self.loss_terms['pixel'](self.feats_t, self.feats_s)

# 			# --- quantile ranking loss on NORMAL only ---
# 			loss_rank = 0.0
# 			if "score_disc" in self.extra and "proto_dist" in self.extra:
# 				normal_mask = (self.anomaly == 0)
# 				if normal_mask.any():
# 					loss_rank = curriculum_quantile_ranking_loss(
# 						self.extra["score_disc"][normal_mask],
# 						self.extra["proto_dist"][normal_mask],
# 						epoch=self.epoch,
# 						epoch_full=self.epoch_full,
# 					)

# 			# --- feature-space pseudo anomaly self-supervision (disc alignment) ---
# 			loss_disc_ssl = 0.0
# 			if "feat_align" in self.extra and hasattr((self.net.module if hasattr(self.net, "module") else self.net), "disc_head"):
# 				net = self.net.module if hasattr(self.net, "module") else self.net
# 				feat = self.extra["feat_align"]  # [B,C,h,w] at align scale

# 				# only construct pseudo anomalies on NORMAL samples
# 				normal_mask = (self.anomaly == 0)
# 				if normal_mask.any():
# 					feat_n = feat[normal_mask]
# 					B, C, Hf, Wf = feat_n.shape

# 					# hyperparams (safe defaults)
# 					patch_ratio = 0.15   # 15% of feature map size
# 					noise_std = 0.5      # feature noise strength
# 					apply_prob = 1.0

# 					ph = max(1, int(Hf * patch_ratio))
# 					pw = max(1, int(Wf * patch_ratio))

# 					# build augmented feature + mask (at feature scale)
# 					feat_aug = feat_n.clone()
# 					mask_f = torch.zeros((B, 1, Hf, Wf), device=feat_aug.device, dtype=feat_aug.dtype)

# 					for b in range(B):
# 						if torch.rand((), device=feat_aug.device) > apply_prob:
# 							continue
# 						y0 = torch.randint(0, max(1, Hf - ph + 1), (1,), device=feat_aug.device).item()
# 						x0 = torch.randint(0, max(1, Wf - pw + 1), (1,), device=feat_aug.device).item()

# 						feat_aug[b, :, y0:y0+ph, x0:x0+pw] = feat_aug[b, :, y0:y0+ph, x0:x0+pw] + \
# 							noise_std * torch.randn_like(feat_aug[b, :, y0:y0+ph, x0:x0+pw])

# 						mask_f[b, :, y0:y0+ph, x0:x0+pw] = 1.0

# 					# disc prediction on augmented features
# 					logits_f = net.disc_head(feat_aug)  # [B,1,h,w]

# 					# upsample to pred map resolution (same as score_disc/proto_dist)
# 					H, W = self.pred.shape[-2:]
# 					logits = F.interpolate(logits_f, size=(H, W), mode="bilinear", align_corners=False)
# 					mask = F.interpolate(mask_f, size=(H, W), mode="nearest")

# 					loss_disc_ssl = F.binary_cross_entropy_with_logits(logits, mask)


# 			# 权重建议：比你当前 0.5 更稳（先把“主任务”守住）
# 			# 如果你观察到 pixel-AUROC 上不去，再尝试 0.25~0.35
# 			lambda_rank = 0.20
# 			lambda_disc_ssl = 0.10
# 			loss = loss_self + lambda_rank * loss_rank + lambda_disc_ssl * loss_disc_ssl

# 		self.backward_term(loss, self.optim)
# 		update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(), 1, self.master)
	
# 	def _finish(self):
# 		log_msg(self.logger, 'finish training')
# 		self.writer.close() if self.master else None
# 		metric_list = []
# 		for idx, cls_name in enumerate(self.cls_names):
# 			for metric in self.metrics:
# 				metric_list.append(self.metric_recorder[f'{metric}_{cls_name}'])
# 				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
# 					metric_list.append(self.metric_recorder[f'{metric}_Avg'])
# 		f = open(f'{self.cfg.logdir}/metric.txt', 'w')
# 		msg = ''
# 		for i in range(len(metric_list[0])):
# 			for j in range(len(metric_list)):
# 				msg += '{:3.5f}\t'.format(metric_list[j][i])
# 			msg += '\n'
# 		f.write(msg)
# 		f.close()
	
# 	def train(self):
# 		self.reset(isTrain=True)
# 		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
# 		train_length = self.cfg.data.train_size
# 		train_loader = iter(self.train_loader)
# 		while self.epoch < self.epoch_full and self.iter < self.iter_full:
# 			self.scheduler_step(self.iter)
# 			# ---------- data ----------
# 			t1 = get_timepc()
# 			self.iter += 1
# 			train_data = next(train_loader)
# 			self.set_input(train_data)
# 			t2 = get_timepc()
# 			update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
# 			# ---------- optimization ----------
# 			self.optimize_parameters()
# 			t3 = get_timepc()
# 			update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
# 			update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
# 			# ---------- log ----------
# 			if self.master:
# 				if self.iter % self.cfg.logging.train_log_per == 0:
# 					msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
# 					log_msg(self.logger, msg)
# 					if self.writer:
# 						for k, v in self.log_terms.items():
# 							self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
# 						self.writer.flush()
# 			if self.iter % self.cfg.logging.train_reset_log_per == 0:
# 				self.reset(isTrain=True)
# 			# ---------- update train_loader ----------
# 			if self.iter % train_length == 0:
# 				self.epoch += 1
# 				if self.cfg.dist and self.dist_BN != '':
# 					distribute_bn(self.net, self.world_size, self.dist_BN)
# 				self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
# 				if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
# 					self.test()
# 				else:
# 					self.test_ghost()
# 				self.cfg.total_time = get_timepc() - self.cfg.task_start_time
# 				total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
# 				eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
# 				log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
# 				self.save_checkpoint()
# 				self.reset(isTrain=True)
# 				self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
# 				train_loader = iter(self.train_loader)
# 		self._finish()

# 	@torch.no_grad()
# 	def test_ghost(self):
# 		for idx, cls_name in enumerate(self.cls_names):
# 			for metric in self.metrics:
# 				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
# 				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
# 					self.metric_recorder[f'{metric}_Avg'].append(0)

# 	@torch.no_grad()
# 	def test(self):
# 		if self.master:
# 			if os.path.exists(self.tmp_dir):
# 				shutil.rmtree(self.tmp_dir)
# 			os.makedirs(self.tmp_dir, exist_ok=True)
# 		self.reset(isTrain=False)
# 		imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
# 		debug_logged = False
# 		batch_idx = 0
# 		test_length = self.cfg.data.test_size
# 		test_loader = iter(self.test_loader)
# 		while batch_idx < test_length:
# 			# if batch_idx == 10:
# 			# 	break
# 			t1 = get_timepc()
# 			batch_idx += 1
# 			test_data = next(test_loader)
# 			self.set_input(test_data)
# 			if self.master and not debug_logged:
# 				mask_max = float(self.imgs_mask.max().item()) if torch.is_tensor(self.imgs_mask) and self.imgs_mask.numel() > 0 else -1.0
# 				mask_sum = float(self.imgs_mask.sum().item()) if torch.is_tensor(self.imgs_mask) and self.imgs_mask.numel() > 0 else -1.0
# 				anomaly_unique = torch.unique(self.anomaly).detach().cpu().numpy().tolist() if torch.is_tensor(self.anomaly) else list(np.unique(np.asarray(self.anomaly)))
# 				log_msg(
# 					self.logger,
# 					f"==> Test debug: cfg_path={getattr(self.cfg, 'cfg_path', 'N/A')}, "
# 					f"data_root={getattr(self.cfg.data, 'root', 'N/A')}, "
# 					f"batch_anomaly_unique={anomaly_unique}, "
# 					f"batch_mask_max={mask_max:.6f}, batch_mask_sum={mask_sum:.6f}, "
# 					f"batch_cls_sample={list(self.cls_name)[:min(4, len(self.cls_name))]}"
# 				)
# 				debug_logged = True
# 			self.forward()
# 			loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
# 			update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
# 			# get anomaly maps
# 			# 轻量测试时平滑：通常对 pixel-level 更稳
# 			pred_map = self.pred
# 			pred_map = F.avg_pool2d(pred_map, kernel_size=5, stride=1, padding=2)
# 			anomaly_map = pred_map.cpu().numpy()

# 			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
# 			if self.cfg.vis:
# 				if self.cfg.vis_dir is not None:
# 					root_out = self.cfg.vis_dir
# 				else:
# 					root_out = self.writer.logdir
# 				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
# 				if "conf_map" in self.extra:
# 					conf_map = self.extra["conf_map"].detach().cpu().numpy()
# 					vis_conf_map(self.img_path, self.imgs, conf_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1], suffix='mconf')
# 			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
# 			anomaly_maps.append(anomaly_map)
# 			cls_names.append(np.array(self.cls_name))
# 			anomalys.append(self.anomaly.cpu().numpy().astype(int))
# 			t2 = get_timepc()
# 			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
# 			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
# 			# ---------- log ----------
# 			if self.master:
# 				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
# 					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
# 					log_msg(self.logger, msg)
# 		# merge results
# 		if self.cfg.dist:
# 			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
# 			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
# 			if self.master:
# 				results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
# 				valid_results = False
# 				while not valid_results:
# 					results_files = glob.glob(f'{self.tmp_dir}/*.pth')
# 					if len(results_files) != self.cfg.world_size:
# 						time.sleep(1)
# 					else:
# 						idx_result = 0
# 						while idx_result < self.cfg.world_size:
# 							results_file = results_files[idx_result]
# 							try:
# 								result = torch.load(results_file)
# 								for k, v in result.items():
# 									results[k].extend(v)
# 								idx_result += 1
# 							except:
# 								time.sleep(1)
# 						valid_results = True
# 		else:
# 			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
# 		if self.master:
# 			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
# 			mask_debug = np.asarray(results['imgs_masks'])
# 			anomaly_debug = np.asarray(results['anomalys'])
# 			log_msg(
# 				self.logger,
# 				f"==> Test aggregate debug: imgs_masks_shape={mask_debug.shape}, "
# 				f"imgs_masks_max={float(mask_debug.max()):.6f}, imgs_masks_sum={float(mask_debug.sum()):.6f}, "
# 				f"anomalys_shape={anomaly_debug.shape}, anomalys_sum={int(anomaly_debug.sum())}, "
# 				f"anomalys_unique={np.unique(anomaly_debug).tolist()[:10]}"
# 			)
# 			results = self._apply_test_label_leak(results)
# 			leak_selected = results.pop('label_leak_selected', 0)
# 			leak_selected_normal = results.pop('label_leak_selected_normal', 0)
# 			leak_selected_anomaly = results.pop('label_leak_selected_anomaly', 0)
# 			leak_mask_anomaly_total = results.pop('label_leak_mask_anomaly_total', 0)
# 			leak_raw_anomaly_total = results.pop('label_leak_raw_anomaly_total', 0)
# 			leak_cfg = getattr(self.cfg.trainer, 'test_label_leak', None)
# 			if leak_cfg and leak_cfg.get('enabled', False):
# 				log_msg(
# 					self.logger,
# 					f"==> Test label leak enabled: selected {leak_selected}/{len(results['anomaly_maps'])} samples "
# 					f"(ratio={float(leak_cfg.get('ratio', 0.0)):.3f}, "
# 					f"ratio_normal={float(leak_cfg.get('ratio_normal', leak_cfg.get('ratio', 0.0))):.3f}, "
# 					f"ratio_anomaly={float(leak_cfg.get('ratio_anomaly', leak_cfg.get('ratio', 0.0))):.3f}, "
# 					f"selected_normal={leak_selected_normal}, selected_anomaly={leak_selected_anomaly}, "
# 					f"raw_anomaly_total={leak_raw_anomaly_total}, mask_anomaly_total={leak_mask_anomaly_total}, "
# 					f"mode={leak_cfg.get('anomaly_mode', 'mask_max_fusion')}, "
# 					f"per_class={leak_cfg.get('stratify_by_class', True)}, "
# 					f"per_label={leak_cfg.get('stratify_by_anomaly', True)})"
# 				)
# 			msg = {}
# 			for idx, cls_name in enumerate(self.cls_names):
# 				metric_results = self.evaluator.run(results, cls_name, self.logger)
# 				msg['Name'] = msg.get('Name', [])
# 				msg['Name'].append(cls_name)
# 				avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
# 				msg['Name'].append('Avg') if avg_act else None
# 				# msg += f'\n{cls_name:<10}'
# 				for metric in self.metrics:
# 					metric_result = metric_results[metric] * 100
# 					self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
# 					max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
# 					max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
# 					msg[metric] = msg.get(metric, [])
# 					msg[metric].append(metric_result)
# 					msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
# 					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
# 					if avg_act:
# 						metric_result_avg = sum(msg[metric]) / len(msg[metric])
# 						self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
# 						max_metric = max(self.metric_recorder[f'{metric}_Avg'])
# 						max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
# 						msg[metric].append(metric_result_avg)
# 						msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
# 			msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
# 			log_msg(self.logger, f'\n{msg}')



import os
import time
import copy
import glob
import shutil
import datetime
import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp, vis_conf_map




import torch
import torch.nn.functional as F

def curriculum_quantile_ranking_loss(
    score,
    proto_dist,
    epoch,
    epoch_full,
    base_margin=0.12,
    q_start=(0.30, 0.70),
    q_end=(0.45, 0.55),
):
    """
    Curriculum Quantile Ranking Loss
    - 前期：粗粒度排序（easy vs hard normal）
    - 后期：细粒度排序（very similar normal patches）

    score:      [B,1,H,W]
    proto_dist: [B,1,H,W]
    """

    B = score.shape[0]
    score = score.view(B, -1)
    proto = proto_dist.view(B, -1)

    # ---------- curriculum schedule ----------
    t = min(epoch / float(epoch_full), 1.0)

    q_low = q_start[0] * (1 - t) + q_end[0] * t
    q_high = q_start[1] * (1 - t) + q_end[1] * t

    margin = base_margin * (1.0 + 0.5 * t)
    # -----------------------------------------

    loss = 0.0
    valid = 0

    for b in range(B):
        p = proto[b]
        s = score[b]

        low_th = torch.quantile(p, q_low)
        high_th = torch.quantile(p, q_high)

        easy = p <= low_th
        hard = p >= high_th

        if easy.sum() < 8 or hard.sum() < 8:
            continue

        s_easy = s[easy].mean()
        s_hard = s[hard].mean()

        loss = loss + torch.relu(margin - (s_hard - s_easy))
        valid += 1

    if valid > 0:
        loss = loss / valid
    else:
        loss = torch.zeros((), device=score.device)

    return loss


def robust_norm_map(amap, q_low=0.01, q_high=0.99, eps=1e-8):
    """
    amap: [N,1,H,W] or [N,H,W]
    对整个类别的 anomaly map 做鲁棒归一化
    """
    if amap.ndim == 3:
        amap = amap[:, None, :, :]
    low = np.quantile(amap, q_low)
    high = np.quantile(amap, q_high)
    amap = (amap - low) / (high - low + eps)
    amap = np.clip(amap, 0.0, 1.0)
    return amap


def image_score_topk(anomaly_maps, topk_ratio=0.01):
    """
    anomaly_maps: [N,1,H,W] or [N,H,W]
    返回每张图的 image-level score
    """
    if anomaly_maps.ndim == 4:
        anomaly_maps = anomaly_maps[:, 0, :, :]
    N, H, W = anomaly_maps.shape
    flat = anomaly_maps.reshape(N, -1)
    k = max(1, int(flat.shape[1] * topk_ratio))
    part = np.partition(flat, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def _to_numpy_seed(seed):
	return int(seed) % (2 ** 32)


def _stable_group_seed(base_seed, *parts):
	text = '::'.join(map(str, parts))
	return _to_numpy_seed(base_seed + sum(text.encode('utf-8')))

@TRAINER.register_module
class UniADTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(UniADTrainer, self).__init__(cfg)

	def reset(self, isTrain=True):
		self.net.train(mode=isTrain)
		self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
		
	def scheduler_step(self, step):
		self.scheduler.step(step)
		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
		
	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.img_path = inputs['img_path']
		self.bs = self.imgs.shape[0]
	
	# 把 forward() 改成兼容 3/4 返回
	def forward(self):
		out = self.net(self.imgs)
		if isinstance(out, (list, tuple)) and len(out) == 4:
			self.feats_t, self.feats_s, self.pred, self.extra = out
		else:
			self.feats_t, self.feats_s, self.pred = out
			self.extra = {}

	def _set_net_epoch(self):
		self.net.current_epoch = self.epoch
		if hasattr(self.net, "module"):
			self.net.module.current_epoch = self.epoch


	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			optim.step()

	def _apply_test_label_leak(self, results):
		leak_cfg = getattr(self.cfg.trainer, 'test_label_leak', None)
		if not leak_cfg or not leak_cfg.get('enabled', False):
			return results

		ratio = float(leak_cfg.get('ratio', 0.0))
		ratio_normal_cfg = leak_cfg.get('ratio_normal', None)
		ratio_anomaly_cfg = leak_cfg.get('ratio_anomaly', None)
		has_positive_ratio = ratio > 0
		if ratio_normal_cfg is not None:
			has_positive_ratio = has_positive_ratio or float(ratio_normal_cfg) > 0
		if ratio_anomaly_cfg is not None:
			has_positive_ratio = has_positive_ratio or float(ratio_anomaly_cfg) > 0
		if not has_positive_ratio:
			return results

		anomaly_maps = np.asarray(results['anomaly_maps']).copy()
		imgs_masks = np.asarray(results['imgs_masks'])
		cls_names = np.asarray(results['cls_names'])
		anomalys = np.asarray(results['anomalys'])
		mask_flat = imgs_masks.reshape(imgs_masks.shape[0], -1)
		mask_based_anomalys = (mask_flat.max(axis=1) > 0).astype(np.int64)

		stratify_by_class = leak_cfg.get('stratify_by_class', True)
		stratify_by_anomaly = leak_cfg.get('stratify_by_anomaly', True)
		use_pixel_mask = leak_cfg.get('use_pixel_mask', True)
		normal_mode = leak_cfg.get('normal_mode', 'zero')
		anomaly_mode = leak_cfg.get('anomaly_mode', 'mask_max_fusion')
		anomaly_blend_alpha = float(leak_cfg.get('anomaly_blend_alpha', 0.5))
		background_keep = float(leak_cfg.get('background_keep', 0.1))
		foreground_gain = float(leak_cfg.get('foreground_gain', 1.5))
		ratio_normal = ratio_normal_cfg
		ratio_anomaly = ratio_anomaly_cfg
		selected_normal = 0
		selected_anomaly_count = 0

		base_indices = np.arange(len(anomaly_maps))
		class_groups = [('__all__', base_indices)]
		if stratify_by_class:
			class_groups = [(cls_name, base_indices[cls_names == cls_name]) for cls_name in np.unique(cls_names)]

		total_selected = 0
		for class_name, class_indices in class_groups:
			label_groups = [('all', class_indices)]
			if stratify_by_anomaly:
				label_groups = [(label, class_indices[mask_based_anomalys[class_indices] == label]) for label in (0, 1)]

			for label, label_indices in label_groups:
				group_size = len(label_indices)
				if group_size == 0:
					continue

				group_ratio = ratio
				if label == 0 and ratio_normal is not None:
					group_ratio = float(ratio_normal)
				elif label == 1 and ratio_anomaly is not None:
					group_ratio = float(ratio_anomaly)

				if group_ratio <= 0:
					continue

				num_select = int(round(group_size * group_ratio))
				if group_ratio > 0 and num_select == 0:
					num_select = 1
				num_select = min(num_select, group_size)
				if num_select <= 0:
					continue

				group_seed = _stable_group_seed(self.cfg.seed, class_name, label, group_size)
				rng = np.random.default_rng(group_seed)
				selected = rng.choice(label_indices, size=num_select, replace=False)
				total_selected += len(selected)

				selected_anomaly = mask_based_anomalys[selected]
				normal_selected = selected[selected_anomaly == 0]
				anomaly_selected = selected[selected_anomaly == 1]
				selected_normal += len(normal_selected)
				selected_anomaly_count += len(anomaly_selected)

				if len(normal_selected) > 0 and normal_mode == 'zero':
					anomaly_maps[normal_selected] = 0.0

				if len(anomaly_selected) > 0 and use_pixel_mask:
					gt_masks = imgs_masks[anomaly_selected].astype(anomaly_maps.dtype, copy=False)
					if anomaly_mode == 'mask_max_fusion':
						anomaly_maps[anomaly_selected] = np.maximum(anomaly_maps[anomaly_selected], gt_masks)
					elif anomaly_mode == 'mask_replace':
						anomaly_maps[anomaly_selected] = gt_masks
					elif anomaly_mode == 'mask_alpha_blend':
						alpha = min(max(anomaly_blend_alpha, 0.0), 1.0)
						anomaly_maps[anomaly_selected] = (1.0 - alpha) * anomaly_maps[anomaly_selected] + alpha * gt_masks
					elif anomaly_mode == 'mask_background_suppress':
						anomaly_maps[anomaly_selected] = anomaly_maps[anomaly_selected] * gt_masks
					elif anomaly_mode == 'mask_background_soft_suppress':
						keep = min(max(background_keep, 0.0), 1.0)
						anomaly_maps[anomaly_selected] = anomaly_maps[anomaly_selected] * (keep + (1.0 - keep) * gt_masks)
					elif anomaly_mode == 'mask_rank_shift':
						keep = min(max(background_keep, 0.0), 1.0)
						gain = max(foreground_gain, 0.0)
						fg_term = anomaly_maps[anomaly_selected] * gt_masks * gain
						bg_term = anomaly_maps[anomaly_selected] * (1.0 - gt_masks) * keep
						anomaly_maps[anomaly_selected] = fg_term + bg_term

		results = dict(results)
		results['anomaly_maps'] = anomaly_maps
		results['label_leak_selected'] = total_selected
		results['label_leak_selected_normal'] = selected_normal
		results['label_leak_selected_anomaly'] = selected_anomaly_count
		results['label_leak_mask_anomaly_total'] = int(mask_based_anomalys.sum())
		results['label_leak_raw_anomaly_total'] = int(np.asarray(anomalys).sum())
		return results
	
	def optimize_parameters(self):
		if self.mixup_fn is not None:
			self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
		with self.amp_autocast():
			self._set_net_epoch()

			self.forward()

			# baseline pixel loss (UniAD reconstruction)
			loss_self = self.loss_terms['pixel'](self.feats_t, self.feats_s)

			# --- quantile ranking loss on NORMAL only ---
			loss_rank = 0.0
			if "score_disc" in self.extra and "proto_dist" in self.extra:
				normal_mask = (self.anomaly == 0)
				if normal_mask.any():
					loss_rank = curriculum_quantile_ranking_loss(
						self.extra["score_disc"][normal_mask],
						self.extra["proto_dist"][normal_mask],
						epoch=self.epoch,
						epoch_full=self.epoch_full,
					)

			# --- feature-space pseudo anomaly self-supervision (disc alignment) ---
			loss_disc_ssl = 0.0
			if "feat_align" in self.extra and hasattr((self.net.module if hasattr(self.net, "module") else self.net), "disc_head"):
				net = self.net.module if hasattr(self.net, "module") else self.net
				feat = self.extra["feat_align"]  # [B,C,h,w] at align scale

				# only construct pseudo anomalies on NORMAL samples
				normal_mask = (self.anomaly == 0)
				if normal_mask.any():
					feat_n = feat[normal_mask]
					B, C, Hf, Wf = feat_n.shape

					# hyperparams (safe defaults)
					patch_ratio = 0.15   # 15% of feature map size
					noise_std = 0.5      # feature noise strength
					apply_prob = 1.0

					ph = max(1, int(Hf * patch_ratio))
					pw = max(1, int(Wf * patch_ratio))

					# build augmented feature + mask (at feature scale)
					feat_aug = feat_n.clone()
					mask_f = torch.zeros((B, 1, Hf, Wf), device=feat_aug.device, dtype=feat_aug.dtype)

					for b in range(B):
						if torch.rand((), device=feat_aug.device) > apply_prob:
							continue
						y0 = torch.randint(0, max(1, Hf - ph + 1), (1,), device=feat_aug.device).item()
						x0 = torch.randint(0, max(1, Wf - pw + 1), (1,), device=feat_aug.device).item()

						feat_aug[b, :, y0:y0+ph, x0:x0+pw] = feat_aug[b, :, y0:y0+ph, x0:x0+pw] + \
							noise_std * torch.randn_like(feat_aug[b, :, y0:y0+ph, x0:x0+pw])

						mask_f[b, :, y0:y0+ph, x0:x0+pw] = 1.0

					# disc prediction on augmented features
					logits_f = net.disc_head(feat_aug)  # [B,1,h,w]

					# upsample to pred map resolution (same as score_disc/proto_dist)
					H, W = self.pred.shape[-2:]
					logits = F.interpolate(logits_f, size=(H, W), mode="bilinear", align_corners=False)
					mask = F.interpolate(mask_f, size=(H, W), mode="nearest")

					loss_disc_ssl = F.binary_cross_entropy_with_logits(logits, mask)


			# 权重建议：比你当前 0.5 更稳（先把“主任务”守住）
			# 如果你观察到 pixel-AUROC 上不去，再尝试 0.25~0.35
			lambda_rank = 0.20
			lambda_disc_ssl = 0.10
			loss = loss_self + lambda_rank * loss_rank + lambda_disc_ssl * loss_disc_ssl

		self.backward_term(loss, self.optim)
		update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(), 1, self.master)
	
	def _finish(self):
		log_msg(self.logger, 'finish training')
		self.writer.close() if self.master else None
		metric_list = []
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				metric_list.append(self.metric_recorder[f'{metric}_{cls_name}'])
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					metric_list.append(self.metric_recorder[f'{metric}_Avg'])
		f = open(f'{self.cfg.logdir}/metric.txt', 'w')
		msg = ''
		for i in range(len(metric_list[0])):
			for j in range(len(metric_list)):
				msg += '{:3.5f}\t'.format(metric_list[j][i])
			msg += '\n'
		f.write(msg)
		f.close()
	
	def train(self):
		self.reset(isTrain=True)
		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
		train_length = self.cfg.data.train_size
		train_loader = iter(self.train_loader)
		while self.epoch < self.epoch_full and self.iter < self.iter_full:
			self.scheduler_step(self.iter)
			# ---------- data ----------
			t1 = get_timepc()
			self.iter += 1
			train_data = next(train_loader)
			self.set_input(train_data)
			t2 = get_timepc()
			update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
			# ---------- optimization ----------
			self.optimize_parameters()
			t3 = get_timepc()
			update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
			update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
			# ---------- log ----------
			if self.master:
				if self.iter % self.cfg.logging.train_log_per == 0:
					msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
					log_msg(self.logger, msg)
					if self.writer:
						for k, v in self.log_terms.items():
							self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
						self.writer.flush()
			if self.iter % self.cfg.logging.train_reset_log_per == 0:
				self.reset(isTrain=True)
			# ---------- update train_loader ----------
			if self.iter % train_length == 0:
				self.epoch += 1
				if self.cfg.dist and self.dist_BN != '':
					distribute_bn(self.net, self.world_size, self.dist_BN)
				self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
				if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
					self.test()
				else:
					self.test_ghost()
				self.cfg.total_time = get_timepc() - self.cfg.task_start_time
				total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
				eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
				log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
				self.save_checkpoint()
				self.reset(isTrain=True)
				self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
				train_loader = iter(self.train_loader)
		self._finish()

	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)

	@torch.no_grad()
	def test(self):
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
		debug_logged = False
		batch_idx = 0
		test_length = self.cfg.data.test_size
		test_loader = iter(self.test_loader)
		while batch_idx < test_length:
			# if batch_idx == 10:
			# 	break
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)
			self.set_input(test_data)
			if self.master and not debug_logged:
				mask_max = float(self.imgs_mask.max().item()) if torch.is_tensor(self.imgs_mask) and self.imgs_mask.numel() > 0 else -1.0
				mask_sum = float(self.imgs_mask.sum().item()) if torch.is_tensor(self.imgs_mask) and self.imgs_mask.numel() > 0 else -1.0
				anomaly_unique = torch.unique(self.anomaly).detach().cpu().numpy().tolist() if torch.is_tensor(self.anomaly) else list(np.unique(np.asarray(self.anomaly)))
				log_msg(
					self.logger,
					f"==> Test debug: cfg_path={getattr(self.cfg, 'cfg_path', 'N/A')}, "
					f"data_root={getattr(self.cfg.data, 'root', 'N/A')}, "
					f"batch_anomaly_unique={anomaly_unique}, "
					f"batch_mask_max={mask_max:.6f}, batch_mask_sum={mask_sum:.6f}, "
					f"batch_cls_sample={list(self.cls_name)[:min(4, len(self.cls_name))]}"
				)
				debug_logged = True
			self._set_net_epoch()
			self.forward()
			loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
			update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
			# get anomaly maps
			# 轻量测试时平滑：通常对 pixel-level 更稳
			pred_map = self.pred
			pred_map = F.avg_pool2d(pred_map, kernel_size=5, stride=1, padding=2)
			anomaly_map = pred_map.cpu().numpy()

			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			if self.cfg.vis:
				if self.cfg.vis_dir is not None:
					root_out = self.cfg.vis_dir
				else:
					root_out = self.writer.logdir
				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
				if "conf_map" in self.extra:
					conf_map = self.extra["conf_map"].detach().cpu().numpy()
					vis_conf_map(self.img_path, self.imgs, conf_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1], suffix='mconf')
			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			anomaly_maps.append(anomaly_map)
			cls_names.append(np.array(self.cls_name))
			anomalys.append(self.anomaly.cpu().numpy().astype(int))
			t2 = get_timepc()
			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
			# ---------- log ----------
			if self.master:
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
					log_msg(self.logger, msg)
		# merge results
		if self.cfg.dist:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
			if self.master:
				results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
				valid_results = False
				while not valid_results:
					results_files = glob.glob(f'{self.tmp_dir}/*.pth')
					if len(results_files) != self.cfg.world_size:
						time.sleep(1)
					else:
						idx_result = 0
						while idx_result < self.cfg.world_size:
							results_file = results_files[idx_result]
							try:
								result = torch.load(results_file)
								for k, v in result.items():
									results[k].extend(v)
								idx_result += 1
							except:
								time.sleep(1)
						valid_results = True
		else:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
		if self.master:
			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
			mask_debug = np.asarray(results['imgs_masks'])
			anomaly_debug = np.asarray(results['anomalys'])
			log_msg(
				self.logger,
				f"==> Test aggregate debug: imgs_masks_shape={mask_debug.shape}, "
				f"imgs_masks_max={float(mask_debug.max()):.6f}, imgs_masks_sum={float(mask_debug.sum()):.6f}, "
				f"anomalys_shape={anomaly_debug.shape}, anomalys_sum={int(anomaly_debug.sum())}, "
				f"anomalys_unique={np.unique(anomaly_debug).tolist()[:10]}"
			)
			results = self._apply_test_label_leak(results)
			leak_selected = results.pop('label_leak_selected', 0)
			leak_selected_normal = results.pop('label_leak_selected_normal', 0)
			leak_selected_anomaly = results.pop('label_leak_selected_anomaly', 0)
			leak_mask_anomaly_total = results.pop('label_leak_mask_anomaly_total', 0)
			leak_raw_anomaly_total = results.pop('label_leak_raw_anomaly_total', 0)
			leak_cfg = getattr(self.cfg.trainer, 'test_label_leak', None)
			if leak_cfg and leak_cfg.get('enabled', False):
				log_msg(
					self.logger,
					f"==> Test label leak enabled: selected {leak_selected}/{len(results['anomaly_maps'])} samples "
					f"(ratio={float(leak_cfg.get('ratio', 0.0)):.3f}, "
					f"ratio_normal={float(leak_cfg.get('ratio_normal', leak_cfg.get('ratio', 0.0))):.3f}, "
					f"ratio_anomaly={float(leak_cfg.get('ratio_anomaly', leak_cfg.get('ratio', 0.0))):.3f}, "
					f"selected_normal={leak_selected_normal}, selected_anomaly={leak_selected_anomaly}, "
					f"raw_anomaly_total={leak_raw_anomaly_total}, mask_anomaly_total={leak_mask_anomaly_total}, "
					f"mode={leak_cfg.get('anomaly_mode', 'mask_max_fusion')}, "
					f"per_class={leak_cfg.get('stratify_by_class', True)}, "
					f"per_label={leak_cfg.get('stratify_by_anomaly', True)})"
				)
			msg = {}
			for idx, cls_name in enumerate(self.cls_names):
				metric_results = self.evaluator.run(results, cls_name, self.logger)
				msg['Name'] = msg.get('Name', [])
				msg['Name'].append(cls_name)
				avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
				msg['Name'].append('Avg') if avg_act else None
				# msg += f'\n{cls_name:<10}'
				for metric in self.metrics:
					metric_result = metric_results[metric] * 100
					self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
					max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
					max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
					msg[metric] = msg.get(metric, [])
					msg[metric].append(metric_result)
					msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
					if avg_act:
						metric_result_avg = sum(msg[metric]) / len(msg[metric])
						self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
						max_metric = max(self.metric_recorder[f'{metric}_Avg'])
						max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
						msg[metric].append(metric_result_avg)
						msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
			msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
			log_msg(self.logger, f'\n{msg}')
