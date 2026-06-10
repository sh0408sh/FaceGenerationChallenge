# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class VGG16FeatureStats(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError, AttributeError):
            vgg = vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
            vgg[16:23], # relu4_3
        ])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def forward(self, img):
        x = (img.to(torch.float32) + 1) * 0.5
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        stats = []
        for block in self.blocks:
            x = block(x)
            stats.append(x.mean(dim=[0, 2, 3]))
            stats.append(x.std(dim=[0, 2, 3], unbiased=False))
        return torch.cat(stats)

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
        teacher_G=None, l2sp_G=None, face_anchor_G=None, lambda_stat=0, lambda_freq=0, lambda_l2sp=0, lambda_lpips=0, lambda_face=0, distill_feature='multiscale', lpips_net='vgg', distill_interval=1, distill_batch=0, freq_alpha=0.1, freq_beta=0.3,
        arcface_pth=None, face_interval=4, face_batch=4):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.teacher_G          = teacher_G
        self.face_anchor_G      = face_anchor_G
        self.l2sp_params        = {name: param.detach().clone() for name, param in l2sp_G.named_parameters()} if l2sp_G is not None else None
        self.lambda_stat        = lambda_stat
        self.lambda_freq        = lambda_freq
        self.lambda_l2sp        = lambda_l2sp
        self.lambda_lpips       = lambda_lpips
        self.lambda_face        = lambda_face
        self.distill_feature    = distill_feature
        self.distill_interval   = max(int(distill_interval), 1)
        self.distill_batch      = int(distill_batch)
        self.face_interval      = max(int(face_interval), 1)
        self.face_batch         = int(face_batch)
        self.freq_alpha         = freq_alpha
        self.freq_beta          = max(freq_beta, freq_alpha)
        self.gmain_calls        = 0
        self.lap_kernel         = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32, device=device)
        self.vgg_feature_stats  = None
        self.lpips_model        = None
        self.arcface_model      = None
        if self.lambda_stat > 0 and self.distill_feature == 'vgg16':
            self.vgg_feature_stats = VGG16FeatureStats().to(device)
        if self.lambda_lpips > 0:
            import lpips
            self.lpips_model = lpips.LPIPS(net=lpips_net, verbose=False).eval().requires_grad_(False).to(device)
        if self.lambda_face > 0:
            if self.face_anchor_G is None:
                raise RuntimeError('face_anchor_G is required when lambda_face is nonzero')
            if arcface_pth is None:
                raise RuntimeError('arcface_pth is required when lambda_face is nonzero')
            from training.arcface_torch import ArcFaceEmbedder
            self.arcface_model = ArcFaceEmbedder(arcface_pth, device).eval().requires_grad_(False).to(device)

    def _make_teacher_images(self, batch_size):
        assert self.teacher_G is not None
        z = torch.randn([batch_size, self.teacher_G.z_dim], device=self.device)
        c = torch.zeros([batch_size, self.teacher_G.c_dim], device=self.device)
        with torch.no_grad():
            img = self.teacher_G(z, c, noise_mode='random')
        return img.detach()

    @staticmethod
    def _feature_stats(img):
        feats = []
        x = img.to(torch.float32)
        for scale in [1, 2, 4, 8]:
            cur = x if scale == 1 else F.avg_pool2d(x, kernel_size=scale, stride=scale)
            feats.append(cur.mean(dim=[0, 2, 3]))
            feats.append(cur.std(dim=[0, 2, 3], unbiased=False))
        return torch.cat(feats)

    def _feature_stat_loss(self, student_img, teacher_img):
        if self.vgg_feature_stats is not None:
            student_stats = self.vgg_feature_stats(student_img)
            with torch.no_grad():
                teacher_stats = self.vgg_feature_stats(teacher_img)
        else:
            student_stats = self._feature_stats(student_img)
            teacher_stats = self._feature_stats(teacher_img)
        return F.mse_loss(student_stats, teacher_stats)

    def _lpips_loss(self, student_img, teacher_img):
        assert self.lpips_model is not None
        with torch.no_grad():
            teacher_img = teacher_img.detach()
        return self.lpips_model(student_img, teacher_img).mean()

    def _face_consistency_loss(self, student_img, anchor_img):
        assert self.arcface_model is not None
        student_embed = self.arcface_model(student_img)
        with torch.no_grad():
            anchor_embed = self.arcface_model(anchor_img.detach())
        loss = 1 - (student_embed * anchor_embed).sum(dim=1)
        return loss.mean()

    def _hf_stats(self, img):
        channels = img.shape[1]
        kernel = self.lap_kernel.to(img.dtype).repeat(channels, 1, 1, 1)
        lap = F.conv2d(img.to(torch.float32), kernel.to(torch.float32), padding=1, groups=channels).abs()
        energy = lap.mean(dim=[1, 2, 3])
        return energy.mean(), energy.std(unbiased=False)

    def _frequency_range_loss(self, student_img, teacher_img, real_img):
        s_mean, s_std = self._hf_stats(student_img)
        t_mean, t_std = self._hf_stats(teacher_img)
        r_mean, r_std = self._hf_stats(real_img.detach())
        min_mean = r_mean + self.freq_alpha * (t_mean - r_mean)
        max_mean = r_mean + self.freq_beta * (t_mean - r_mean)
        min_std = r_std + self.freq_alpha * (t_std - r_std)
        max_std = r_std + self.freq_beta * (t_std - r_std)
        lo_mean, hi_mean = torch.minimum(min_mean, max_mean), torch.maximum(min_mean, max_mean)
        lo_std, hi_std = torch.minimum(min_std, max_std), torch.maximum(min_std, max_std)
        loss = F.relu(lo_mean - s_mean).square() + 0.5 * F.relu(s_mean - hi_mean).square()
        loss = loss + F.relu(lo_std - s_std).square() + 0.5 * F.relu(s_std - hi_std).square()
        training_stats.report('Loss/distill/hf_student_mean', s_mean.detach())
        training_stats.report('Loss/distill/hf_real_mean', r_mean.detach())
        training_stats.report('Loss/distill/hf_teacher_mean', t_mean.detach())
        return loss

    def _l2sp_loss(self):
        if self.l2sp_params is None:
            return None
        loss = torch.zeros([], device=self.device)
        count = 0
        for name, param in self.G.named_parameters():
            ref = self.l2sp_params.get(name)
            if ref is None or not param.requires_grad:
                continue
            loss = loss + (param.float() - ref.float()).square().mean()
            count += 1
        if count == 0:
            return None
        return loss / count

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                loss_Gtotal = loss_Gmain.mean()

                self.gmain_calls += 1
                use_distill = (self.teacher_G is not None) and (self.gmain_calls % self.distill_interval == 0)
                if use_distill and (self.lambda_stat > 0 or self.lambda_freq > 0 or self.lambda_lpips > 0):
                    d_batch = min(self.distill_batch or gen_img.shape[0], gen_img.shape[0])
                    teacher_img = self._make_teacher_images(d_batch)
                    student_img = gen_img[:d_batch]
                    real_img_distill = real_img[:d_batch]
                    if self.lambda_stat > 0:
                        loss_stat = self._feature_stat_loss(student_img, teacher_img)
                        training_stats.report('Loss/distill/feature_stat', loss_stat)
                        loss_Gtotal = loss_Gtotal + loss_stat * self.lambda_stat
                    if self.lambda_freq > 0:
                        loss_freq = self._frequency_range_loss(student_img, teacher_img, real_img_distill)
                        training_stats.report('Loss/distill/frequency', loss_freq)
                        loss_Gtotal = loss_Gtotal + loss_freq * self.lambda_freq
                    if self.lambda_lpips > 0:
                        loss_lpips = self._lpips_loss(student_img, teacher_img)
                        training_stats.report('Loss/distill/lpips_pair', loss_lpips)
                        loss_Gtotal = loss_Gtotal + loss_lpips * self.lambda_lpips

                if self.lambda_l2sp > 0:
                    loss_l2sp = self._l2sp_loss()
                    if loss_l2sp is not None:
                        training_stats.report('Loss/distill/l2sp', loss_l2sp)
                        loss_Gtotal = loss_Gtotal + loss_l2sp * self.lambda_l2sp

                use_face = (self.face_anchor_G is not None) and (self.gmain_calls % self.face_interval == 0)
                if use_face and self.lambda_face > 0:
                    f_batch = min(self.face_batch or gen_img.shape[0], gen_img.shape[0])
                    with torch.no_grad():
                        anchor_img = self.face_anchor_G(gen_z[:f_batch], gen_c[:f_batch], noise_mode='const')
                    loss_face = self._face_consistency_loss(gen_img[:f_batch], anchor_img)
                    training_stats.report('Loss/distill/face_consistency', loss_face)
                    loss_Gtotal = loss_Gtotal + loss_face * self.lambda_face
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gtotal.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
