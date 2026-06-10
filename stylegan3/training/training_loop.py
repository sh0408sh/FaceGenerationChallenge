# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
from pathlib import Path
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
import generate_seed_eval
from metrics import metric_main

#----------------------------------------------------------------------------

def _parse_stylegan2_resolution(name):
    parts = name.split('.')
    for part in parts:
        if len(part) > 1 and part[0] == 'b' and part[1:].isdigit():
            return int(part[1:])
    return None

#----------------------------------------------------------------------------

def _parse_stylegan3_size(name):
    parts = name.split('.')
    for part in parts:
        if part.startswith('L'):
            chunks = part.split('_')
            if len(chunks) >= 2 and chunks[0][1:].isdigit() and chunks[1].isdigit():
                return int(chunks[1])
    return None

#----------------------------------------------------------------------------

def _classify_g_param(name, freeze_mapping=True, freeze_coarse_until=0, train_synthesis_until=0, fine_from=0):
    if name.startswith('mapping.'):
        return False if freeze_mapping else True, False
    if name.startswith('synthesis.input.'):
        return freeze_coarse_until <= 0, False
    res_or_size = _parse_stylegan2_resolution(name)
    if res_or_size is None:
        res_or_size = _parse_stylegan3_size(name)
    if res_or_size is not None:
        if freeze_coarse_until > 0 and res_or_size <= freeze_coarse_until:
            return False, False
        if train_synthesis_until > 0 and res_or_size > train_synthesis_until:
            return False, False
        is_fine = fine_from > 0 and res_or_size >= fine_from
        return True, is_fine
    return True, False

#----------------------------------------------------------------------------

def _build_g_param_groups(G, base_lr, freeze_mapping=True, freeze_coarse_until=0, train_synthesis_until=0, fine_from=0, fine_lr_mul=1):
    main_params = []
    fine_params = []
    trainable_names = set()
    frozen_count = 0
    trainable_count = 0
    for name, param in G.named_parameters():
        trainable, is_fine = _classify_g_param(
            name, freeze_mapping=freeze_mapping, freeze_coarse_until=freeze_coarse_until,
            train_synthesis_until=train_synthesis_until, fine_from=fine_from)
        param.requires_grad_(False)
        if not trainable:
            frozen_count += param.numel()
            continue
        trainable_names.add(name)
        trainable_count += param.numel()
        if is_fine and fine_lr_mul != 1:
            fine_params.append(param)
        else:
            main_params.append(param)
    groups = []
    if len(main_params) > 0:
        groups.append(dict(params=main_params))
    if len(fine_params) > 0:
        groups.append(dict(params=fine_params, lr=base_lr * fine_lr_mul))
    return groups, trainable_names, trainable_count, frozen_count

#----------------------------------------------------------------------------

def _set_trainable(module, enabled, trainable_names=None):
    if not enabled:
        module.requires_grad_(False)
        return
    if trainable_names is None:
        module.requires_grad_(True)
        return
    for name, param in module.named_parameters():
        param.requires_grad_(name in trainable_names)

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def _assert_compatible_module(src_module, dst_module, label, min_match_ratio=0.5):
    src_tensors = dict(misc.named_params_and_buffers(src_module))
    dst_tensors = dict(misc.named_params_and_buffers(dst_module))
    matched = 0
    mismatched = []
    for name, dst in dst_tensors.items():
        src = src_tensors.get(name)
        if src is None:
            continue
        matched += 1
        if tuple(src.shape) != tuple(dst.shape):
            mismatched.append((name, tuple(src.shape), tuple(dst.shape)))
    match_ratio = matched / max(len(dst_tensors), 1)
    if mismatched or match_ratio < min_match_ratio:
        preview = '\n'.join(
            f'    {name}: source {src_shape} vs target {dst_shape}'
            for name, src_shape, dst_shape in mismatched[:8])
        if not preview:
            preview = f'    only {matched}/{len(dst_tensors)} tensors matched by name'
        raise RuntimeError(
            f'Incompatible network pickle for {label}. This usually means --cfg does not match '
            f'the checkpoint family, e.g. StyleGAN2 pkl with --cfg=stylegan3-* or vice versa.\n{preview}')

#----------------------------------------------------------------------------

def _write_quick_eval_csv(path, rows):
    import csv
    fields = ['kimg', 'snapshot_pkl', 'fid', 'is', 'is_std', 'kid', 'precision', 'recall', 'toppr', 'is_best_fid']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

#----------------------------------------------------------------------------

def _summarize_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return dict(
        count=int(logits.size),
        logit_mean=float(np.mean(logits)),
        logit_std=float(np.std(logits)),
        logit_min=float(np.min(logits)),
        logit_p05=float(np.percentile(logits, 5)),
        logit_p25=float(np.percentile(logits, 25)),
        logit_median=float(np.percentile(logits, 50)),
        logit_p75=float(np.percentile(logits, 75)),
        logit_p95=float(np.percentile(logits, 95)),
        logit_max=float(np.max(logits)),
        prob_real_mean=float(np.mean(probs)),
        prob_real_std=float(np.std(probs)),
        real_decision_rate_logit_gt_0=float(np.mean(logits > 0)),
        fake_decision_rate_logit_lt_0=float(np.mean(logits < 0)),
    )

#----------------------------------------------------------------------------

@torch.no_grad()
def _eval_discriminator_after_warmup(run_dir, D, G, ffhq_G, training_set, device, num_images=5000, batch_gpu=16, seed=0):
    import csv
    outdir = os.path.join(run_dir, 'd_warmup_eval')
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, 'discriminator_scores_raw.csv')
    summary_path = os.path.join(outdir, 'discriminator_scores_summary.csv')
    json_path = os.path.join(outdir, 'discriminator_scores_summary.json')

    def write_rows(writer, group, logits, offset):
        for idx, logit in enumerate(logits):
            logit = float(logit)
            writer.writerow(dict(
                group=group,
                index=offset + idx,
                logit=logit,
                prob_real=float(1.0 / (1.0 + np.exp(-logit))),
                decision='real' if logit > 0 else 'fake',
            ))

    def score_real(writer):
        logits_all = []
        num = min(num_images, len(training_set))
        for start in range(0, num, batch_gpu):
            cur = min(batch_gpu, num - start)
            images = []
            for idx in range(start, start + cur):
                img, _label = training_set[idx]
                images.append(img)
            img = torch.from_numpy(np.stack(images)).to(device).to(torch.float32) / 127.5 - 1
            c = torch.zeros([cur, D.c_dim], device=device)
            logits = D(img, c).detach().flatten().cpu().numpy()
            write_rows(writer, 'real_dataset', logits, start)
            logits_all.extend(logits.tolist())
        return logits_all

    def score_generator(writer, group, generator):
        logits_all = []
        for start in range(0, num_images, batch_gpu):
            cur = min(batch_gpu, num_images - start)
            rng = np.random.RandomState(seed + start)
            z = torch.from_numpy(rng.randn(cur, generator.z_dim)).to(device)
            c = torch.zeros([cur, generator.c_dim], device=device)
            img = generator(z, c, truncation_psi=1.0, noise_mode='const')
            d_c = torch.zeros([cur, D.c_dim], device=device)
            logits = D(img, d_c).detach().flatten().cpu().numpy()
            write_rows(writer, group, logits, start)
            logits_all.extend(logits.tolist())
        return logits_all

    rows = []
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'index', 'logit', 'prob_real', 'decision'])
        writer.writeheader()
        real_logits = score_real(writer)
        rows.append(dict(group='real_dataset', **_summarize_logits(real_logits)))
        g_logits = score_generator(writer, 'current_G_generated', G)
        rows.append(dict(group='current_G_generated', **_summarize_logits(g_logits)))
        if ffhq_G is not None:
            ffhq_logits = score_generator(writer, 'ffhq_anchor_generated', ffhq_G)
            rows.append(dict(group='ffhq_anchor_generated', **_summarize_logits(ffhq_logits)))

    with open(summary_path, 'w', newline='') as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    return rows, outdir

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_g_only           = False,    # Load only G/G_ema from resume_pkl, leaving D randomly initialized.
    resume_d_pkl            = None,     # Optional network pickle to load D from after resume_pkl.
    teacher_pkl             = None,     # Frozen source generator pickle for statistic distillation.
    l2sp_anchor_pkl         = None,     # Optional frozen generator pickle for L2-SP anchor.
    face_anchor_pkl         = None,     # Optional frozen generator pickle for ArcFace consistency anchor.
    freeze_g_mapping        = False,    # Freeze G mapping network during fine-tuning?
    freeze_g_coarse_until   = 0,        # Freeze synthesis params at resolution/size <= this value.
    train_g_synthesis_until = 0,        # Train only synthesis params at resolution/size <= this value, 0 = all.
    g_fine_from             = 0,        # Apply fine LR multiplier to synthesis params at resolution/size >= this value.
    g_fine_lr_mul           = 1,        # LR multiplier for fine synthesis params.
    metric_data             = None,     # Optional real dataset for built-in metrics.
    quick_eval_data         = None,     # Optional real dataset for quick 1k snapshot eval during training.
    quick_eval_interval_kimg= 0,        # Evaluate every N kimg, 0 = disable.
    quick_eval_num_images   = 1000,     # Number of generated images for quick eval.
    quick_eval_seed         = 0,        # Latent seed for quick eval image generation.
    quick_eval_batch_size   = 16,       # Batch size for quick eval image generation/feature extraction.
    quick_eval_trunc        = 1.0,      # Truncation psi for quick eval.
    quick_eval_noise_mode   = 'const',  # Noise mode for quick eval.
    quick_eval_save_best    = True,     # Save a copy whenever quick eval FID reaches a new best.
    save_regular_snapshots  = True,     # Save regular image/network snapshots.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    d_warmup_kimg           = 0,        # Train D only for this many kimg after resume_kimg.
    d_warmup_eval_num       = 0,        # Evaluate D score distributions after D-only warmup; 0 = disable.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()
    metric_dataset_kwargs = dnnlib.EasyDict(training_set_kwargs)
    if metric_data is not None:
        metric_dataset_kwargs.path = metric_data
        if rank == 0:
            print(f'Built-in metrics real dataset override: {metric_data}')
            print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        resume_modules = [('G', G), ('G_ema', G_ema)] if resume_g_only else [('G', G), ('D', D), ('G_ema', G_ema)]
        if resume_g_only:
            print('Loading G/G_ema only; keeping D randomly initialized.')
        for name, module in resume_modules:
            _assert_compatible_module(resume_data[name], module, f'resume {name}')
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    if (resume_d_pkl is not None) and (rank == 0):
        print(f'Loading discriminator from "{resume_d_pkl}"')
        with dnnlib.util.open_url(resume_d_pkl) as f:
            resume_d_data = legacy.load_network_pkl(f)
        _assert_compatible_module(resume_d_data['D'], D, 'resume D')
        misc.copy_params_and_buffers(resume_d_data['D'], D, require_all=False)
    if (resume_pkl is not None or resume_d_pkl is not None) and (num_gpus > 1):
        for module in [G, D, G_ema]:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Frozen reference networks for distillation/regularization.
    teacher_G = None
    if teacher_pkl is not None:
        if rank == 0:
            print(f'Loading frozen teacher generator from "{teacher_pkl}"')
        with dnnlib.util.open_url(teacher_pkl) as f:
            teacher_data = legacy.load_network_pkl(f)
        teacher_G = teacher_data['G_ema'].eval().requires_grad_(False).to(device)
        if teacher_G.img_resolution != G.img_resolution or teacher_G.img_channels != G.img_channels:
            raise RuntimeError(f'Teacher output shape mismatch: teacher {teacher_G.img_resolution}x{teacher_G.img_channels}, student {G.img_resolution}x{G.img_channels}')
    if l2sp_anchor_pkl is not None:
        if rank == 0:
            print(f'Loading frozen L2-SP anchor generator from "{l2sp_anchor_pkl}"')
        with dnnlib.util.open_url(l2sp_anchor_pkl) as f:
            l2sp_data = legacy.load_network_pkl(f)
        l2sp_G = l2sp_data['G_ema'].eval().requires_grad_(False).to(device)
        _assert_compatible_module(l2sp_G, G, 'L2-SP anchor G_ema')
        if l2sp_G.img_resolution != G.img_resolution or l2sp_G.img_channels != G.img_channels:
            raise RuntimeError(f'L2-SP anchor output shape mismatch: anchor {l2sp_G.img_resolution}x{l2sp_G.img_channels}, student {G.img_resolution}x{G.img_channels}')
    else:
        l2sp_G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    face_anchor_G = None
    if face_anchor_pkl is not None:
        if rank == 0:
            print(f'Loading frozen face anchor generator from "{face_anchor_pkl}"')
        with dnnlib.util.open_url(face_anchor_pkl) as f:
            face_anchor_data = legacy.load_network_pkl(f)
        face_anchor_G = face_anchor_data['G_ema'].eval().requires_grad_(False).to(device)
        if face_anchor_G.img_resolution != G.img_resolution or face_anchor_G.img_channels != G.img_channels:
            raise RuntimeError(f'Face anchor output shape mismatch: anchor {face_anchor_G.img_resolution}x{face_anchor_G.img_channels}, student {G.img_resolution}x{G.img_channels}')

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, teacher_G=teacher_G, l2sp_G=l2sp_G, face_anchor_G=face_anchor_G, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    G_trainable_names = None
    G_param_groups = None
    if freeze_g_mapping or freeze_g_coarse_until > 0 or train_g_synthesis_until > 0 or g_fine_from > 0 or g_fine_lr_mul != 1:
        G_param_groups, G_trainable_names, trainable_count, frozen_count = _build_g_param_groups(
            G, base_lr=G_opt_kwargs.lr, freeze_mapping=freeze_g_mapping,
            freeze_coarse_until=freeze_g_coarse_until, train_synthesis_until=train_g_synthesis_until,
            fine_from=g_fine_from, fine_lr_mul=g_fine_lr_mul)
        if rank == 0:
            print(f'G trainable parameters: {trainable_count:,}')
            print(f'G frozen parameters:    {frozen_count:,}')
            print(f'G freeze mapping={freeze_g_mapping}, freeze coarse <= {freeze_g_coarse_until}, train synthesis <= {train_g_synthesis_until}, fine >= {g_fine_from}, fine lr mul={g_fine_lr_mul}')
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        params = G_param_groups if name == 'G' and G_param_groups is not None else module.parameters()
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            if name == 'G' and G_param_groups is not None:
                for group in G_param_groups:
                    if 'lr' in group:
                        group['lr'] *= mb_ratio
                params = G_param_groups
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.trainable_names = G_trainable_names if phase.name.startswith('G') else None
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        phase.ran_since_tick = False
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)
    quick_eval_real_stats = None
    quick_eval_rows = []
    quick_eval_best_fid = None
    quick_eval_last_kimg = None
    builtin_metric_best_fid = None
    quick_eval_dir = os.path.join(run_dir, 'quick_eval')
    if rank == 0 and quick_eval_data is not None and quick_eval_interval_kimg > 0:
        os.makedirs(quick_eval_dir, exist_ok=True)
        print(f'Quick eval enabled: data={quick_eval_data}, interval={quick_eval_interval_kimg} kimg, num_images={quick_eval_num_images}')

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        if d_warmup_kimg > 0:
            print(f'D-only warmup enabled for {d_warmup_kimg} kimg: G phases skipped until kimg {resume_kimg + d_warmup_kimg}.')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    d_warmup_eval_done = False
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if d_warmup_kimg > 0 and phase.name.startswith('G') and cur_nimg < (resume_kimg + d_warmup_kimg) * 1000:
                continue
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            _set_trainable(phase.module, True, phase.trainable_names)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            _set_trainable(phase.module, False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
            phase.ran_since_tick = True

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        warmup_end_nimg = (resume_kimg + d_warmup_kimg) * 1000
        do_d_warmup_eval = (
            d_warmup_kimg > 0 and
            d_warmup_eval_num > 0 and
            not d_warmup_eval_done and
            cur_nimg >= warmup_end_nimg
        )
        if rank == 0 and do_d_warmup_eval:
            print(f'Evaluating discriminator after D-only warmup at kimg {cur_nimg / 1e3:.1f}...')
            rows, eval_outdir = _eval_discriminator_after_warmup(
                run_dir=run_dir, D=D, G=G_ema, ffhq_G=l2sp_G, training_set=training_set,
                device=device, num_images=d_warmup_eval_num, batch_gpu=batch_gpu, seed=random_seed)
            print(f'D warmup discriminator eval written to {eval_outdir}')
            print(json.dumps(rows, indent=2))
        if num_gpus > 1 and do_d_warmup_eval:
            torch.distributed.barrier()
        if do_d_warmup_eval:
            d_warmup_eval_done = True

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        in_d_warmup = d_warmup_kimg > 0 and cur_nimg < (resume_kimg + d_warmup_kimg) * 1000
        if (rank == 0) and (not in_d_warmup) and save_regular_snapshots and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        regular_snapshot_due = (not in_d_warmup) and save_regular_snapshots and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)
        quick_eval_due = (
            (not in_d_warmup) and
            quick_eval_data is not None and
            quick_eval_interval_kimg > 0 and
            (done or (cur_nimg // 1000) % quick_eval_interval_kimg == 0) and
            quick_eval_last_kimg != cur_nimg // 1000
        )
        if regular_snapshot_due or quick_eval_due:
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            if regular_snapshot_due:
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0 and regular_snapshot_due:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if regular_snapshot_due and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=metric_dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    if 'fid50k_full' in result_dict.results:
                        fid = float(result_dict.results['fid50k_full'])
                        if builtin_metric_best_fid is None or fid < builtin_metric_best_fid:
                            builtin_metric_best_fid = fid
                            best_pkl = os.path.join(run_dir, 'network-snapshot-best-fid.pkl')
                            best_info = os.path.join(run_dir, 'network-snapshot-best-fid.json')
                            with open(best_pkl, 'wb') as f:
                                pickle.dump(snapshot_data, f)
                            with open(best_info, 'w', encoding='utf-8') as f:
                                json.dump(dict(
                                    metric='fid50k_full',
                                    fid50k_full=fid,
                                    kimg=cur_nimg // 1000,
                                    snapshot_pkl=snapshot_pkl,
                                    best_pkl=best_pkl,
                                    metric_data=metric_data if metric_data is not None else training_set_kwargs.get('path', None),
                                ), f, indent=2, sort_keys=True)
                            print(f'New best fid50k_full {fid:.6f}; saved {best_pkl}')
                stats_metrics.update(result_dict.results)

        # Quick 1k eval against a separate real dataset, plus best-FID snapshot copy.
        if quick_eval_due and snapshot_data is not None:
            if rank == 0:
                print('Evaluating quick metrics...')
                quick_eval_last_kimg = cur_nimg // 1000
                if quick_eval_real_stats is None:
                    quick_eval_real_stats = generate_seed_eval.compute_real_stats(
                        data=Path(quick_eval_data).resolve(),
                        resolution=snapshot_data['G_ema'].img_resolution,
                        device=device)
                eval_outdir = os.path.join(quick_eval_dir, f'snapshot_{cur_nimg//1000:06d}')
                eval_G = copy.deepcopy(snapshot_data['G_ema']).eval().requires_grad_(False).to(device)
                metrics_row = generate_seed_eval.evaluate_generator(
                    G=eval_G,
                    dataset_path=Path(quick_eval_data).resolve(),
                    real_inception_stats=quick_eval_real_stats[0],
                    real_vgg_stats=quick_eval_real_stats[1],
                    resolution=snapshot_data['G_ema'].img_resolution,
                    device=device,
                    num_images=quick_eval_num_images,
                    batch_size=quick_eval_batch_size,
                    seed=quick_eval_seed,
                    truncation_psi=quick_eval_trunc,
                    noise_mode=quick_eval_noise_mode)
                del eval_G
                is_best = quick_eval_best_fid is None or metrics_row['fid'] < quick_eval_best_fid
                if is_best:
                    quick_eval_best_fid = metrics_row['fid']
                    if quick_eval_save_best:
                        best_pkl = os.path.join(run_dir, 'network-snapshot-best-fid.pkl')
                        best_info = os.path.join(run_dir, 'network-snapshot-best-fid.json')
                        with open(best_pkl, 'wb') as f:
                            pickle.dump(snapshot_data, f)
                        with open(best_info, 'w', encoding='utf-8') as f:
                            json.dump(dict(kimg=cur_nimg // 1000, snapshot_pkl=snapshot_pkl, best_pkl=best_pkl, **metrics_row), f, indent=2, sort_keys=True)
                row = dict(kimg=cur_nimg // 1000, snapshot_pkl=snapshot_pkl, **metrics_row, is_best_fid=int(is_best))
                quick_eval_rows.append(row)
                _write_quick_eval_csv(os.path.join(quick_eval_dir, 'quick_eval_metrics.csv'), quick_eval_rows)
                with open(os.path.join(quick_eval_dir, 'quick_eval_metrics.json'), 'w', encoding='utf-8') as f:
                    json.dump(quick_eval_rows, f, indent=2, sort_keys=True)
                print(json.dumps(row, sort_keys=True))
                stats_metrics.update({
                    'quick_fid1k': metrics_row['fid'],
                    'quick_is1k': metrics_row['is'],
                    'quick_kid1k': metrics_row['kid'],
                    'quick_precision1k': metrics_row['precision'],
                    'quick_recall1k': metrics_row['recall'],
                    'quick_toppr1k': metrics_row['toppr'],
                    'quick_best_fid1k': quick_eval_best_fid,
                })
            if num_gpus > 1:
                torch.distributed.barrier()
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if phase.ran_since_tick and (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
            phase.ran_since_tick = False
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
