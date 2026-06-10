# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main API for computing and reporting quality metrics."""

import os
import time
import json
import copy
import numpy as np
import scipy.linalg
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import equivariance

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn
NUM_GEN_FOR_METRICS = int(os.environ.get('STYLEGAN_METRIC_NUM_GEN', '50000'))

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def _harmonic_mean(a, b):
    denom = a + b
    return 0.0 if denom <= 0 else 2 * a * b / denom

#----------------------------------------------------------------------------

def _compute_kid_from_features(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    total = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        total += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(total / num_subsets / m)

def _compute_is_from_probs(gen_probs, num_splits=10):
    scores = []
    num_gen = gen_probs.shape[0]
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

def _compute_pr_from_features(real_features, gen_features, opts, nhood_size=3, row_batch_size=10000, col_batch_size=10000):
    real_features = real_features.to(torch.float16).to(opts.device)
    gen_features = gen_features.to(torch.float16).to(opts.device)
    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = precision_recall.compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = precision_recall.compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']

def _compute_combined_generator_stats(opts, num_gen, batch_size=64, batch_gen=None):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    inception_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    vgg_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_gen)
    progress = opts.progress.sub(tag='generator combined', num_items=num_gen, rel_lo=0, rel_hi=1)
    inception = metric_utils.get_feature_detector(url=inception_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    vgg = metric_utils.get_feature_detector(url=vgg_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    inception_stats = metric_utils.FeatureStats(capture_all=True, capture_mean_cov=True, max_items=num_gen)
    prob_stats = metric_utils.FeatureStats(capture_all=True, max_items=num_gen)
    vgg_stats = metric_utils.FeatureStats(capture_all=True, max_items=num_gen)

    while not inception_stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            img = G(z=z, c=next(c_iter), **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        inception_features = inception(images, return_features=True)
        inception_probs = inception(images, no_output_bias=True)
        vgg_features = vgg(images, return_features=True)
        inception_stats.append_torch(inception_features, num_gpus=opts.num_gpus, rank=opts.rank)
        prob_stats.append_torch(inception_probs, num_gpus=opts.num_gpus, rank=opts.rank)
        vgg_stats.append_torch(vgg_features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(inception_stats.num_items)
    return inception_stats, prob_stats, vgg_stats

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Recommended metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=NUM_GEN_FOR_METRICS)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=NUM_GEN_FOR_METRICS, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=NUM_GEN_FOR_METRICS, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall, pr50k3_full_toppr=_harmonic_mean(precision, recall))

@register_metric
def fid_kid_is_pr_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    inception_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    vgg_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'

    real_inception_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=inception_url, detector_kwargs=dict(return_features=True),
        rel_lo=0, rel_hi=0, capture_all=True, capture_mean_cov=True, max_items=None)
    real_vgg_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=vgg_url, detector_kwargs=dict(return_features=True),
        rel_lo=0, rel_hi=0, capture_all=True, max_items=200000).get_all_torch()
    gen_inception_stats, gen_prob_stats, gen_vgg_stats = _compute_combined_generator_stats(
        opts=opts, num_gen=NUM_GEN_FOR_METRICS)
    precision, recall = _compute_pr_from_features(
        real_features=real_vgg_features, gen_features=gen_vgg_stats.get_all_torch(), opts=opts,
        nhood_size=3, row_batch_size=10000, col_batch_size=10000)

    if opts.rank != 0:
        return dict(
            fid50k_full=float('nan'), kid50k_full=float('nan'),
            is50k_mean=float('nan'), is50k_std=float('nan'),
            pr50k3_full_precision=precision, pr50k3_full_recall=recall, pr50k3_full_toppr=_harmonic_mean(precision, recall))

    mu_real, sigma_real = real_inception_stats.get_mean_cov()
    real_inception_features = real_inception_stats.get_all()
    mu_gen, sigma_gen = gen_inception_stats.get_mean_cov()
    gen_inception_features = gen_inception_stats.get_all()
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = float(np.real(m + np.trace(sigma_gen + sigma_real - s * 2)))
    kid = _compute_kid_from_features(real_inception_features, gen_inception_features, num_subsets=100, max_subset_size=1000)
    is_mean, is_std = _compute_is_from_probs(gen_prob_stats.get_all(), num_splits=10)
    return dict(
        fid50k_full=fid,
        kid50k_full=kid,
        is50k_mean=is_mean,
        is50k_std=is_std,
        pr50k3_full_precision=precision,
        pr50k3_full_recall=recall,
        pr50k3_full_toppr=_harmonic_mean(precision, recall))

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def eqt50k_int(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_int=True)
    return dict(eqt50k_int=psnr)

@register_metric
def eqt50k_frac(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_frac=True)
    return dict(eqt50k_frac=psnr)

@register_metric
def eqr50k(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqr=True)
    return dict(eqr50k=psnr)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=NUM_GEN_FOR_METRICS)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=NUM_GEN_FOR_METRICS, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=NUM_GEN_FOR_METRICS, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall, pr50k3_toppr=_harmonic_mean(precision, recall))

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=NUM_GEN_FOR_METRICS, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

#----------------------------------------------------------------------------
