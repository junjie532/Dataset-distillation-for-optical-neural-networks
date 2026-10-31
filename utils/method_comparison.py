"""Fresh SI method comparison with held-out images and transmission matrices.

This module never reads archived supports, losses or accuracy results. Only the
official Fashion-MNIST training/test images are inputs to the experiment.
"""
from __future__ import annotations
from .datasets import flat_result_path, flat_result_directory

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .compression_baselines import (
    balanced_validation_indices, classwise_herding, classwise_kmeans_prototypes,
    optimize_rbf_kip, optimize_optical_dm, raw_optical_intensity,
    ridge_accuracy_from_raw,
)
from .datasets import random_coreset, save_npz
from .distillation import DistillationConfig, distill_coreset
from .reservoir_onn import tm_ensemble

METHODS = ('random', 'herding', 'kmeans', 'rbf_kip', 'optical_dm', 'ours')
LABELS = {'random': 'Random subset', 'herding': 'Pixel-space herding',
          'kmeans': 'K-means prototypes', 'rbf_kip': 'RBF-KIP',
          'optical_dm': 'Optical DM', 'ours': 'Ours'}


@dataclass(frozen=True)
class MethodComparisonConfig:
    ipc_values: tuple[int, ...] = (1, 2, 5, 10)
    main_repetitions: int = 20
    baseline_repetitions: int = 5
    output_dim: int = 1024
    steps: int = 700
    validation_steps: int = 700
    training_tm_count: int = 50
    test_tm_count: int = 50
    validation_tm_count: int = 10
    validation_per_class: int = 200
    split_seed: int = 33025
    ridge_grid: tuple[float, ...] = (.001, .003, .01, .03, .1, .3, 1., 3., 10.)
    query_batch_size: int = 1024
    tms_per_step: int = 4
    kmeans_max_iter: int = 40
    cache_training_features: bool = True


def _write_json(path, value):
    flat_result_directory(Path(path).parent).mkdir(parents=True, exist_ok=True)
    flat_result_path(Path(path)).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


@torch.no_grad()
def _raw_cache(x, tms, label, verbose=True):
    """Cache fixed real-image forward passes, without normalization or labels."""
    required = len(tms) * len(x) * tms[0].shape[1] * 4
    if x.is_cuda:
        free, _ = torch.cuda.mem_get_info(x.device)
        if required > free * .85:
            raise MemoryError(f'{label} cache requires {required / 2**30:.1f} GiB; '
                              'set cache_training_features=False to use uncached training.')
    if verbose: print(f'{label}: caching raw features ({required / 2**30:.2f} GiB)', flush=True)
    # One contiguous cache supports a single batched CUDA gather per update.
    cache=torch.empty((len(tms),len(x),tms[0].shape[1]),device=x.device,dtype=torch.float32)
    for index,tm in enumerate(tms):
        cache[index]=raw_optical_intensity(x,tm)
    return cache


@torch.no_grad()
def _dm_targets(x, y, tms, cache):
    means, scales = [], []
    for index, tm in enumerate(tms):
        raw = raw_optical_intensity(x, tm) if cache is None else cache[index]
        scale = raw.max().clamp_min(1e-12)
        means.append(torch.stack([raw[y == c].mean(dim=0) / scale for c in range(10)]))
        scales.append(scale)
    return torch.stack(means), torch.stack(scales)


@torch.no_grad()
def _evaluate(supports, x_query, y_query, tms, *, grid=None, beta=None, query_cache=None):
    """Compute each query forward pass once, retaining per-support normalization."""
    if (grid is None) == (beta is None):
        raise ValueError('Supply either a ridge grid or a beta mapping')
    keys = list(supports)
    sizes = [len(supports[k][0]) for k in keys]
    offsets = np.cumsum([0] + sizes)
    xs = torch.cat([supports[k][0] for k in keys])
    result = {k: np.empty((len(grid), len(tms)) if grid is not None else (len(tms),)) for k in keys}
    for j, tm in enumerate(tms):
        query_raw = raw_optical_intensity(x_query, tm) if query_cache is None else query_cache[j]
        support_raw = raw_optical_intensity(xs, tm)
        for i, key in enumerate(keys):
            z = support_raw[offsets[i]:offsets[i+1]]
            targets = supports[key][1]
            if grid is None:
                result[key][j] = ridge_accuracy_from_raw(z, targets, query_raw, y_query, beta[key])
            else:
                for g, value in enumerate(grid):
                    result[key][g,j] = ridge_accuracy_from_raw(z, targets, query_raw, y_query, value)
    return result


def _pm_config(config, ipc, beta, *, validation=False, rep=0):
    return DistillationConfig(images_per_class=int(ipc), ridge_lambda=float(beta),
        steps=config.validation_steps if validation else config.steps,
        target_batch_size=config.query_batch_size, tms_per_step=config.tms_per_step,
        coreset_seed=17 if validation else 2025+rep,
        batch_seed=123 if validation else 9025+rep, ridge_solver='dual',
        batched_cached_tms=config.cache_training_features)


def _support(config, method, x, y, ipc, seed, *, tms=None, means=None, scales=None, steps=None):
    diagnostics = {}
    if method == 'random':
        sx, sy, _, indices = random_coreset(x,y,ipc,seed)
        diagnostics['source_indices'] = indices
    elif method == 'herding':
        sx, sy, indices = classwise_herding(x,y,ipc)
        diagnostics['source_indices'] = indices
    elif method == 'kmeans':
        sx,sy,iterations = classwise_kmeans_prototypes(x,y,ipc,seed,config.kmeans_max_iter)
        diagnostics['iterations'] = iterations
    elif method == 'rbf_kip':
        sx,sy,diagnostics = optimize_rbf_kip(x,y,ipc,seed,
            config.steps if steps is None else steps,config.query_batch_size,.01,.01,.001)
    elif method == 'optical_dm':
        sx,sy,diagnostics = optimize_optical_dm(x,y,ipc,seed,tms,means,scales,
            config.steps if steps is None else steps,config.tms_per_step,.01,5.)
    else:
        raise ValueError(method)
    return (sx,sy),diagnostics


def _save_support(path, support, diagnostics, **metadata):
    save_npz(path, {'x':support[0], 'y':support[1], **diagnostics}, config=metadata)


def select_method_lambdas(x_train, y_train, config, destination, *, methods=METHODS, verbose=True, save_details=False):
    """Synthesize validation candidates without ever using held-out images.

    Ours is retrained separately at each candidate ridge value because its
    synthesis objective depends on that value. All methods share the same
    held-out images and held-out validation TMs.
    """
    destination = Path(destination)
    flat_result_directory(destination).mkdir(parents=True,exist_ok=True)
    validation_indices = balanced_validation_indices(y_train,config.validation_per_class,config.split_seed)
    keep = torch.ones(len(y_train),device=y_train.device,dtype=torch.bool)
    keep[validation_indices] = False
    training_indices = torch.nonzero(keep,as_tuple=True)[0]
    assert len(training_indices)+len(validation_indices)==len(y_train)
    assert not bool(keep[validation_indices].any())
    x,y = x_train[training_indices],y_train[training_indices]
    xv,yv = x_train[validation_indices],y_train[validation_indices]
    save_npz(destination / 'split.npz', {'training_indices':training_indices,
        'validation_indices':validation_indices}, config={'split_seed':config.split_seed,
        'validation_images_never_used_for_synthesis':True})
    train_seeds = list(range(40000,40000+config.training_tm_count))
    val_seeds = list(range(7000,7000+config.validation_tm_count))
    assert set(train_seeds).isdisjoint(val_seeds)
    tms = tm_ensemble(x.shape[1],config.output_dim,train_seeds,x.device)
    cache = _raw_cache(x,tms,'Validation synthesis',verbose) if config.cache_training_features else None
    means,scales = _dm_targets(x,y,tms,cache) if 'optical_dm' in methods else (None,None)
    val_tms = tm_ensemble(x.shape[1],config.output_dim,val_seeds,x.device)
    val_cache = _raw_cache(xv,val_tms,'Held-out validation',verbose)
    selected,scores = {},{}
    if verbose: print(f'Independent split: {len(x)} synthesis images / {len(xv)} validation images',flush=True)
    for ipc in config.ipc_values:
        support_dict = {}
        for method in (m for m in methods if m != 'ours'):
            support,diagnostics = _support(config,method,x,y,ipc,
                22025 if method=='random' else 2025,tms=tms,means=means,scales=scales,
                steps=config.validation_steps)
            support_dict[method,ipc] = support
            if save_details:
                _save_support(destination / f'{method}_ipc{ipc}.npz',support,diagnostics,
                    method=method,ipc=int(ipc),phase='held-out-validation candidate',
                    source_indices_refer_to='split.npz: training_indices',
                    validation_images_used_during_synthesis=False)
        evaluated = _evaluate(support_dict,xv,yv,val_tms,grid=config.ridge_grid,query_cache=val_cache) if support_dict else {}
        for key,values in evaluated.items():
            scores[key] = values
            selected[key] = float(config.ridge_grid[int(np.argmax(values.mean(axis=1)))])
        if 'ours' in methods:
            pm_scores = np.empty((len(config.ridge_grid),len(val_tms)))
            for g,beta in enumerate(config.ridge_grid):
                pm_cfg = _pm_config(config,ipc,beta,validation=True)
                trained = distill_coreset(x,y,tms,pm_cfg,precomputed_query_intensity=cache)
                pm_support = trained.x_coreset,trained.y_coreset
                pm_scores[g] = _evaluate({('ours',ipc):pm_support},xv,yv,val_tms,
                    beta={('ours',ipc):float(beta)},query_cache=val_cache)['ours',ipc]
                if save_details:
                    _save_support(destination / f'ours_ipc{ipc}_lambda{beta:g}.npz',pm_support,
                        {'loss_history':trained.loss_history,'source_indices':trained.coreset_indices,
                         'x_initial':trained.x_coreset_init,'temperature':trained.final_temperature},
                        method='ours',ipc=int(ipc),distillation=asdict(pm_cfg),
                        phase='held-out-validation candidate',validation_images_used_during_synthesis=False)
                if verbose: print(f'Validation Ours IPC={ipc}, lambda={beta:g}: {100*pm_scores[g].mean():.3f}%',flush=True)
            scores['ours',ipc] = pm_scores
            selected['ours',ipc] = float(config.ridge_grid[int(np.argmax(pm_scores.mean(axis=1)))])
        if verbose: print('Selected IPC='+str(ipc)+': '+', '.join(f'{m}={selected[m,ipc]:g}' for m in methods),flush=True)
        del support_dict
        if 'ours' in methods: del trained,pm_support
    arrays={'ridge_grid':config.ridge_grid,'training_tm_seeds':train_seeds,'validation_tm_seeds':val_seeds}
    for (method,ipc),values in scores.items():
        arrays[f'{method}_ipc{ipc}_accuracy_grid']=values
        arrays[f'{method}_ipc{ipc}_selected_lambda']=selected[method,ipc]
    save_npz(destination / 'selection.npz',arrays,config={'configuration':asdict(config),
        'validation_images_used_during_synthesis':False,'test_images_used_for_selection':False})
    _write_json(destination / 'selection.json',{method:{str(ipc):selected[method,ipc]
        for ipc in config.ipc_values} for method in methods})
    del cache,tms,means,scales,val_cache,val_tms,x,xv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected


