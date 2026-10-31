"""Retrospective simulated-random matching to measured classification coresets.

The three random permutations are fixed before scoring; no labels, ridge
parameters or TM seeds are selected using the matching scores. This estimates
a sample-equivalence on the existing test sets, not a held-out budget selection.
"""
from __future__ import annotations
from .datasets import flat_result_path, flat_result_directory

from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .classification import evaluate_support
from .evaluation import evaluate_coreset_on_tm
from .reservoir_onn import tm_ensemble

CONFIGS = {
    'agnews': dict(name='AG News', ipc=5, coarse=25, fine=5, cap=700),
    'cora': dict(name='Cora', ipc=2, coarse=10, fine=2, cap=None),
    'uci_har': dict(name='UCI-HAR', ipc=5, coarse=25, fine=5, cap=None),
}
RANDOM_SEEDS = (0, 1, 2)
BETA = 1.0


def _save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    flat_result_path(temporary).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    flat_result_path(temporary).replace(flat_result_path(path))


@torch.no_grad()
def _intensity(x, tms):
    field = torch.exp(1j * torch.pi * x.float())
    return torch.stack([torch.abs(field @ tm) ** 2 for tm in tms])


@torch.no_grad()
def _cached_scores(support_raw, targets, query_raw, truth, beta=BETA):
    scale = torch.maximum(support_raw.amax((1, 2)), query_raw.amax((1, 2)))
    scale = scale.clamp_min(1e-12)[:, None, None]
    support = support_raw / scale
    rhs = targets[None].expand(len(support), -1, -1)
    if support.shape[1] <= support.shape[2]:
        gram = support @ support.transpose(1, 2)
        gram.diagonal(dim1=1, dim2=2).add_(beta)
        weights = support.transpose(1, 2) @ torch.linalg.solve(gram, rhs)
    else:
        gram = support.transpose(1, 2) @ support
        gram.diagonal(dim1=1, dim2=2).add_(beta)
        weights = torch.linalg.solve(gram, support.transpose(1, 2) @ rhs)
    prediction = ((query_raw / scale) @ weights).argmax(2)
    return (prediction == truth[None]).float().mean(1).cpu().numpy().astype(float)


def _load_inputs(root, task, config):
    dataset_path = root / f'results/cache/datasets/{task}.npz'
    experiment_path = root / f'results/experimental/{task}_experiment.npz'
    with np.load(flat_result_path(dataset_path), allow_pickle=False) as data:
        if task == 'cora':
            train, test = data['train_indices'], data['test_indices']
            arrays = [data['x'][train], data['labels'][train],
                      data['x'][test], data['labels'][test]]
        else:
            arrays = [data[k].copy() for k in ['x_train', 'y_train', 'x_test', 'y_test']]
    with np.load(flat_result_path(experiment_path), allow_pickle=False) as data:
        column = list(data['size_values']).index(config['ipc'])
        targets = data['coreset_accuracy'][:, column].astype(float)
        equal_random = data['random_accuracy'][:, column].astype(float)
        reps = data['source_reps'].astype(int)
        acquisitions = data['source_acquisitions'].astype(str)
    archives = []
    for rep in sorted(set(reps)):
        path = root / f'results/simulation/multimodal/{task}/spc{config["ipc"]}_rep{rep-1}.npz'
        with np.load(flat_result_path(path), allow_pickle=False) as data:
            archives.append((rep, path, {k: data[k].copy() for k in data.files}))
    assert len(targets) == 10 and len(archives) == 5
    assert len(set(zip(acquisitions, reps))) == len(targets)
    assert all(sum(reps == r) == 2 for r, _, _ in archives)
    assert np.isfinite(targets).all() and ((targets >= 0) & (targets <= 1)).all()
    return arrays, targets, equal_random, reps, acquisitions, archives


@torch.no_grad()
def run_task(root, task, output_dir, *, device='cuda:1', recompute=False, sampling='balanced'):
    """Search one task; reuse results only when the requested configuration matches.

    Coarse points locate a useful upper budget. Every fine-grid point below
    that upper budget is then evaluated, so a nonmonotone curve cannot hide an
    earlier fine-grid crossing. Decision points and numerically close lower
    points are checked using the task's original uncached evaluator.
    """
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    assert sampling in ('balanced', 'uniform')
    config = dict(CONFIGS[task], sampling=sampling)
    data, targets, equal_random, reps, acquisitions, archives = _load_inputs(root, task, config)
    suffix = '_uniform' if sampling == 'uniform' else ''
    folder = output_dir / f'{task}{suffix}_ipc{config["ipc"]}'
    flat_result_directory(folder).mkdir(parents=True, exist_ok=True)
    summary_path = folder / 'summary.json'
    if not recompute and flat_result_path(summary_path).exists():
        saved = json.loads(flat_result_path(summary_path).read_text())
        if saved['config'] == config:
            return saved
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)
    x_train, labels, x_test, test_labels = data
    labels, test_labels = labels.astype(np.int64), test_labels.astype(np.int64)
    classes = int(labels.max()) + 1
    assert np.array_equal(np.unique(labels), np.arange(classes))
    class_counts = np.bincount(labels)
    available = len(labels) // classes if sampling == 'uniform' else int(class_counts.min())
    cap = min(available, config['cap'] or available)
    permutations = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        if sampling == 'uniform':
            permutations.append(rng.permutation(len(labels))[:cap*classes].reshape(classes, cap))
        else:
            permutations.append(np.stack([rng.permutation(np.flatnonzero(labels == c))[:cap] for c in range(classes)]))
    permutations = np.stack(permutations)
    flat_indices = permutations.reshape(-1)
    pool_x = tensor(x_train[flat_indices])
    x_train, x_test, truth = tensor(x_train), tensor(x_test), tensor(test_labels, torch.long)
    matched = np.full(len(targets), -1, dtype=int)
    matched_scores = np.full(len(targets), np.nan)
    curves, rows = [], []
    for rep, archive_path, archive in archives:
        tms = tm_ensemble(x_train.shape[1], 1024, archive['test_tm_seeds'], device)
        assert len(tms) == 50
        query_raw, pool_raw = _intensity(x_test, tms), _intensity(pool_x, tms)
        cache, phases, order, native = {}, {}, [], set()
        def local_indices(ipc, draw):
            if sampling == 'uniform':
                return np.arange(draw*classes*cap, draw*classes*cap+ipc*classes)
            return np.concatenate([np.arange(draw*classes*cap+c*cap, draw*classes*cap+c*cap+ipc) for c in range(classes)])
        def evaluate(ipc, phase='fine'):
            ipc = int(ipc)
            if ipc not in cache:
                values = []
                for draw in range(len(RANDOM_SEEDS)):
                    local = local_indices(ipc, draw)
                    indices = flat_indices[local]
                    assert len(np.unique(indices)) == classes * ipc
                    ys = F.one_hot(tensor(labels[indices], torch.long), classes).float()
                    values.append(_cached_scores(pool_raw[:, tensor(local, torch.long)], ys, query_raw, truth))
                cache[ipc] = np.stack(values)
                phases[ipc] = phase
                order.append(ipc)
            return float(cache[ipc].mean())
        def verify(ipc):
            if ipc in native:
                return
            values = []
            for draw in range(len(RANDOM_SEEDS)):
                indices = flat_indices[local_indices(ipc, draw)]
                xs = x_train[tensor(indices, torch.long)]
                ys = F.one_hot(tensor(labels[indices], torch.long), classes).float()
                if task == 'agnews':
                    values.append([evaluate_coreset_on_tm(xs, ys, x_test, truth, tm, BETA) for tm in tms])
                else:
                    values.append(evaluate_support(xs, ys, x_test, truth, tms, BETA)['accuracy_per_tm'])
            cache[ipc] = np.asarray(values, dtype=float)
            native.add(ipc)
        target_indices = np.flatnonzero(reps == rep)
        max_goal = float(targets[target_indices].max())
        evaluate(config['ipc'], 'equal_size')
        coarse_grid = list(range(config['coarse'], cap+1, config['coarse']))
        if not coarse_grid or coarse_grid[-1] != cap:
            coarse_grid.append(cap)
        upper = cap
        for budget in coarse_grid:
            if evaluate(budget, 'coarse') >= max_goal:
                upper = budget
                break
        fine_grid = list(range(config['fine'], cap+1, config['fine']))
        if fine_grid[-1] != cap:
            fine_grid.append(cap)
        for budget in fine_grid:
            if budget <= upper:
                evaluate(budget)
        for target_index in target_indices:
            goal = float(targets[target_index])
            # A few changed predictions can arise from different float32 GEMM
            # shapes; verify close lower points as well as the crossing itself.
            tolerance = 4 / (len(RANDOM_SEEDS) * len(tms) * len(truth))
            for position, budget in enumerate(fine_grid):
                value = evaluate(budget)
                if value >= goal - tolerance:
                    verify(budget)
                if evaluate(budget) >= goal:
                    if position:
                        verify(fine_grid[position-1])
                        if evaluate(fine_grid[position-1]) >= goal:
                            budget = fine_grid[position-1]
                    matched[target_index] = budget
                    matched_scores[target_index] = evaluate(budget)
                    break
            rows.append(dict(acquisition=str(acquisitions[target_index]), rep=int(rep),
                experimental_accuracy=goal, matched_random_ipc=int(matched[target_index]) if matched[target_index] > 0 else None,
                matched_random_accuracy=float(matched_scores[target_index]) if matched[target_index] > 0 else None,
                gain=float(matched[target_index] / config['ipc']) if matched[target_index] > 0 else None))
        ordered = sorted(cache)
        accuracies = np.stack([cache[k] for k in ordered])
        assert np.isfinite(accuracies).all()
        np.savez_compressed(flat_result_path(folder/f'rep{rep}_curves.npz'), candidate_ipc=np.asarray(ordered),
            candidate_accuracy_per_draw_tm=accuracies, evaluation_order=np.asarray(order),
            candidate_phase=np.asarray([phases[k] for k in ordered]),
            test_tm_seeds=archive['test_tm_seeds'])
        curves.append(dict(rep=int(rep), candidate_ipc=ordered,
            mean_accuracy=accuracies.mean((1, 2)).tolist(), evaluation_order=order,
            phase=[phases[k] for k in ordered]))
        print(f'{config["name"]}: rep {rep}/5, {len(ordered)} candidate budgets; '
              f'matched random/class = {matched[target_indices].tolist()}', flush=True)
        _save_json(folder/'progress.json', dict(completed_repetitions=int(rep), matches=rows,
            elapsed_seconds=time.monotonic()-started))
        del tms, query_raw, pool_raw
        torch.cuda.empty_cache()
    complete = bool((matched > 0).all())
    gains = matched / config['ipc']
    summary = dict(task=task, config=config,
        device=device, sampling=('3 fixed nested uniform permutations of the full training split, without replacement; class counts are unconstrained' if sampling == 'uniform' else '3 fixed nested class-balanced permutations, without replacement'),
        budget_axis=('average samples per class (total budget / class count)' if sampling == 'uniform' else 'samples per class'),
        random_seeds=list(RANDOM_SEEDS), ridge_lambda=BETA, tms_per_rep=50,
        fine_search='first crossing on all fine-grid points from the smallest positive budget; no interpolation',
        train_size=len(labels), test_size=len(test_labels), classes=classes,
        train_class_counts=class_counts.tolist(), random_cap_per_class=cap,
        coreset_total=config['ipc']*classes, measurement_count=len(targets),
        independent_coreset_repetitions=5, acquisition_rounds=2,
        experimental_accuracy_mean=float(targets.mean()), experimental_accuracy_sd=float(targets.std(ddof=1)),
        experimental_equal_random_accuracy_mean=float(equal_random.mean()),
        experimental_equal_random_accuracy_sd=float(equal_random.std(ddof=1)),
        matched_count=int((matched > 0).sum()), all_matched=complete,
        gain_geometric_mean=float(np.exp(np.log(gains).mean())) if complete else None,
        gain_min=float(gains.min()) if complete else None, gain_max=float(gains.max()) if complete else None,
        matched_random_ipc=matched.tolist(), gains=[float(g) if m > 0 else None for g,m in zip(gains,matched)],
        acquisitions=acquisitions.tolist(), source_reps=reps.tolist(), matches=rows,
        curves=curves, runtime_seconds=time.monotonic()-started,
        note='Simulated random versus measured coreset. SD and gain range describe 10 measurements of 5 coresets across 2 acquisitions; they are not confidence intervals.')
    pool_arrays = ({'random_pool_indices_by_draw':permutations.reshape(len(RANDOM_SEEDS),-1)} if sampling == 'uniform'
                   else {'random_pool_indices_by_draw_class':permutations})
    np.savez_compressed(flat_result_path(folder/'arrays.npz'), experimental_accuracy=targets,
        experimental_equal_random_accuracy=equal_random, matched_random_ipc=matched,
        matched_random_accuracy=matched_scores, gain=np.where(matched>0,gains,np.nan),
        source_reps=reps, source_acquisitions=acquisitions,
        random_seeds=np.asarray(RANDOM_SEEDS), **pool_arrays)
    _save_json(summary_path, summary)
    return summary


def run_classification_search(root, *, device='cuda:1', recompute=False, output_dir=None, cora_sampling='balanced'):
    root = Path(root).resolve()
    output_dir = Path(output_dir) if output_dir else root/'results/SI/sample_efficiency/classification'
    return {task: run_task(root, task, output_dir, device=device, recompute=recompute,
            sampling=cora_sampling if task=='cora' else 'balanced') for task in CONFIGS}


