"""Ideal-source training and unrolled coreset/update-rule optimization.

Only training images enter gradient updates. Source/DD checkpoints are selected
using the validation images; DD selection also uses separate validation chips.
Calls always perform training, even when output files already exist.
"""
from .datasets import flat_result_path
from pathlib import Path
import math
import time
import torch
import torch.nn.functional as F
from .datasets import save_checkpoint, save_json
from .preprocessing import PCAEncoder
from .mzi_network import MZINetwork, make_chips
from .pat import adapt, expand_state, per_chip_ce


def train_source(dataset, config, output_path, progress=print):
    """Train the fixed paper source recipe (60 epochs, PCA power 0.25)."""
    (x_train, y_train), (x_val, y_val), _ = dataset
    device = x_train.device
    cfg = config['source']
    torch.manual_seed(cfg['seed'])
    encoder = PCAEncoder(x_train, config['modes'], cfg['power'])
    network = MZINetwork(config['modes'], config['layers'])
    train_fields, val_fields = encoder.encode(x_train), encoder.encode(x_val)
    generator = torch.Generator().manual_seed(cfg['seed'])
    state = dict(th=(torch.rand(network.L, network.geo.n, generator=generator) * 2 * math.pi).to(device),
                 ph=(torch.rand(network.L, network.geo.n, generator=generator) * 2 * math.pi).to(device),
                 W=(torch.randn(10, network.N, generator=generator) * .1).to(device),
                 b=torch.zeros(10, device=device))
    for value in state.values():
        value.requires_grad_()
    optimizer = torch.optim.Adam(state.values(), lr=cfg['lr'])
    steps = len(train_fields) // cfg['batch']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg['lr'], total_steps=cfg['epochs'] * steps, pct_start=.1)
    best, rows, start = -1., [], time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        permutation = torch.randperm(len(train_fields), device=device)
        for step in range(steps):
            indices = permutation[step * cfg['batch']:(step + 1) * cfg['batch']]
            loss = F.cross_entropy(network.forward(state, train_fields[indices]), y_train[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        value = float(accuracy(network, state, val_fields, y_val))
        row = dict(epoch=epoch, val=value, sec=time.monotonic() - start)
        rows.append(row)
        if value > best:
            best = value
            checkpoint = dict(st=state, act='none', inp='uncent', ref=None,
                              N=network.N, L=network.L, power=cfg['power'], bias=0.,
                              seed=cfg['seed'], val=value, best_epoch=epoch,
                              dataset='FashionMNIST', encoder=vars(encoder),
                              source_config=cfg, source_epochs=cfg['epochs'], train_batch=cfg['batch'])
            save_checkpoint(output_path, checkpoint)
        save_json(Path(output_path).with_suffix('.history.json'), dict(best_val=best, rows=rows))
        if progress:
            progress(f'Source epoch {epoch}/{cfg["epochs"]}: validation={100 * value:.4f}%')
    return flat_result_path(output_path)


def distill_coreset(dataset, source_path, config, output_path, progress=print, *, stop_after=None):
    """Learn 80 inputs, signed targets and an eight-step PAT update rule.

    Per-step layer learning rates and per-phase multipliers use exp; momentum
    is 0.99*sigmoid(raw). Class coefficients are unconstrained, not probabilities.
    Each outer step samples four training chips and 1,024 query images per chip.
    stop_after truncates the original trajectory without changing its cosine
    schedule horizon, which remains config['distillation']['outer']. The last
    checkpoint stores the exact stopping iteration, independently of selection.
    Optional input_constraint='nonnegative_field' optimizes 64 pre-normalized
    field amplitudes in [0, 1], projected initially and after each Adam update.
    Their L2-normalized fields are also in [0, 1], with zero input phase.
    Compatible checkpoint u = amplitudes - off preserves the shared evaluator.
    """
    (x_train, y_train), (x_val, y_val), _ = dataset
    device, cfg = x_train.device, config['distillation']
    input_constraint = cfg.get('input_constraint', 'none')
    if input_constraint not in ('none', 'nonnegative_field'):
        raise ValueError('input_constraint must be none or nonnegative_field.')
    stop_iteration = cfg['outer'] if stop_after is None else stop_after
    if not isinstance(stop_iteration, int) or not 1 <= stop_iteration <= cfg['outer']:
        raise ValueError('stop_after must be an integer between 1 and the configured outer iterations.')
    torch.manual_seed(cfg['seed'])
    source, encoder, network = load_source(source_path, device)
    M, B, K, C = (cfg[key] for key in ('M', 'B', 'K', 'chips_per_step'))
    if M != B * K or M % 10:
        raise ValueError('The paper support uses one pass through K class-interleaved batches.')
    fields_train, fields_val = encoder.encode(x_train), encoder.encode(x_val)
    indices = balanced_indices(y_train, M // 10, cfg['seed']).view(10, -1).T.flatten()
    inputs = encoder.u(x_train)[indices].detach().clone().requires_grad_()
    if input_constraint == 'nonnegative_field':
        with torch.no_grad():
            inputs.add_(encoder.off)
            inputs.clamp_(0., 1.)
    targets = F.one_hot(y_train[indices], 10).float().requires_grad_()
    log_lr = torch.full((K, network.L, 1), math.log(cfg['inner_lr']), device=device, requires_grad=True)
    log_theta = torch.zeros(K, network.L, network.geo.n, device=device, requires_grad=True)
    log_phi = torch.zeros_like(log_theta, requires_grad=True)
    raw_momentum = torch.full((K,), math.log(.5 / (.99 - .5)), device=device, requires_grad=True)
    groups = [dict(params=[inputs, targets], lr=cfg['data_lr']),
              dict(params=[log_lr, log_theta, log_phi], lr=cfg['rule_lr']),
              dict(params=[raw_momentum], lr=cfg['momentum_lr'])]
    parameters = [value for group in groups for value in group['params']]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['outer'])
    chip_rng = torch.Generator().manual_seed(cfg['seed'] + 100)
    query_rng = torch.Generator(device=device).manual_seed(cfg['seed'] + 200)
    rows, best, start = [], -1., time.monotonic()
    first_seed, last_seed = config['training_chip_pool']

    def coefficients():
        return inputs - encoder.off if input_constraint == 'nonnegative_field' else inputs

    def multipliers(detach=False):
        values = dict(th=log_theta.exp(), ph=log_phi.exp(), _perstep=True)
        return {k: v.detach() if detach and torch.is_tensor(v) else v for k, v in values.items()}

    def checkpoint(iteration):
        result = dict(dataset='FashionMNIST', u=coefficients(), t=targets, lr=log_lr.exp(),
                    pre=multipliers(), momentum_schedule=.99 * raw_momentum.sigmoid(),
                    M=M, B=B, K=K, opt='sgd', ids=indices, rows=list(rows), best_dev=best,
                    completed_outer=iteration, done=iteration == cfg['outer'],
                    input_constraint=input_constraint,
                    elapsed=time.monotonic() - start, config=cfg,
                    objective='Training-query CE after unrolled PAT updates; jointly learned inputs and update rule.')
        if input_constraint == 'nonnegative_field':
            result['field_amplitudes'] = inputs
        return result

    def validate():
        values = []
        seeds = config['validation_chips']
        for offset in range(0, len(seeds), config['chip_batch']):
            subset = seeds[offset:offset + config['chip_batch']]
            hardware = make_chips(subset, network, device, config['errors'])
            state = adapt(network, expand_state(source['st'], len(subset)),
                          cyclic_batches(encoder.field(coefficients().detach()), targets.detach(), B, len(subset)),
                          hardware, K, log_lr.exp().detach(), opt='sgd', precond=multipliers(True),
                          momentum_schedule=(.99 * raw_momentum.sigmoid()).detach())
            values.extend(accuracy(network, state, fields_val, y_val, hardware).tolist())
        return values

    for iteration in range(stop_iteration + 1):
        if iteration % cfg['validation_interval'] == 0 or iteration == stop_iteration:
            values = validate()
            score = sum(values) / len(values)
            if not math.isfinite(score):
                raise FloatingPointError('Non-finite validation score.')
            rows.append(dict(it=iteration, dev=score, per_chip=values, sec=time.monotonic() - start))
            if score > best:
                best = score
                save_checkpoint(output_path, checkpoint(iteration))
            save_json(Path(output_path).with_suffix('.history.json'), dict(best_dev=best, rows=rows))
            if progress:
                progress(f'Distillation {iteration}/{cfg["outer"]}: validation={100 * score:.4f}%, best={100 * best:.4f}%')
        if iteration == stop_iteration:
            break
        seeds = (first_seed + torch.randint(last_seed - first_seed + 1, (C,), generator=chip_rng)).tolist()
        hardware = make_chips(seeds, network, device, config['errors'])
        state = adapt(network, expand_state(source['st'], C),
                      cyclic_batches(encoder.field(coefficients()), targets, B, C), hardware, K,
                      log_lr.exp(), opt='sgd', precond=multipliers(),
                      momentum_schedule=.99 * raw_momentum.sigmoid(), create_graph=True)
        query = torch.randint(len(fields_train), (C, cfg['query_batch']), generator=query_rng, device=device)
        loss = per_chip_ce(network.forward(state, fields_train[query], hardware), y_train[query]) / C
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training query loss.')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if input_constraint == 'nonnegative_field':
            with torch.no_grad():
                inputs.clamp_(0., 1.)
        scheduler.step()
    save_checkpoint(Path(output_path).with_name(Path(output_path).stem + '_last.pt'), checkpoint(stop_iteration))
    return flat_result_path(output_path)


def load_source(path, device):
    from .mzi_network import MZINetwork
    checkpoint = torch.load(flat_result_path(path), map_location=device, weights_only=True)
    if checkpoint['dataset'] != 'FashionMNIST' or checkpoint['act'] != 'none':
        raise ValueError('Expected the PD-only Fashion-MNIST source model.')
    if checkpoint.get('ref') is not None:
        raise ValueError('The published model has no dedicated reference mode.')
    encoder = PCAEncoder.from_state(checkpoint['encoder'], device)
    return checkpoint, encoder, MZINetwork(checkpoint['N'], checkpoint['L'])


def balanced_indices(labels, per_class, seed):
    """Class-major indices; transpose a (10, IPC) view for class-interleaved batches."""
    generator = torch.Generator().manual_seed(int(seed))
    indices = []
    for label in range(10):
        candidates = torch.where(labels.cpu() == label)[0]
        indices.append(candidates[torch.randperm(len(candidates), generator=generator)[:per_class]])
    return torch.cat(indices).to(labels.device)


def cyclic_batches(fields, targets, batch_size, chips):
    """Consume an ordered fixed support in consecutive batches, cycling if needed."""
    if len(fields) % batch_size:
        raise ValueError('Support size must be divisible by the calibration batch size.')
    groups = len(fields) // batch_size
    def batch(step):
        start = (step % groups) * batch_size
        z = fields[start:start + batch_size]
        return z[None].expand(chips, *z.shape), targets[start:start + batch_size]
    return batch


def random_batches(fields, targets, batch_size, seed, chips):
    """A continuous device-local sampling stream, shared across a chip batch."""
    generator = torch.Generator(device=fields.device).manual_seed(seed)
    def batch(step):
        indices = torch.randint(len(fields), (batch_size,), device=fields.device, generator=generator)
        return fields[indices][None].expand(chips, batch_size, fields.shape[-1]), targets[indices]
    return batch


@torch.no_grad()
def accuracy(net, st, x_field, y, chip=None, noise=0.0, batch=2500):
    """Accuracy per chip (returns tensor (C,) if state is chip-batched, else scalar)."""
    good = 0
    for bz, by in zip(x_field.split(batch), y.split(batch)):
        lg = net.forward(st, bz, chip, noise)
        good = good + (lg.argmax(-1) == by).float().sum(-1)
    return good / len(y)


def _curve_checks(steps):
    return list(range(1, min(steps, 8) + 1)) + [k for k in (16, 32, 64, 100, 128) if 8 < k <= steps]


def _baseline_lr(recipe, steps, device):
    rate = recipe['lr']
    if recipe.get('schedule') == 'step4':
        return torch.tensor([rate * (recipe['factor'] if k >= 4 else 1.) for k in range(steps)], device=device)
    if recipe.get('schedule') == 'cosine':
        return torch.tensor([rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * k / (steps - 1))))
                             for k in range(steps)], device=device)
    if recipe.get('decay', 1.) != 1.:
        return torch.tensor([rate * recipe['decay'] ** int(k >= 32) * recipe['decay'] ** int(k >= 64)
                             for k in range(steps)], device=device)
    return rate


def _percent_record(values, curves=None, **metadata):
    """Convert fraction accuracies to percent; SD is across chip averages."""
    import statistics
    percent = [100 * value for value in values]
    return dict(mean=statistics.mean(percent), sd=statistics.stdev(percent) if len(percent) > 1 else 0.,
                min=min(percent), max=max(percent), per_chip=percent,
                curves={str(step): [100 * v for v in row] for step, row in (curves or {}).items()},
                **metadata)


def evaluate_main(dataset, source_path, coreset_path, config, output_path, progress=print):
    """Rerun all fixed-recipe comparisons on the same unseen simulated chips.

    Each chip sees all 10,000 test images. Baselines average three independent
    calibration runs within chip before computing the across-chip mean and SD.
    The electronic head remains fixed during adaptation.
    Hyperparameters are read from the original validation-selected config.
    """
    import statistics
    (x_train, y_train), _, (x_test, y_test) = dataset
    device = x_train.device
    source, encoder, network = load_source(source_path, device)
    coreset = torch.load(flat_result_path(coreset_path), map_location=device, weights_only=True)
    seeds, group_size = config['test_chips'], config['chip_batch']
    train_low, train_high = config['training_chip_pool']
    if set(seeds) & set(config['validation_chips']) or any(train_low <= s <= train_high for s in seeds):
        raise ValueError('Training, validation and test chip seeds must be disjoint.')
    z_train, z_test = encoder.encode(x_train), encoder.encode(x_test)
    output = dict(origin='fresh_evaluation', config=config,
                  test_image_count=len(y_test), static={})
    start = time.monotonic()

    def save(name, values, curves=None, **metadata):
        output['static'][name] = _percent_record(values, curves, **metadata)
        output['elapsed_seconds'] = time.monotonic() - start
        save_json(output_path, output)
        if progress:
            row = output['static'][name]
            progress(f'{name}: {row["mean"]:.5f}% +/- {row["sd"]:.5f} (chip SD)')

    ideal = float(accuracy(network, source['st'], z_test, y_test))
    save('ideal', [ideal] * len(seeds), B=0, K=0, presentations=0)
    direct = []
    for offset in range(0, len(seeds), group_size):
        subset = seeds[offset:offset + group_size]
        hardware = make_chips(subset, network, device, config['errors'])
        direct.extend(accuracy(network, source['st'], z_test, y_test, hardware).tolist())
    save('direct_load', direct, B=0, K=0, presentations=0)

    values, curves = [], {'0': direct}
    for offset in range(0, len(seeds), group_size):
        subset = seeds[offset:offset + group_size]
        hardware = make_chips(subset, network, device, config['errors'])
        state = expand_state(source['st'], len(subset))
        def record(step, current):
            curves.setdefault(str(step), []).extend(accuracy(network, current, z_test, y_test, hardware).tolist())
        final = adapt(network, state,
                      cyclic_batches(encoder.field(coreset['u']), coreset['t'], coreset['B'], len(subset)),
                      hardware, coreset['K'], coreset['lr'] * config['dd_lr_scale'],
                      opt='sgd', precond=coreset['pre'], momentum_schedule=coreset['momentum_schedule'],
                      callback=record)
        values.extend(accuracy(network, final, z_test, y_test, hardware).tolist())
    save('dd80', values, curves, M=coreset['M'], B=coreset['B'], K=coreset['K'],
         presentations=coreset['B'] * coreset['K'],
         mesh_intensity_vector_measurements=network.L * coreset['B'] * coreset['K'])

    for name, specification in config['baselines'].items():
        batch, steps, recipe = specification['B'], specification['K'], specification['recipe']
        runs = []
        for run_seed in specification['seeds']:
            run_values, run_curves = [], {}
            indices = balanced_indices(y_train, 8, run_seed).view(10, -1).T.flatten()
            for offset in range(0, len(seeds), group_size):
                subset = seeds[offset:offset + group_size]
                hardware = make_chips(subset, network, device, config['errors'])
                state = expand_state(source['st'], len(subset))
                data_fn = (cyclic_batches(z_train[indices], y_train[indices], batch, len(subset))
                           if specification['kind'] == 'fixed' else
                           random_batches(z_train, y_train, batch, run_seed, len(subset)))
                checks = _curve_checks(steps)
                def record(step, current):
                    if step in checks:
                        run_curves.setdefault(str(step), []).extend(
                            accuracy(network, current, z_test, y_test, hardware).tolist())
                final = adapt(network, state, data_fn, hardware, steps,
                              _baseline_lr(recipe, steps, device), opt=recipe['opt'],
                              betas=(recipe['momentum'], .999), callback=record)
                run_values.extend(accuracy(network, final, z_test, y_test, hardware).tolist())
            runs.append(dict(seed=run_seed, values=run_values, curves=run_curves))
        values = [statistics.mean(v) for v in zip(*(r['values'] for r in runs))]
        curves = {step: [statistics.mean(v) for v in zip(*(r['curves'][step] for r in runs))]
                  for step in runs[0]['curves']}
        curves['0'] = direct
        save(name, values, curves, B=batch, K=steps, presentations=batch * steps,
             mesh_intensity_vector_measurements=network.L * batch * steps,
             recipe=recipe, raw_runs=runs,
             aggregation='Mean of independent calibration runs within chip, then across chips; no ensemble deployment.')
    save_json(output_path, output)
    return output


def evaluate_random_extension(dataset, source_path, config, main_result, output_path, progress=print):
    """One continuous frozen Fresh Random trajectory through the recorded budget.

    Adam moments and the sampling RNG are never reset at intermediate checks.
    The stored observation grid is descriptive; no test-based learning-rate
    selection or interpolation is performed. Test evaluation costs are excluded
    from the online input budget, just as in the main comparison.
    """
    import statistics
    (x_train, y_train), _, (x_test, y_test) = dataset
    device = x_train.device
    source, encoder, network = load_source(source_path, device)
    z_train, z_test = encoder.encode(x_train), encoder.encode(x_test)
    spec = config['baselines']['fresh10_long']
    recipe, seeds = spec['recipe'], config['test_chips']
    checks = set(config['extended_checks'])
    steps, group_size = max(checks), config['chip_batch']
    output = dict(origin='fresh_evaluation', target_accuracy_percent=main_result['static']['dd80']['mean'],
                  DD_input_presentations=main_result['static']['dd80']['presentations'],
                  recipe=recipe, test_chip_seeds=seeds,
                  stream_seeds=spec['seeds'], test_images=len(y_test), curves={}, raw={},
                  checkpoint_interpretation='First recorded crossing, conditional on the frozen recipe and observation grid.')
    start = time.monotonic()
    for run_seed in spec['seeds']:
        output['raw'][str(run_seed)] = {}
        for offset in range(0, len(seeds), group_size):
            subset = seeds[offset:offset + group_size]
            hardware = make_chips(subset, network, device, config['errors'])
            state = expand_state(source['st'], len(subset))
            def record(step, current):
                if step in checks:
                    values = (100 * accuracy(network, current, z_test, y_test, hardware)).tolist()
                    output['raw'][str(run_seed)].setdefault(str(step), []).extend(values)
            final = adapt(network, state,
                          random_batches(z_train, y_train, spec['B'], run_seed, len(subset)),
                          hardware, steps, recipe['lr'], opt=recipe['opt'],
                          betas=(recipe['momentum'], .999), callback=record)
            if progress:
                progress(f'Random extension: seed={run_seed}, chips={offset + len(subset)}/{len(seeds)}, steps={steps}')
        save_json(Path(output_path).with_suffix('.partial.json'), output)
    for step in sorted(checks):
        values = [statistics.mean(v) for v in zip(*(output['raw'][str(seed)][str(step)] for seed in spec['seeds']))]
        output['curves'][str(step)] = dict(mean=statistics.mean(values),
                                         sd=statistics.stdev(values) if len(values) > 1 else 0.,
                                         per_chip=values, presentations=spec['B'] * step,
                                         mesh_vector_measurements=network.L * spec['B'] * step)
    hit = next((k for k in sorted(checks) if output['curves'][str(k)]['mean'] >= output['target_accuracy_percent']), None)
    output['threshold'] = dict(first_recorded_step=hit,
                              input_ratio_at_first_recorded_hit=None if hit is None else
                              spec['B'] * hit / output['DD_input_presentations'])
    output['elapsed_seconds'] = time.monotonic() - start
    save_json(output_path, output)
    return output
