"""Layer-wise physics-aware training (PAT) and differentiable online updates.

PAT uses erroneous-chip forward values and the ideal mesh's local Jacobian at
the actual layer input. The true hardware Jacobian is not the online gradient.
create_graph=True retains the full simulator graph for offline meta-learning.
The oracle exact_grad is provided only for numerical comparisons.

Reference: Wright et al., Nature 601, 549-555 (2022),
https://doi.org/10.1038/s41586-021-04223-6
"""
import torch
from .mzi_network import apply_mesh

def pat_grad(net, st, z, target, chip, noise=0.0, create_graph=False, train_readout=False):
    """Layer-wise physics-aware training gradient.
    Forward values: actual chip (with detector noise).  Optical backward: twin Jacobian of each
    mesh evaluated at the *measured* input of that mesh.  Electronics: exact Jacobians.
    Returns dict of grads for th, ph (and W, b if train_readout) and the actual logits."""
    th, ph = st['th'], st['ph']
    logits, ins, Is = net.forward(st, z, chip, noise, keep=True)
    loss = per_chip_ce(logits, target)
    wrt = [Is[-1]] + ([st['W'], st['b']] if train_readout else [])
    g = torch.autograd.grad(loss, wrt, create_graph=create_graph)
    gI = g[0]
    gth, gph = [None] * net.L, [None] * net.L
    for l in reversed(range(net.L)):
        tl, pl = th[..., l, :], ph[..., l, :]
        o = apply_mesh(ins[l], tl, pl, net.geo, None)
        Inom = o.real ** 2 + o.imag ** 2
        if l > 0:
            gt, gp, gh = torch.autograd.grad(Inom, [tl, pl, ins[l]], gI,
                                             create_graph=create_graph, retain_graph=True)
            gI = torch.autograd.grad(ins[l], Is[l - 1], gh, create_graph=create_graph,
                                     retain_graph=True)[0]
        else:
            gt, gp = torch.autograd.grad(Inom, [tl, pl], gI, create_graph=create_graph,
                                         retain_graph=True)
        gth[l], gph[l] = gt, gp
    out = dict(th=torch.stack(gth, -2), ph=torch.stack(gph, -2))
    if train_readout:
        out['W'], out['b'] = g[1], g[2]
    return out, logits


def exact_grad(net, st, z, target, chip, noise=0.0, create_graph=False, keys=('th', 'ph')):
    """Oracle: true gradient through the actual chip."""
    logits = net.forward(st, z, chip, noise)
    loss = per_chip_ce(logits, target)
    g = torch.autograd.grad(loss, [st[k] for k in keys], create_graph=create_graph)
    return dict(zip(keys, g)), logits


def adapt(net, st0, data, chips, K, lr, keys=('th', 'ph'), learner='pat', opt='adam',
          noise=0.0, create_graph=False, callback=None, betas=(0.9, 0.999), precond=None,
          momentum_schedule=None):
    """Run K in situ steps.  data(k) -> (z, target) with z (C or 1, B, N).
    lr: scalar or tensor (K, L, 1), broadcast over chips and MZIs.  Returns final state."""
    if learner not in ('pat', 'exact') or opt not in ('adam', 'sgd'):
        raise ValueError("Choose learner='pat'/'exact' and opt='adam'/'sgd'.")
    st = dict(st0)
    m = {k: torch.zeros_like(st[k]) for k in keys}
    v = {k: torch.zeros_like(st[k]) for k in keys}
    b1, b2 = betas
    for k in range(K):
        z, tgt = data(k)
        if learner == 'pat':
            tr = any(x in keys for x in ('W', 'b'))
            g, _ = pat_grad(net, st, z, tgt, chips, noise, create_graph, train_readout=tr)
        else:
            g, _ = exact_grad(net, st, z, tgt, chips, noise, create_graph, keys)
        eta = lr[k] if torch.is_tensor(lr) and lr.dim() else lr
        mu = momentum_schedule[k] if momentum_schedule is not None else b1
        new = dict(st)
        for key in keys:
            if opt == 'adam':
                m[key] = b1 * m[key] + (1 - b1) * g[key]
                v[key] = b2 * v[key] + (1 - b2) * g[key] ** 2
                mh = m[key] / (1 - b1 ** (k + 1))
                vh = v[key] / (1 - b2 ** (k + 1))
                upd = mh / ((vh + 1e-16).sqrt() + 1e-8)
            else:  # sgd with momentum
                m[key] = mu * m[key] + g[key]
                upd = m[key]
            if precond is not None and key in precond:
                p = precond[key]
                if precond.get('_perstep'):                    # (K', ...) one map per step (last reused)
                    p = p[min(k, len(p) - 1)]
                upd = p * upd                                  # learned per-parameter lr multiplier
            new[key] = st[key] - eta * upd
            if not create_graph:
                new[key] = new[key].detach().requires_grad_(True)
                m[key], v[key] = m[key].detach(), v[key].detach()
        st = new
        if callback is not None:
            callback(k + 1, st)
    return st


def per_chip_ce(logits, target):
    """Mean over batch, summed over leading chip dims (so each chip's grads are independent)."""
    if target.dtype == torch.long:
        target = target.expand(logits.shape[:-1])
        lp = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
        return -lp.mean(-1).sum()
    return -(target * logits.log_softmax(-1)).sum(-1).mean(-1).sum()


def expand_state(src, C):
    return {k: v.detach()[None].expand(C, *v.shape).clone().requires_grad_(True)
            for k, v in src.items()}
