"""Clements MZI mesh, fixed fabrication errors and a hybrid optical network.

Inputs are signed, unit-norm fields. Every mesh is followed by square-law
detection. Hidden intensities are power-normalized and remodulated as the next
field; the final intensity feeds a digital linear head. Only optical phases are
updated during calibration. Unpaired column modes bypass internal-arm loss.
No temperature drift is included in this experiment.

Clements et al., Optica 3, 1460-1465 (2016), doi:10.1364/OPTICA.3.001460.
"""
import math
import torch
import torch.nn.functional as F

CDT = torch.complex64
INV_SQRT2 = 1 / math.sqrt(2)
_IDX = {}
DEFAULT_ERRORS = dict(sph=0.05, sbs=0.03, loss_db=0.1, loss_sd=0.02)

class ClementsMesh:
    """Rectangular (Clements) mesh: N columns alternating even/odd nearest-neighbour pairs."""

    def __init__(self, N, depth=None):
        if N < 2 or N % 2:
            raise ValueError("This rectangular mesh requires an even mode count >= 2.")
        self.N = N
        self.cols = []
        off = 0
        for c in range(depth or N):
            s = c % 2
            npair = (N - s) // 2
            self.cols.append((s, npair, off))
            off += npair
        self.n = off


def mzi_transfer_matrix(th, ph, err=None):
    """All MZI 2x2 transfer matrices at once: (..., n, 2, 2) complex.
    T = B(a2) diag(ea e^{i th}, eb) B(a1) diag(e^{i ph}, 1),  B(a) = [[cos, i sin], [i sin, cos]]."""
    if err is None:
        c1 = s1 = c2 = s2 = INV_SQRT2
        et = torch.polar(torch.ones_like(th), th)
        eb = 1.0
    else:
        th, ph = th + err['dth'], ph + err['dph']
        c1, s1, c2, s2 = err['c1'], err['s1'], err['c2'], err['s2']
        et = torch.polar(err['ea'].expand_as(th), th)
        eb = err['eb']
    ep = torch.polar(torch.ones_like(ph), ph)
    d00 = et * (c1 * ep)
    d01 = et * (1j * s1)
    d10 = eb * (1j * s1) * ep
    d11 = (eb * c1) * torch.ones_like(ep)
    T00 = c2 * d00 + 1j * s2 * d10
    T01 = c2 * d01 + 1j * s2 * d11
    T10 = 1j * s2 * d00 + c2 * d10
    T11 = 1j * s2 * d01 + c2 * d11
    return torch.stack((torch.stack((T00, T01), -1), torch.stack((T10, T11), -1)), -2)


def _col_index(geo, device):
    """Flat scatter indices of every MZI's 4 entries into the stacked column matrices
    (ncol, N, N), row-vector convention (x_out = x_in @ M, so M[i, j] = T[j, i])."""
    key = (geo.N, len(geo.cols), str(device))
    if key in _IDX:
        return _IDX[key]
    N = geo.N
    idx, eye = [], torch.zeros(len(geo.cols), N, N)
    for c, (s, npair, off) in enumerate(geo.cols):
        for p in range(npair):
            i = s + 2 * p
            base = c * N * N
            # order matches T.flatten(-2): T00, T01, T10, T11  ->  M[i,i], M[i+1,i], M[i,i+1], M[i+1,i+1]
            idx += [base + i * N + i, base + (i + 1) * N + i, base + i * N + i + 1, base + (i + 1) * N + i + 1]
        paired = set(range(s, s + 2 * npair))
        for m in range(N):
            if m not in paired:
                eye[c, m, m] = 1
    out = (torch.tensor(idx, device=device), eye.to(device))
    _IDX[key] = out
    return out


def mesh_matrix(th, ph, geo, err=None):
    """Mesh transfer matrix (..., N, N), row-vector convention, via a log-depth product of
    dense block-diagonal column matrices."""
    T = mzi_transfer_matrix(th, ph, err)                                     # (..., n, 2, 2)
    idx, eye = _col_index(geo, T.device)
    ncol, N = len(geo.cols), geo.N
    Mflat = torch.zeros(*T.shape[:-3], ncol * N * N, dtype=T.dtype, device=T.device)
    Mflat = Mflat.index_copy(-1, idx, T.flatten(-3))
    M = Mflat.unflatten(-1, (ncol, N, N)) + eye
    while M.shape[-3] > 1:
        if M.shape[-3] % 2:
            tail = M[..., -1:, :, :]
            M = torch.cat([M[..., 0:-1:2, :, :] @ M[..., 1::2, :, :], tail], -3)
        else:
            M = M[..., 0::2, :, :] @ M[..., 1::2, :, :]
    return M[..., 0, :, :]


def apply_mesh(x, th, ph, geo, err=None):
    """x (..., B, N) complex; th, ph (..., n) real (leading dims broadcast with x's).
    err: dict with dth, dph, c1, s1, c2, s2, ea, eb each (..., n), or None (ideal twin)."""
    return x @ mesh_matrix(th, ph, geo, err)


def sample_chip(seed, L, geo, sph, sbs, loss_db=0.1, loss_sd=0.02, device='cuda', dtype=torch.float32):
    g = torch.Generator().manual_seed(int(seed))
    n = geo.n
    r = lambda: torch.randn(L, n, generator=g, dtype=torch.float64)
    dth, dph, a1, a2 = r() * sph, r() * sph, r() * sbs, r() * sbs
    la, lb = loss_db + loss_sd * r(), loss_db + loss_sd * r()
    chip = dict(dth=dth, dph=dph,
                c1=torch.cos(math.pi / 4 + a1), s1=torch.sin(math.pi / 4 + a1),
                c2=torch.cos(math.pi / 4 + a2), s2=torch.sin(math.pi / 4 + a2),
                ea=10 ** (-la / 20), eb=10 ** (-lb / 20))
    return {k: v.to(device, dtype) for k, v in chip.items()}


def stack_chips(chips):
    return {k: torch.stack([c[k] for c in chips]) for k in chips[0]}


def layer_err(chip, l):
    return None if chip is None else {k: v[..., l, :] for k, v in chip.items()}


def make_chips(seeds, network, device, errors=None):
    """Stack independent error instances along the leading chip dimension."""
    parameters = dict(DEFAULT_ERRORS if errors is None else errors)
    return stack_chips([sample_chip(seed, network.L, network.geo,
                                   device=device, **parameters) for seed in seeds])

class MZINetwork:
    """Holds geometry and frozen electronic parameters.  Optical state = (th, ph) of shape
    (..., L, n).  Readout state = (W (...,10,N), b (...,10))."""

    def __init__(self, N=64, L=2):
        self.N, self.L = N, L
        self.geo = ClementsMesh(N)

    def elec(self, intensity, layer):
        """Power normalization, intensity-to-amplitude remodulation and L2 scaling."""
        intensity = intensity / intensity.sum(-1, keepdim=True).clamp_min(1e-12)
        return F.normalize(self.N * intensity, dim=-1).to(CDT)

    def head(self, I, W, b):
        In = self.N * I / I.sum(-1, keepdim=True).clamp_min(1e-12)
        return In @ W.transpose(-1, -2) + b[..., None, :]

    def forward(self, st, z, chip=None, noise=0.0, keep=False):
        """st: dict th, ph, W, b.  z (..., B, N) real/complex.  Returns logits (and layer
        inputs / intensities if keep)."""
        h = z.to(CDT)
        ins, Is = [], []
        for l in range(self.L):
            ins.append(h)
            o = apply_mesh(h, st['th'][..., l, :], st['ph'][..., l, :], self.geo, layer_err(chip, l))
            I = o.real ** 2 + o.imag ** 2
            if noise:
                I = I + noise * I.mean(-1, keepdim=True) * torch.randn_like(I)
            Is.append(I)
            if l < self.L - 1:
                h = self.elec(I, l)
        logits = self.head(Is[-1], st['W'], st['b'])
        return (logits, ins, Is) if keep else logits
