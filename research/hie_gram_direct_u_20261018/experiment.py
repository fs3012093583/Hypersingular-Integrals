"""Fresh experiment 1: smooth raw-u first-kind HIE, four matched methods."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from torch import nn
from torch.nn import functional as F

import rhs_data as data

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('pinn', 'mc', 'equal', 'gram')
SEEDS = (20261030, 20261031, 20261032)
LBFGS = dict(lr=1., history_size=100, tolerance_grad=1e-11,
             tolerance_change=1e-13, line_search_fn='strong_wolfe')


@lru_cache(maxsize=16)
def legendre_rule(q):
    return leggauss(q)


@dataclass(frozen=True)
class Config:
    width: int = 128
    layers: int = 3
    activation: str = 'tanh'
    q: int = 192
    modes: int = 64
    mc_samples: int = 50
    adam_steps: int = 1000
    lr: float = .002
    lbfgs_steps: int = 300
    near: float = .05
    near_degree: int = 8
    device: str = 'cpu'
    log_every: int = 100


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_receipt():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.iterdir())
            if p.suffix in ('.py', '.md')}


def data_receipt(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).iterdir()) if p.suffix in ('.npz', '.json')}


def state_hash(model):
    return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.parameters())).hexdigest()


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def gradient(loss, model):
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])


class Net(nn.Module):
    def __init__(self, cfg, seed):
        super().__init__()
        torch.manual_seed(seed)
        dims = [1]+[cfg.width]*cfg.layers+[1]
        self.layers = nn.ModuleList([nn.Linear(a, b, dtype=torch.float64) for a, b in zip(dims[:-1], dims[1:])])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)
        self.to(cfg.device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

    def jet(self, x, order):
        """Exact Taylor-mode AD, output column j is N^(j)/j!."""
        c = ([x, torch.ones_like(x)]+[torch.zeros_like(x)]*max(0, order-1))[:order+1]
        for index, layer in enumerate(self.layers):
            z = [layer(c[0])]+[F.linear(v, layer.weight) for v in c[1:]]
            if index == len(self.layers)-1:
                return torch.stack(z, dim=-1).squeeze(1)
            y, q = [torch.tanh(z[0])], [1-torch.tanh(z[0]).square()]
            for n in range(1, order+1):
                y.append(sum(k*z[k]*q[n-k] for k in range(1, n+1))/n)
                if n < order:
                    q.append(-sum(y[j]*y[n-j] for j in range(n+1)))
            c = y
        raise AssertionError('no output layer')


class Problem:
    def __init__(self, m, family, cfg, data_dir):
        start = time.perf_counter()
        self.m, self.family, self.cfg = m, family, cfg
        self.cm = (-1)**(m//2)*math.pi
        with np.load(Path(data_dir)/f'm{m}_{family}.npz') as saved:
            self.x_np = saved[f'x{cfg.q}'].copy()
            self.w_np = saved[f'w{cfg.q}'].copy()
            self.f_np = saved[f'f{cfg.q}'].copy()
            # No reading of verification_only_b in the training implementation.
            self.b_np = saved['b'].copy()
        phi = data.basis(self.x_np, cfg.modes, m//2)
        self.lam = np.array([math.comb(k+m-1, m-1) for k in range(cfg.modes)], float)
        self.moments_np = phi.T*self.w_np[None, :]
        gx, gw = legendre_rule(192)
        bp = data.basis(gx, cfg.modes, m//2)
        T = bp.T@(gw[:, None]*bp)
        scale = np.sqrt(np.diag(T))
        L = np.linalg.cholesky(T/scale[:, None]/scale[None, :])
        B = np.linalg.solve(L, np.diag(1/scale))
        self.T_np, self.whitening_np = T, B
        self.whitening_defect = float(np.linalg.norm(B@T@B.T-np.eye(cfg.modes)))
        assert self.whitening_defect < 1e-7
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=cfg.device)
        self.x = tensor(self.x_np[:, None])
        self.w, self.f = tensor(self.w_np), tensor(self.f_np)
        self.point_scale = 1+self.f.abs()
        self.equal_map = tensor(self.cm*self.lam[:, None]*self.moments_np)
        self.equal_target = tensor(self.b_np)
        self.gram_map = tensor(B@self.moments_np)
        self.gram_target = tensor(B@(self.b_np/(self.cm*self.lam)))
        self.equal_den = float(self.b_np@self.b_np)
        self.gram_den = float(self.gram_target.square().sum())
        self.h = self.x.T-self.x
        self.near = self.h.abs() <= cfg.near
        self.safe_h = torch.where(self.near, torch.ones_like(self.h), self.h)
        t = self.x_np
        J = {1: np.log1p(-t)-np.log1p(t)}
        J.update({q: ((1-t)**(1-q)-(-1-t)**(1-q))/(1-q) for q in range(2, m+1)})
        self.analytic = tensor(np.stack([J[m-j] for j in range(m)], axis=1))
        report = json.loads((Path(data_dir)/'validation.json').read_text())
        self.rhs_setup_seconds = next(r['setup_seconds'] for r in report['checks'] if r['m'] == m and r['family'] == family)
        self.offline_seconds = time.perf_counter()-start

    def samples(self, rng, count=None):
        shape = (self.cfg.q, self.cfg.mc_samples if count is None else count)
        return rng.uniform(-1, 1, shape), rng.beta(1, self.m, shape)

    def direct_from_jet(self, values, coeff):
        p = coeff[:, self.m-1, None]
        for j in range(self.m-2, -1, -1):
            p = p*self.h+coeff[:, j, None]
        far = (values[None, :]-p)/self.safe_h**self.m
        local = coeff[:, -1, None]
        for j in range(coeff.shape[1]-2, self.m-1, -1):
            local = local*self.h+coeff[:, j, None]
        rem = torch.where(self.near, local, far)
        return (self.analytic*coeff[:, :self.m]).sum(dim=1)+rem@self.w

    def residual(self, model, method, samples=None):
        if method in ('gram', 'equal'):
            values = model(self.x).flatten()
            return self.gram_map@values-self.gram_target if method == 'gram' else self.equal_map@values-self.equal_target
        if method == 'pinn':
            coeff = model.jet(self.x, self.m+self.cfg.near_degree)
            return (self.direct_from_jet(coeff[:, 0], coeff)-self.f)/self.point_scale
        if method == 'mc':
            s, eta = samples
            points = self.x_np[:, None]+eta*(s-self.x_np[:, None])
            nodes = torch.as_tensor(points.reshape(-1, 1), dtype=torch.float64, device=self.cfg.device)
            rem = model.jet(nodes, self.m)[:, self.m].reshape(points.shape).mean(dim=1)
            coeff = model.jet(self.x, self.m-1)
            return ((self.analytic*coeff).sum(dim=1)+2*rem-self.f)/self.point_scale
        raise ValueError(method)

    def loss(self, model, method, samples=None):
        r = self.residual(model, method, samples)
        return r.square().sum()/({'gram': self.gram_den, 'equal': self.equal_den}[method]) if method in ('equal', 'gram') else r.square().mean()


def tensor_values(model, x, device):
    with torch.no_grad():
        return model(torch.as_tensor(x[:, None], dtype=torch.float64, device=device)).cpu().numpy().ravel()


def metrics(model, p, refined=False):
    result = {}
    for q in ([1024, 2048] if refined else [1024]):
        x, w = legendre_rule(q)
        truth = data.exact(x, p.family)
        error = tensor_values(model, x, p.cfg.device)-truth
        l2 = float(np.sqrt(w@(error*error)))
        relative = l2/float(np.sqrt(w@(truth*truth)))
        if q == 1024:
            result.update(solution_l2=l2, solution_relative_l2=relative)
        else:
            result.update(solution_l2_refinement_absolute=abs(l2-result['solution_l2']),
                          solution_relative_l2_refinement_absolute=abs(relative-result['solution_relative_l2']))
    x = np.linspace(-1, 1, 10001)
    truth = data.exact(x, p.family)
    pred = tensor_values(model, x, p.cfg.device)
    error = pred-truth
    result.update(solution_max_sampled=float(np.max(np.abs(error))), solution_rmse_1001=float(np.sqrt(np.mean(error[::10]**2))))
    return result, dict(x=x, prediction=pred, target=truth)


def timed_step(fn, device):
    sync(device)
    t = time.perf_counter()
    value = fn()
    sync(device)
    return value, time.perf_counter()-t


def train_one(out, p, method, seed, deadline):
    out.mkdir(parents=True, exist_ok=False)
    cfg, model = p.cfg, Net(p.cfg, seed)
    initial_hash = state_hash(model)
    torch.save(model.state_dict(), out/'initial.pt')
    rng = np.random.default_rng(seed+100000)
    fixed = p.samples(np.random.default_rng(seed+200000)) if method == 'mc' else None
    if fixed is not None:
        np.savez(out/'lbfgs_samples.npz', s=fixed[0], eta=fixed[1])
    history = []
    adam_seconds, lb_seconds, eval_seconds = 0., 0., 0.
    def objective(samples=fixed):
        return p.loss(model, method, samples)
    def checked_backward(samples):
        if time.monotonic() > deadline:
            raise TimeoutError('explicit batch timeout exceeded')
        model.zero_grad(set_to_none=True)
        loss = objective(samples)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite objective')
        loss.backward()
        if not all(torch.isfinite(t.grad).all() for t in model.parameters()):
            raise FloatingPointError('nonfinite gradient')
        return loss
    def record(phase, step, iterations=0, closures=0):
        nonlocal eval_seconds
        tt = time.perf_counter()
        measured, _ = metrics(model, p)
        row = dict(phase=phase, adam_step=step, lbfgs_iterations=iterations, closures=closures,
                   adam_seconds=adam_seconds, lbfgs_seconds=lb_seconds, **measured)
        history.append(row)
        dump(out/'history.json', history)
        print(json.dumps(dict(event='progress', run=out.name, **row)), flush=True)
        eval_seconds += time.perf_counter()-tt
    _, warmup_seconds = timed_step(lambda: checked_backward(fixed), cfg.device)
    model.zero_grad(set_to_none=True)
    if cfg.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    record('initial', 0)
    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.adam_steps+1):
        def update():
            checked_backward(p.samples(rng) if method == 'mc' else None)
            adam.step()
        _, elapsed = timed_step(update, cfg.device)
        adam_seconds += elapsed
        if step % cfg.log_every == 0 or step == cfg.adam_steps:
            record('adam', step)
    torch.save(model.state_dict(), out/'adam.pt')
    torch.save(adam.state_dict(), out/'adam_optimizer.pt')
    opt = torch.optim.LBFGS(model.parameters(), max_iter=20, max_eval=80, **LBFGS)
    first = next(model.parameters())
    calls, iterations, blocks = 0, 0, []
    def closure():
        nonlocal calls
        calls += 1
        return checked_backward(fixed)
    while iterations < cfg.lbfgs_steps:
        block = min(20, cfg.lbfgs_steps-iterations)
        opt.param_groups[0].update(max_iter=block, max_eval=4*block)
        before = iterations
        _, elapsed = timed_step(lambda: opt.step(closure), cfg.device)
        lb_seconds += elapsed
        iterations = int(opt.state[first].get('n_iter', 0))
        blocks.append(dict(previous=before, actual=iterations, requested=block))
        record('lbfgs', cfg.adam_steps, iterations, calls)
        if iterations-before < block:
            break
    assert iterations <= cfg.lbfgs_steps
    assert int(opt.state[first].get('func_evals', 0)) == calls
    wall = time.perf_counter()-start
    torch.save(model.state_dict(), out/'final.pt')
    torch.save(opt.state_dict(), out/'lbfgs_optimizer.pt')
    result, predictions = metrics(model, p, refined=True)
    np.savez(out/'predictions.npz', **predictions)
    result.update(status='complete', method=method, m=p.m, family=p.family, seed=seed,
                  config=asdict(cfg), initial_hash=initial_hash, final_hash=state_hash(model),
                  parameters=sum(t.numel() for t in model.parameters()),
                  adam_steps=cfg.adam_steps, lbfgs_iterations=iterations, closure_calls=calls,
                  lbfgs_blocks=blocks, adam_seconds=adam_seconds, lbfgs_seconds=lb_seconds,
                  monitored_loop_seconds=wall, evaluation_seconds=eval_seconds,
                  warmup_seconds=warmup_seconds, offline_seconds_shared=p.offline_seconds,
                  rhs_setup_seconds_shared=p.rhs_setup_seconds, final_loss=float(objective().detach()),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device == 'cuda' else None)
    dump(out/'result.json', result)
    print(json.dumps(dict(event='complete', run=out.name, **result)), flush=True)
    return result


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(device=args.device, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required, no silent CPU fallback')
    receipt, datasets = source_receipt(), data_receipt(args.data)
    gate = json.loads(Path(args.preflight).read_text())
    assert gate['status'] == 'passed' and gate['device'] == cfg.device
    assert gate['source_sha256'] == receipt and gate['data_sha256'] == datasets
    assert gate['all_orders_and_families_checked']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for path in receipt:
        dst = output/'frozen_sources'/path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, dst)
    shutil.copytree(args.data, output/'frozen_data')
    shutil.copy2(args.preflight, output/'preflight.json')
    jobs = [(m, family, seed, method) for m in args.orders for family in ('exp', 'multiscale')
            for seed in args.seeds for method in METHODS]
    manifest = dict(status='running', jobs=jobs, expected_runs=len(jobs), config=asdict(cfg),
                    source_sha256=receipt, data_sha256=datasets, preflight_sha256=sha(args.preflight),
                    host=platform.node(), python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name() if cfg.device == 'cuda' else None, pid=os.getpid(),
                    completed_runs=0, experiments_2_3_enabled=False, raw_u=True, invalidated_batch_reused=False)
    dump(output/'manifest.json', manifest)
    deadline, rows = time.monotonic()+args.timeout_hours*3600, []
    try:
        for m in args.orders:
            for family in ('exp', 'multiscale'):
                p = Problem(m, family, cfg, args.data)
                np.savez(output/f'm{m}_{family}_operator.npz', moments=p.moments_np,
                         ordinary_l2_gram=p.T_np, whitening=p.whitening_np, lam=p.lam)
                for index, seed in enumerate(args.seeds):
                    for method in METHODS[index:]+METHODS[:index]:
                        name = f'm{m}_{family}_{method}_seed{seed}'
                        print(json.dumps(dict(event='start', run=name, completed_runs=len(rows), expected_runs=len(jobs))), flush=True)
                        rows.append(train_one(output/name, p, method, seed, deadline))
                        dump(output/'results.json', rows)
                        manifest['completed_runs'] = len(rows)
                        dump(output/'manifest.json', manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc), completed_runs=len(rows))
        dump(output/'manifest.json', manifest)
        raise
    dump(output/'manifest.json', manifest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--preflight', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--orders', nargs='+', type=int, default=[2, 4, 6])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--adam-steps', type=int, default=1000)
    ap.add_argument('--lbfgs-steps', type=int, default=300)
    ap.add_argument('--timeout-hours', type=float, default=24.)
    run(ap.parse_args())
