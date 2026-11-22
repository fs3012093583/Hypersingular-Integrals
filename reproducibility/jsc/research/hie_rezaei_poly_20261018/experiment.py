"""54 authorized supplements: Rezaei product quadrature and mapped cubic case.

The frozen old experiment is imported as a dependency, never edited.
Its network, initialization, PINN/MC/modal residuals and optimizer settings
are retained; physical metrics use the newly confirmed exact solution.
"""
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

import data
import product_quadrature as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/hie_gram_direct_u_20261018'))
# Import the old module under a unique name: this file is also experiment.py.
import importlib.util
spec = importlib.util.spec_from_file_location('direct_u_frozen_base', ROOT/'research/hie_gram_direct_u_20261018/experiment.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

METHODS = ('pinn', 'mc', 'product', 'equal', 'gram')
SEEDS = base.SEEDS
Config, Net, gradient, dump, sha = base.Config, base.Net, base.gradient, base.dump, base.sha
PRODUCT_NODES = 64


def receipt():
    files = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.md'))
    files += list((ROOT/'research/hie_gram_direct_u_20261018').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


class Problem(base.Problem):
    def __init__(self, m, family, cfg, data_dir, product_nodes=PRODUCT_NODES):
        super().__init__(m, family, cfg, data_dir)
        start = time.perf_counter()
        self.product_nodes = product_nodes
        nodes, matrix = pq.matrix(m, self.x_np, product_nodes)
        self.product_x = torch.as_tensor(nodes[:, None], dtype=torch.float64, device=cfg.device)
        self.product_map = torch.as_tensor(matrix, dtype=torch.float64, device=cfg.device)
        self.product_assembly_seconds = time.perf_counter()-start

    def residual(self, model, method, samples=None):
        if method == 'product':
            return (self.product_map@model(self.product_x).flatten()-self.f)/self.point_scale
        return super().residual(model, method, samples)


def metrics(model, p, refined=False):
    result = {}
    for q in ([1024, 2048] if refined else [1024]):
        x, w = base.legendre_rule(q)
        truth = data.exact(x, p.family)
        error = base.tensor_values(model, x, p.cfg.device)-truth
        l2 = float(np.sqrt(w@(error*error)))
        relative = l2/float(np.sqrt(w@(truth*truth)))
        if q == 1024:
            result.update(solution_l2=l2, solution_relative_l2=relative)
        else:
            result.update(solution_l2_refinement_absolute=abs(l2-result['solution_l2']),
                          solution_relative_l2_refinement_absolute=abs(relative-result['solution_relative_l2']))
    x = np.linspace(-1, 1, 10001)
    truth = data.exact(x, p.family)
    pred = base.tensor_values(model, x, p.cfg.device)
    error = pred-truth
    result.update(solution_max_sampled=float(np.max(abs(error))),
                  solution_rmse_1001=float(np.sqrt(np.mean(error[::10]**2))))
    return result, dict(x=x, prediction=pred, target=truth)


def train_one(out, p, method, seed, deadline):
    out.mkdir(parents=True, exist_ok=False)
    cfg, model = p.cfg, Net(p.cfg, seed)
    initial_hash = base.state_hash(model)
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
            raise TimeoutError('batch deadline exceeded')
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
    _, warmup_seconds = base.timed_step(lambda: checked_backward(fixed), cfg.device)
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
        _, elapsed = base.timed_step(update, cfg.device)
        adam_seconds += elapsed
        if step % cfg.log_every == 0 or step == cfg.adam_steps:
            record('adam', step)
    torch.save(model.state_dict(), out/'adam.pt')
    torch.save(adam.state_dict(), out/'adam_optimizer.pt')
    opt = torch.optim.LBFGS(model.parameters(), max_iter=20, max_eval=80, **base.LBFGS)
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
        _, elapsed = base.timed_step(lambda: opt.step(closure), cfg.device)
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
                  config=asdict(cfg), initial_hash=initial_hash, final_hash=base.state_hash(model),
                  parameters=sum(t.numel() for t in model.parameters()), adam_steps=cfg.adam_steps,
                  lbfgs_iterations=iterations, closure_calls=calls, lbfgs_blocks=blocks,
                  adam_seconds=adam_seconds, lbfgs_seconds=lb_seconds, monitored_loop_seconds=wall,
                  evaluation_seconds=eval_seconds, warmup_seconds=warmup_seconds,
                  offline_seconds_shared=p.offline_seconds, rhs_setup_seconds_shared=p.rhs_setup_seconds,
                  product_nodes=p.product_nodes, product_assembly_seconds=p.product_assembly_seconds,
                  final_loss=float(objective().detach()),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device == 'cuda' else None)
    dump(out/'result.json', result)
    print(json.dumps(dict(event='complete', run=out.name, **result)), flush=True)
    return result


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(device=args.device)
    gate = json.loads(Path(args.preflight).read_text())
    assert gate['status'] == 'passed' and gate['device'] == cfg.device
    assert gate['source_sha256'] == receipt()
    assert gate['data_sha256'] == base.data_receipt(args.data)
    assert cfg.device == 'cuda' and torch.cuda.is_available()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.data, out/'frozen_data')
    for rel in receipt():
        p = out/'frozen_sources'/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, p)
    jobs = [(m, family, seed, method) for m in (2,4,6) for family in ('exp','polynomial')
            for seed in SEEDS for method in (('product',) if family == 'exp' else METHODS)]
    manifest = dict(status='running', expected_runs=54, completed_runs=0, jobs=jobs,
                    config=asdict(cfg), product_nodes=PRODUCT_NODES, source_sha256=receipt(),
                    data_sha256=base.data_receipt(args.data), preflight_sha256=sha(args.preflight),
                    host=platform.node(), python=platform.python_version(), numpy=np.__version__,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(), pid=os.getpid(),
                    raw_u=True, experiments_2_3_enabled=False, old_multiscale_used=False,
                    extra_endpoint_loss=False)
    dump(out/'manifest.json', manifest)
    results, problems = [], {}
    deadline = time.monotonic()+24*3600
    try:
        for m, family, seed, method in jobs:
            key = m, family
            if key not in problems:
                problems[key] = Problem(m, family, cfg, args.data)
            name = f'm{m}_{family}_{method}_seed{seed}'
            results.append(train_one(out/name, problems[key], method, seed, deadline))
            dump(out/'results.json', results)
            manifest['completed_runs'] = len(results)
            dump(out/'manifest.json', manifest)
        manifest['status'] = 'complete'
        dump(out/'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        failure=repr(exc), completed_runs=len(results))
        dump(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--preflight', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
