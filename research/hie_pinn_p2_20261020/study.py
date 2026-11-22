"""Isolated, user-authorized m+2 PINN replacement; old sources are read-only."""
from dataclasses import asdict, replace
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'research/hie_rezaei_poly_20261018'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('p2_original_training', OLD / 'experiment.py')
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)
FAMILIES = ('exp', 'polynomial', 'sine')
ORDERS = (2, 4, 6)
NEAR = .01


def exact(x, family):
    if family == 'sine':
        return np.sin(2*np.pi*x)
    return e.data.exact(x, family)


def metrics(model, p, refined=False):
    out = {}
    for q in ([1024, 2048] if refined else [1024]):
        x, w = e.base.legendre_rule(q)
        truth = exact(x, p.family)
        error = e.base.tensor_values(model, x, p.cfg.device)-truth
        l2 = float(np.sqrt(w@(error*error)))
        rel = l2 / float(np.sqrt(w@(truth*truth)))
        if q == 1024:
            out.update(solution_l2=l2, solution_relative_l2=rel)
        else:
            out.update(solution_l2_refinement_absolute=abs(l2-out['solution_l2']),
                       solution_relative_l2_refinement_absolute=abs(rel-out['solution_relative_l2']))
    x = np.linspace(-1, 1, 10001)
    truth = exact(x, p.family)
    prediction = e.base.tensor_values(model, x, p.cfg.device)
    error = prediction-truth
    out.update(solution_max_sampled=float(abs(error).max()),
               solution_rmse_1001=float(np.sqrt(np.mean(error[::10]**2))))
    return out, dict(x=x, prediction=prediction, target=truth)


# The original timer, initialization and optimizer loop is reused unmodified.
e.metrics = metrics


class Problem(e.base.Problem):
    def __init__(self, m, family, cfg, data_dir):
        super().__init__(m, family, cfg, data_dir)
        self.product_nodes = 0
        self.product_assembly_seconds = 0.

    def probe(self, model, degree=2, near=NEAR, iq=192):
        """Refine integration nodes without changing 192 collocation nodes."""
        s, w = e.base.legendre_rule(iq)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=self.cfg.device)
        h = tensor(s[None, :]-self.x_np[:, None])
        mask = h.abs() <= near
        safe = torch.where(mask, torch.ones_like(h), h)
        c = model.jet(self.x, self.m+degree)
        polynomial = c[:, self.m-1, None]
        for j in range(self.m-2, -1, -1):
            polynomial = polynomial*h+c[:, j, None]
        far = (model(tensor(s[:, None])).flatten()[None, :]-polynomial)/safe**self.m
        local = c[:, -1, None]
        for j in range(c.shape[1]-2, self.m-1, -1):
            local = local*h+c[:, j, None]
        op = (self.analytic*c[:, :self.m]).sum(1)+torch.where(mask, local, far)@tensor(w)
        return (op-self.f)/self.point_scale

    def integral_reference(self, model, iq=192, etaq=12, near=.06):
        """Exact Taylor integral remainder, deterministic quadrature, AD only to m.

        R_m(t,s) = integral_0^1 m(1-a)^(m-1) [u^(m)/m!](t+a(s-t)) da.
        Checkpointing limits the independent audit's memory footprint.
        """
        s, w = e.base.legendre_rule(iq)
        ax, aw = e.base.legendre_rule(etaq)
        a = (ax+1)/2
        weights = aw/2*self.m*(1-a)**(self.m-1)
        tensor = lambda z: torch.as_tensor(z, dtype=torch.float64, device=self.cfg.device)
        h_np = s[None, :]-self.x_np[:, None]
        mask_np = abs(h_np) <= near
        h, mask = tensor(h_np), torch.as_tensor(mask_np, device=self.cfg.device)
        safe = torch.where(mask, torch.ones_like(h), h)
        c = model.jet(self.x, self.m-1)
        poly = c[:, -1, None]
        for j in range(self.m-2, -1, -1):
            poly = poly*h+c[:, j, None]
        far = (model(tensor(s[:, None])).flatten()[None, :]-poly)/safe**self.m
        regular = torch.where(mask, torch.zeros_like(far), far)@tensor(w)
        ii, jj = np.where(mask_np)
        for start in range(0, len(ii), 96):
            ri, si = ii[start:start+96], jj[start:start+96]
            points = tensor((self.x_np[ri, None]+a[None, :]*h_np[ri, si, None]).reshape(-1, 1))
            def evaluate(z):
                return model.jet(z, self.m)[:, self.m]
            jet = checkpoint(evaluate, points, use_reentrant=False) if torch.is_grad_enabled() else evaluate(points)
            values = jet.reshape(len(ri), etaq)@tensor(weights)
            regular = regular.index_add(0, torch.as_tensor(ri, device=self.cfg.device), values*tensor(w[si]))
        op = (self.analytic*c).sum(1)+regular
        return (op-self.f)/self.point_scale


class Exact:
    def __init__(self, family):
        self.family = family

    def __call__(self, x):
        if self.family == 'exp':
            return x.exp()
        if self.family == 'sine':
            return (2*math.pi*x).sin()
        return (x+1).square()*(x-1)/8

    def jet(self, x, order):
        x = x.flatten()
        if self.family == 'exp':
            return torch.stack([x.exp()/math.factorial(j) for j in range(order+1)], 1)
        if self.family == 'sine':
            return torch.stack([(2*math.pi)**j/math.factorial(j)*torch.sin(2*math.pi*x+j*math.pi/2)
                                for j in range(order+1)], 1)
        c = [(x+1).square()*(x-1)/8, (3*x*x+2*x-1)/8, (3*x+1)/8,
             torch.full_like(x, 1/8)]
        return torch.stack((c+[torch.zeros_like(x)]*max(0, order-3))[:order+1], 1)


def assess(model, fn):
    r = fn()
    loss = r.square().mean()
    grad = e.gradient(loss, model).detach()
    return dict(residual=r.detach(), loss=float(loss.detach()), gradient=grad)


def difference(a, b):
    dl = abs(a['loss']-b['loss'])
    dg = float((a['gradient']-b['gradient']).norm())
    gn = float(b['gradient'].norm())
    dr = float((a['residual']-b['residual']).norm()/math.sqrt(len(b['residual'])))
    rn = float(b['residual'].norm()/math.sqrt(len(b['residual'])))
    return dict(loss_absolute=dl, gradient_absolute=dg, gradient_relative=dg/max(gn, 1e-30),
                reference_gradient_norm=gn, residual_rms_absolute=dr,
                passed=dl < 1e-9+1e-5*abs(b['loss']) and dg < 1e-8+1e-4*gn
                and dr < 1e-7+1e-5*rn)


def sources():
    dirs = ('hie_pinn_p2_20261020', 'hie_rezaei_poly_20261018', 'hie_gram_direct_u_20261018')
    return {str(p.relative_to(ROOT)): e.sha(p) for d in dirs
            for p in sorted((ROOT/'research'/d).glob('*')) if p.suffix in ('.py', '.md', '.sh')}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    checks = []
    for family in FAMILIES:
        prior = ROOT/('results/hie_trig_compare_20261018/data_v1' if family == 'sine'
                     else 'results/hie_rezaei_poly_20261018/data_v2')
        validation = json.loads((prior/'validation.json').read_text())
        assert validation['status'] == 'passed'
        for m in ORDERS:
            name = f'm{m}_{family}.npz'
            shutil.copy2(prior/name, out/name)
            checks.append({**next(r for r in validation['checks'] if r['m'] == m
                                 and (r.get('family', family) == family)),
                           'family': family, 'original_path': str((prior/name).relative_to(ROOT)),
                           'original_sha256': e.sha(prior/name)})
    e.dump(out/'validation.json', dict(status='passed', checks=checks, known_f_data_reused=True,
                                      verification_only_b_used_for_training=False))


def preflight(data, device, output):
    cfg = e.Config(device=device, near_degree=2, near=NEAR)
    checks = []
    for family in FAMILIES:
        for m in ORDERS:
            p = Problem(m, family, cfg, data)
            model = e.Net(cfg, e.SEEDS[0])
            ref = assess(model, lambda: p.integral_reference(model))
            check = dict(family=family, m=m, candidates={})
            for near in (.05, .02, .01):
                val = assess(model, lambda: p.probe(model, near=near))
                check['candidates'][str(near)] = difference(val, ref)
            coarse = assess(model, lambda: p.residual(model, 'pinn'))
            assert check['candidates'][str(NEAR)]['passed'], check
            fine = assess(model, lambda: p.probe(model, degree=8))
            check['degree2_to8'] = difference(coarse, fine)
            fine = assess(model, lambda: p.probe(model, iq=384))
            check['outer192_to384'] = difference(coarse, fine)
            fine = assess(model, lambda: p.integral_reference(model, etaq=16))
            check['integral_eta12_to16'] = difference(ref, fine)
            check['exact_residual_rms'] = float(p.residual(Exact(family), 'pinn').square().mean().sqrt())
            assert check['exact_residual_rms'] < 1e-6, check
            assert all(check[k]['passed'] for k in ('degree2_to8', 'outer192_to384', 'integral_eta12_to16')), check
            checks.append(check)
            print(json.dumps(dict(event='preflight', device=device, **check)), flush=True)
    e.dump(output, dict(status='passed', device=device, config=asdict(cfg), checks=checks,
                        source_sha256=sources(), data_sha256=e.base.data_receipt(data),
                        initial_hashes={str(s): e.base.state_hash(e.Net(cfg, s)) for s in e.SEEDS}))


def run(data, output, gate):
    cfg = e.Config(device='cuda', near_degree=2, near=NEAR)
    verified = json.loads(gate.read_text())
    assert verified['status'] == 'passed' and verified['device'] == 'cuda'
    assert verified['config'] == asdict(cfg)
    assert verified['source_sha256'] == sources()
    assert verified['data_sha256'] == e.base.data_receipt(data)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data, output/'frozen_data')
    for rel in sources():
        dst = output/'frozen_sources'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dst)
    jobs = [(f, m, s) for f in FAMILIES for m in ORDERS for s in e.SEEDS]
    manifest = dict(status='running', expected_runs=27, completed_runs=0, jobs=jobs,
                    source_sha256=sources(), data_sha256=e.base.data_receipt(data),
                    config=asdict(cfg), host=platform.node(), python=platform.python_version(),
                    numpy=np.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    pid=os.getpid(), hard_timeout_seconds=10800, method='pinn',
                    replaces_only='27 degree8 PINN records;108 other-method records unchanged')
    e.dump(output/'manifest.json', manifest)
    deadline = time.monotonic()+10800
    rows = []
    try:
        for family, m, seed in jobs:
            p = Problem(m, family, cfg, output/'frozen_data')
            assert e.base.state_hash(e.Net(cfg, seed)) == verified['initial_hashes'][str(seed)]
            rows.append(e.train_one(output/f'm{m}_{family}_pinn_seed{seed}', p, 'pinn', seed, deadline))
            e.dump(output/'results.json', rows)
            manifest['completed_runs'] = len(rows)
            e.dump(output/'manifest.json', manifest)
        manifest['status'] = 'training_complete_pending_audit'
    except BaseException as exc:
        manifest.update(status='failed', failure=repr(exc))
        raise
    finally:
        e.dump(output/'manifest.json', manifest)


def audit(output, device):
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['completed_runs'] == 27
    assert manifest['source_sha256'] == sources()
    assert manifest['data_sha256'] == e.base.data_receipt(output/'frozen_data')
    rows = json.loads((output/'results.json').read_text())
    checks = []
    for row in rows:
        cfg = e.Config(**{**row['config'], 'device': device})
        p = Problem(row['m'], row['family'], cfg, output/'frozen_data')
        model = e.Net(cfg, row['seed'])
        folder = output/f"m{row['m']}_{row['family']}_pinn_seed{row['seed']}"
        history = json.loads((folder/'history.json').read_text())
        for stage in ('initial', 'adam', 'final'):
            model.load_state_dict(torch.load(folder/f'{stage}.pt', map_location=device, weights_only=True))
            record = history[0] if stage == 'initial' else (
                next(r for r in history if r['phase'] == 'adam' and r['adam_step'] == 1000)
                if stage == 'adam' else row)
            measured, _ = metrics(model, p, refined=stage == 'final')
            assert abs(measured['solution_l2']-record['solution_l2']) < 1e-11+1e-7*record['solution_l2']
            coarse = assess(model, lambda: p.residual(model, 'pinn'))
            item = dict(run=folder.name, stage=stage, metrics=measured, checks={})
            for tag, fn in (
                    ('degree2_to8', lambda: p.probe(model, degree=8)),
                    ('outer192_to384', lambda: p.probe(model, iq=384)),
                    ('integral_remainder', lambda: p.integral_reference(model))):
                item['checks'][tag] = difference(coarse, assess(model, fn))
            item['numerical_refinement_pass'] = all(z['passed'] for z in item['checks'].values())
            if stage in ('initial', 'final'):
                assert e.base.state_hash(model) == row[stage+'_hash']
            checks.append(item)
            print(json.dumps(dict(event='audit', **item)), flush=True)
        opt = torch.load(folder/'lbfgs_optimizer.pt', map_location='cpu', weights_only=False)['state'][0]
        assert opt['n_iter'] == row['lbfgs_iterations'] <= 300
        assert opt['func_evals'] == row['closure_calls']
    failed = [f"{c['run']}/{c['stage']}" for c in checks if not c['numerical_refinement_pass']]
    result = dict(status='passed' if not failed else 'numerical_advisories',
                  native_states_checked=len(checks), checks=checks, failed=failed,
                  manuscript_promotion_allowed=not failed, device=device)
    e.dump(output/f'audit_{device}.json', result)
    if device == 'cuda':
        manifest.update(status='audited' if not failed else 'complete_with_numerical_advisories',
                        audit_sha256=e.sha(output/f'audit_{device}.json'))
        e.dump(output/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'preflight', 'run', 'audit'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'preflight':
        preflight(args.data, args.device, args.output)
    elif args.mode == 'run':
        run(args.data, args.output, args.preflight)
    else:
        audit(args.output, args.device)
