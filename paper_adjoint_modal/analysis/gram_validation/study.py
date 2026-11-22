"""Reproduce the supplementary GP-VNN checks and loss ablation."""
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import argparse
import json
import platform
import sys
import tempfile
import time

import mpmath as mp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/hie_gram_direct_u_20261018"))
import experiment as base
import rhs_data
sys.path.insert(0, str(ROOT / "research/hie_rezaei_poly_20261018"))
import product_quadrature
from data import exact as nonoscillatory_solution

RESULTS = ROOT / "results"
SINE_RESULTS = RESULTS / "hie_trig_compare_20261018"
ORDERS = (2, 4, 6)
FAMILIES = ("exp", "polynomial", "sine")
ABLATION_CASES = ((6, "exp"), (4, "polynomial"), (4, "sine"))
RULE_SIZES = (32, 48, 64, 192, 384)


def source_dir(family):
    return ROOT / "data" / "inputs"


def exact(nodes, family):
    if family == "sine":
        return np.sin(2 * np.pi * nodes)
    return nonoscillatory_solution(nodes, family)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def load_sine_model(config, seed, order, method, stage):
    path = SINE_RESULTS / f"run_v1/m{order}_sine_{method}_seed{seed}/{stage}.pt"
    model = base.Net(config, seed)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def numerical_checks(config):
    consistency = []
    differentiators = []
    product_refinement = []
    for family in FAMILIES:
        problems = [base.Problem(m, family, config, source_dir(family)) for m in ORDERS]
        for problem in problems:
            for size in RULE_SIZES:
                if size == config.q:
                    # Use the actual discrete training operator and its saved nodes.
                    nodes = problem.x_np
                    matrix = problem.gram_map.numpy()
                else:
                    nodes, weights = base.legendre_rule(size)
                    moments = rhs_data.basis(nodes, config.modes, problem.m // 2)
                    matrix = problem.whitening_np @ (moments.T * weights)
                residual = matrix @ exact(nodes, family) - problem.gram_target.numpy()
                consistency.append(dict(family=family, order=problem.m, q=size,
                                        relative_defect=float(np.linalg.norm(residual)
                                                              / np.sqrt(problem.gram_den))))
        for seed in base.SEEDS:
            models = {"initial": base.Net(config, seed)}
            for order in ORDERS:
                models[f"sine_m{order}_final"] = load_sine_model(config, seed, order, "gram", "final")
            for state, model in models.items():
                values = model(problems[0].x).flatten()
                actual = [(problem.loss(model, "gram"), problem) for problem in problems]
                loss_gradients = [base.gradient(loss, model) for loss, _ in actual]
                # Independent orthonormal-coordinate evaluation on the same quadrature.
                x = problems[0].x_np
                w = problems[0].w_np
                legendre = np.polynomial.legendre.legvander(x, config.modes - 1)
                legendre *= np.sqrt((2 * np.arange(config.modes) + 1) / 2)
                moments = tensor(legendre.T * w)
                reference_data = moments @ tensor(exact(x, family))
                reference = (moments @ values - reference_data).square().sum() / reference_data.square().sum()
                reference_gradient = base.gradient(reference, model)
                value_scale = max(abs(float(reference.detach())), np.finfo(float).tiny)
                gradient_scale = max(float(reference_gradient.norm()), np.finfo(float).tiny)
                losses = [float(loss.detach()) for loss, _ in actual]
                differentiators.append(dict(
                    family=family, seed=seed, state=state, loss_m2=losses[0],
                    max_relative_loss=float(max(abs(a - b) for a, b in combinations(losses, 2)) / value_scale),
                    max_relative_gradient=max(float((a - b).norm()) for a, b in combinations(loss_gradients, 2))
                    / gradient_scale,
                    max_reference_loss=max(abs(a - float(reference.detach())) for a in losses) / value_scale,
                    max_reference_gradient=max(float((a - reference_gradient).norm()) for a in loss_gradients)
                    / gradient_scale,
                ))
    problem = base.Problem(6, "sine", config, source_dir("sine"))
    products = {}
    for q in (64, 96, 128, 192, 256):
        nodes, matrix = product_quadrature.matrix(6, problem.x_np, q)
        products[q] = tensor(nodes[:, None]), tensor(matrix)
    for seed in base.SEEDS:
        model = load_sine_model(config, seed, 6, "product", "final")
        losses, gradients = {}, {}
        for q, (nodes, matrix) in products.items():
            residual = (matrix @ model(nodes).flatten() - problem.f) / problem.point_scale
            loss = residual.square().mean()
            losses[q] = float(loss.detach())
            gradients[q] = base.gradient(loss, model)
        for q in (96, 128, 192, 256):
            difference = float((gradients[64] - gradients[q]).norm())
            scale = float(gradients[q].norm())
            product_refinement.append(dict(seed=seed, q=q,
                                           loss_absolute=abs(losses[64] - losses[q]),
                                           gradient_absolute=difference,
                                           gradient_reference_norm=scale,
                                           gradient_relative=difference / max(scale, 1e-30),
                                           gradient_pass=bool(difference < 1e-8 + 1e-4 * scale)))
    base.dump(HERE / "numerical_checks.json", dict(
        config=asdict(config), quadrature_sizes=RULE_SIZES,
        training_data={family: base.data_receipt(source_dir(family)) for family in FAMILIES},
        consistency=consistency, order_checks=differentiators,
        rezaei_m6_sine=product_refinement, python=platform.python_version(), torch=torch.__version__,
    ))


def normalized_matrices(problem):
    # G_kk = (c_m*lambda_k)^2*T_kk. Cancel these factors before whitening by the diagonal.
    scale = np.sqrt(np.diag(problem.T_np))
    diagonal_map = tensor(problem.moments_np / scale[:, None])
    diagonal_target = tensor(problem.b_np / (problem.cm * problem.lam * scale))
    representations = {
        "equal": (problem.equal_map, problem.equal_target),
        "diagonal": (diagonal_map, diagonal_target),
        "gram": (problem.gram_map, problem.gram_target),
    }
    return {name: (matrix / target.norm(), target / target.norm())
            for name, (matrix, target) in representations.items()}


def assess(model, family, size):
    nodes, weights = base.legendre_rule(size)
    reference = exact(nodes, family)
    error = base.tensor_values(model, nodes, "cpu") - reference
    return float(np.sqrt((weights @ error**2) / (weights @ reference**2)))


def train(problem, method, seed, match_initial_gradient=False):
    cfg = problem.cfg
    model = base.Net(cfg, seed)
    initial_hash = base.state_hash(model)
    matrices = normalized_matrices(problem)
    matrix, target = matrices[method]
    loss_scale = 1.

    def objective():
        return loss_scale * (matrix @ model(problem.x).flatten() - target).square().sum()

    initial_grad = float(base.gradient(objective(), model).norm())
    if match_initial_gradient:
        gram_map, gram_target = matrices["gram"]
        gram_loss = (gram_map @ model(problem.x).flatten() - gram_target).square().sum()
        loss_scale = float(base.gradient(gram_loss, model).norm()) / initial_grad
    initial_loss = float(objective().detach())

    def closure():
        model.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start = time.perf_counter()
    for _ in range(cfg.adam_steps):
        closure()
        adam.step()
    adam_seconds = time.perf_counter() - start
    adam_error = assess(model, problem.family, 1024)

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20, max_eval=80, **base.LBFGS)
    first = next(model.parameters())
    iterations = 0
    start = time.perf_counter()
    while iterations < cfg.lbfgs_steps:
        block = min(20, cfg.lbfgs_steps - iterations)
        lbfgs.param_groups[0].update(max_iter=block, max_eval=4 * block)
        previous = iterations
        lbfgs.step(closure)
        iterations = int(lbfgs.state[first].get("n_iter", 0))
        if iterations - previous < block:
            break
    lbfgs_seconds = time.perf_counter() - start
    relative_error = assess(model, problem.family, 1024)
    refinement = assess(model, problem.family, 2048)
    nodes, weights = base.legendre_rule(1024)
    error = base.tensor_values(model, nodes, "cpu") - exact(nodes, problem.family)
    basis = np.polynomial.legendre.legvander(nodes, cfg.modes - 1)
    basis *= np.sqrt((2 * np.arange(cfg.modes) + 1) / 2)
    coefficients = basis.T @ (weights * error)
    error_energy = float(weights @ error**2)
    projected_energy = float(coefficients @ coefficients)
    tail_energy = float(weights @ (error - basis @ coefficients)**2)
    assert np.isclose(error_energy, projected_energy + tail_energy, rtol=1e-10, atol=1e-28)
    assert np.isclose(relative_error, refinement, rtol=1e-7, atol=1e-11)
    assert iterations <= cfg.lbfgs_steps and np.isfinite(relative_error)
    return dict(family=problem.family, order=problem.m, method=method, seed=seed, M=cfg.modes,
                Q=cfg.q, initial_hash=initial_hash, initial_loss=initial_loss, loss_scale=loss_scale,
                initial_gradient_norm=loss_scale * initial_grad, adam_relative_l2=adam_error,
                relative_l2=relative_error, refined_relative_l2=refinement,
                projected_error_energy=projected_energy, tail_error_energy=tail_energy,
                final_loss=float(objective().detach()), adam_seconds=adam_seconds,
                lbfgs_seconds=lbfgs_seconds, lbfgs_iterations=iterations,
                closure_calls=int(lbfgs.state[first]["func_evals"]))


def training_study(config, study):
    output = HERE / f"{study}.json"
    report = dict(config=asdict(config), python=platform.python_version(), torch=torch.__version__,
                  machine=platform.machine(), cpu_threads=torch.get_num_threads(),
                  source_sha256=base.sha(__file__), status="running", runs=[])
    if study in ("ablation", "gradient_matched"):
        methods = ("equal", "diagonal", "gram") if study == "ablation" else ("equal", "diagonal")
        groups = [(order, family, config.modes, methods)
                  for order, family in ABLATION_CASES]
    else:
        # Fixed quadrature resolves the integration independently of the test dimension.
        groups = [(4, "sine", modes, ("gram",)) for modes in (16, 32, 64, 128)]
        moments = sine_moments(128)
        report["known_rhs_moments"] = moments.tolist()
        report["moment_precision_digits"] = [170, 200]
    for order, family, modes, methods in groups:
        setting = replace(config, modes=modes)
        if study == "dimension":
            problem = sine_problem(setting, moments)
        else:
            problem = base.Problem(order, family, setting, source_dir(family))
        for seed in base.SEEDS:
            rows = [train(problem, method, seed, match_initial_gradient=study == "gradient_matched")
                    for method in methods]
            assert len({row["initial_hash"] for row in rows}) == 1
            report["runs"].extend(rows)
            base.dump(output, report)
            for row in rows:
                print(json.dumps(row), flush=True)
    report["status"] = "complete"
    base.dump(output, report)


def sine_moments(modes):
    def build(digits, degree):
        with mp.workdps(digits):
            z = 2 * mp.pi * 1j
            coeff = rhs_data.exponential_coeff(z, degree)
            regular = rhs_data.hilbert_regular_coeff(coeff)
            basis = rhs_data.polys(2, modes)
            mu, nu = rhs_data.regular_moments(2, degree + modes)
            data, _, _ = rhs_data.build_b(4, z, basis, mu, nu, coeff, regular)
            reference = rhs_data.verification_reference(4, basis, coeff)
            defect = max(abs(mp.im(a - b)) / max(1, abs(mp.im(b))) for a, b in zip(data, reference))
            assert defect < mp.mpf("1e-70")
            # Even degrees vanish by parity of the odd exact solution.
            return np.array([float(mp.im(value)) if k % 2 else 0. for k, value in enumerate(data)])

    moments = build(170, 256)
    np.testing.assert_array_equal(moments, build(200, 288))
    with np.load(source_dir("sine") / "m4_sine.npz") as saved:
        np.testing.assert_allclose(moments[:64], saved["b"], atol=1e-70, rtol=1e-15)
    return moments


def sine_problem(cfg, moments):
    original = source_dir("sine")
    with np.load(original / "m4_sine.npz") as saved:
        data = {key: saved[key] for key in ("x192", "w192", "f192")}
        data["b"] = saved["b"][:cfg.modes].copy() if cfg.modes <= 64 else moments[:cfg.modes].copy()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.savez(path / "m4_sine.npz", **data)
        (path / "validation.json").write_bytes((original / "validation.json").read_bytes())
        return base.Problem(4, "sine", cfg, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("checks", "ablation", "gradient_matched", "dimension"))
    mode = parser.parse_args().mode
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = base.Config()
    if mode == "checks":
        numerical_checks(cfg)
    else:
        training_study(cfg, mode)


if __name__ == "__main__":
    main()
