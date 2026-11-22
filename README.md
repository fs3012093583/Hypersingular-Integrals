# Numerical Solvers for Hypersingular Integral Equations

This repository contains conventional and neural network solvers for finite-part integral equations, together with numerical experiments and manuscript materials.

## GP-VNN paper and reproducibility

The paper *A Gram-preconditioned variational neural network method for hypersingular integral equations* studies first-kind equations with even-order kernels. It uses a finite-part adjoint pairing to compute residuals from ordinary integral moments and a Gram loss that measures a polynomial projection of the solution error.

The [versioned reproducibility directory](https://github.com/fs3012093583/Hypersingular-Integral-Equation-/tree/7a45bdd0d940b2473259ba2025320023d1767f38/reproducibility/jsc) contains the training code, quadrature inputs, 135 training records for kernel orders 2, 4, and 6, and scripts for regenerating the reported accuracy and timing summaries. Start with the README in that directory to reproduce the paper's experiments.

## Quick start for the midpoint solver

Install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

The conventional midpoint solver handles the following model problem on $(-1,1)$:

$$
\operatorname{f.p.}\!\int_{-1}^{1}\frac{g(x)}{|x-y|}\,\mathrm dx=-1.
$$

```python
from conventional_solvers.hypersingular_solver import HypersingularIntegralSolver

solver = HypersingularIntegralSolver(N=256)
y, g = solver.solve_equation()
print(f"Productivity estimate: {solver.calculate_productivity(g):.8f}")
solver.plot_solution(y, g)
```

The grid parameter $N$ sets the spacing $h=1/N$ and gives $2N$ midpoint nodes. The corrected midpoint rule incorporates the finite-part contribution into the diagonal of the discrete system $A_N\mathbf{u}=-\mathbf{1}$. Here $\mathbf{u}$ contains the solution values at the nodes and $\mathbf{1}$ is the vector of ones. The entries are

$$
a_{ij}=
\begin{cases}
2\psi(1/2)-2\log\!\bigl(1/(2h)\bigr), & i=j,\\
2/|i-j|, & |i-j|\text{ odd},\\
0, & \text{otherwise},
\end{cases}
$$

where $\psi(1/2)=-\gamma-2\log 2$ and $\gamma$ is the Euler–Mascheroni constant. The solver estimates productivity from the discrete solution as $Q_h=h\sum_i u_i$. For $N=256$, the current implementation returns $Q_h\approx 4.15873584$.

Use `solver.convergence_study([32, 64, 128, 256, 512])` to compare grid resolutions against the $N=1024$ reference used by the implementation. The error reported by this routine is the relative difference in productivity. The solver uses a dense $2N\times 2N$ matrix, so memory use grows with grid resolution.

Run the examples or the solver checks from the repository root.

```bash
python conventional_solvers/examples.py
python conventional_solvers/test_solver.py
```

## Repository guide

| Directory | Contents |
|---|---|
| [conventional_solvers](conventional_solvers/) | Corrected midpoint and Taylor spectral collocation solvers |
| [neural_network_solvers](neural_network_solvers/) | Neural solvers, finite-part formulations, and high-precision reference calculations |
| [experiments](experiments/) | Benchmark and ablation entry points |
| [results](results/) | Structured benchmark results |
| [paper](paper/) | LaTeX manuscript and figure and table generation scripts for the Taylor-based studies |
| [reports](reports/) | Experiment reports and audit protocols |
| [tests](tests/) | Numerical and regression tests |

See the [project structure guide](PROJECT_STRUCTURE.md) for file locations and the [Taylor manuscript README](paper/README.md) for benchmark commands and build instructions.
