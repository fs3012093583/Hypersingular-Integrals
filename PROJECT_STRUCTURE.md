# Project Structure

## Solvers

[conventional_solvers/](conventional_solvers/) contains the classical numerical methods.

- [hypersingular_solver.py](conventional_solvers/hypersingular_solver.py) implements the corrected midpoint rule and the productivity calculation.
- [taylor_spectral_collocation.py](conventional_solvers/taylor_spectral_collocation.py) implements Taylor-based spectral collocation.
- [examples.py](conventional_solvers/examples.py) demonstrates solver usage, convergence studies, and grid comparisons.
- [test_solver.py](conventional_solvers/test_solver.py) runs checks for the midpoint solver.

[neural_network_solvers/](neural_network_solvers/) contains the neural variants and supporting numerical methods.

- [neural_hypersingular_solver.py](neural_network_solvers/neural_hypersingular_solver.py) implements a neural solver for the integral equation.
- [discrete_integral_neural_solver.py](neural_network_solvers/discrete_integral_neural_solver.py) uses a discrete integral formulation.
- [neraul_hypersingular_solver_fs.py](neural_network_solvers/neraul_hypersingular_solver_fs.py) contains another neural solver variant.
- [general_taylor_finite_part.py](neural_network_solvers/general_taylor_finite_part.py) implements Taylor-based finite-part evaluation.

## Experiments and manuscripts

| Directory | Contents |
|---|---|
| [experiments](experiments/) | Benchmark drivers for operator accuracy, neural training, and ablation studies |
| [results](results/) | Structured experiment records and metrics |
| [paper](paper/) | LaTeX source, generators, and tables for the Taylor-based manuscript |
| [output/pdf](output/pdf/) | Archived PDF builds |
| [reports](reports/) | Experiment reports and planned audits |
| [tests](tests/) | Tests for solvers, numerical identities, benchmark protocols, and paper assets |

Run the midpoint examples from the repository root. The examples write generated plots such as `custom_analysis.png` and `batch_convergence.png` in the working directory. The [paper README](paper/README.md) explains how to regenerate the Taylor manuscript's figures and tables.

For the Gram-preconditioned variational neural network (GP-VNN) paper, start with the [versioned reproducibility directory](https://github.com/fs3012093583/Hypersingular-Integral-Equation-/tree/7a45bdd0d940b2473259ba2025320023d1767f38/reproducibility/jsc). It includes its own installation and training instructions.
