# Taylor-Based Manuscript and Reproduction Guide

The main source is [main.tex](main.tex). The table and figure generators read structured experiment results. For the separate GP-VNN paper, see its [code and data](https://github.com/fs3012093583/Hypersingular-Integrals/tree/GPVNN).

## Operator benchmarks with audited reference values

Run the following commands from the repository root. The reference calculations use the actual coordinates stored in each floating-point format and perform the error subtraction in high precision. Use the v2 reference data throughout this pipeline and keep earlier batches separate.

```bash
PYTHONPATH=. python -m experiments.general_taylor_operator_benchmark \
  --preset full --mp-dps 90 --no-plots \
  --output-dir results/general_taylor_operator_benchmark_full_dtype_ref_v2

python -m experiments.general_taylor_multiplier_sweep \
  --reference-metrics results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_multiplier_sweep_dtype_ref_v2

python -m experiments.general_taylor_component_ablation \
  --source results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_component_ablation_dtype_ref_v2

python -m experiments.general_taylor_adaptive_order_benchmark \
  --reference-metrics results/general_taylor_operator_benchmark_full_dtype_ref_v2/metrics.json \
  --output-dir results/general_taylor_adaptive_order_dtype_ref_v2

python -m experiments.operator_budget_accuracy_benchmark \
  --quadrature-points 256 --mp-dps 90 --residual-budget 1e-4 \
  --output-dir results/operator_budget_accuracy_v2_20260919

python -m experiments.neural_high_order_derivative_benchmark \
  --highest-order 10 --decimal-digits 100 \
  --output-dir results/neural_high_order_derivatives_ref_v2

python -m paper.build_operator_revision_assets
python -m paper.build_derivative_audit_figure
```

The archived records contain the configurations for the earlier neural experiments, nonlinear examples, and stress tests across solution families. The earlier `build_paper_figures` script reproduces the historical plots.

## Paired benchmark for on-demand fixed-order evaluation

The benchmark compares the original fixed expansion, an on-demand fixed expansion with the same formula and cutoff, and the operator-budget method. Runs are serial, use one CPU thread, and rotate the method order across seeds. Use a fresh output directory for each new run.

```bash
python -m experiments.lazy_fixed_quality_benchmark \
  --task single --orders 4 6 --seeds 20260918 20260919 20260920 \
  --derivative-metrics results/neural_high_order_derivatives_ref_v2/metrics.json \
  --output-dir results/lazy_fixed_quality_single_calibration_v2_20260919

python -m experiments.lazy_fixed_quality_benchmark \
  --task parameterized --orders 4 --seeds 20260918 20260919 20260920 \
  --output-dir results/lazy_fixed_quality_parameterized_calibration_v2_20260919
```

Run the two commands in sequence and schedule each timing batch without concurrent numerical jobs. The v2 batch dated 19 September 2026 contains 18 single-function runs and nine parametric runs. The archived source and calibration files were checked against their SHA-256 digests. The main comparisons use this batch. The earlier single-seed ablation remains labeled with its historical calibration.

- `method_setup_and_training_seconds` includes the initial derivative envelope and its updates during training. The parametric benchmark also includes per-run high-precision calibration. The single-function benchmark uses an existing automatic differentiation calibration file and records its SHA-256 digest. Its timing starts after that file has been generated.
- `solve_seconds_before_validation` additionally includes model, node, and right-hand-side construction. Validation and the isolated backward-pass timing probe are recorded separately.
- The saved tensor payload reports the size of saved tensor data. Use it as a storage measure.

The operator error includes contributions from divided-difference replacement, quadrature, analytic moments, and floating-point accumulation. The weighted divided-difference indicator in the proposition controls the replacement term. Results from the three seeds describe the observed runs.

## Tables and PDF build

The table generator checks the three distinct seeds in each group, paired initializations, and consistency of the experiment settings.

```bash
python -m paper.build_quality_revision_tables \
  --single results/lazy_fixed_quality_single_calibration_v2_20260919/metrics.json \
  --parameterized results/lazy_fixed_quality_parameterized_calibration_v2_20260919/metrics.json \
  --operator results/operator_budget_accuracy_v2_20260919/metrics.json \
  --output-dir paper/tables_quality
```

Build from the `paper` directory so the relative figure and table paths resolve.

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../output/pdf/quality_revision_calibration_v2_20260919 main.tex
```

The build writes to a separate revision directory. The [training reliability audit plan](../reports/training_reliability_audit_plan.md) describes a proposed check of the error indicators along optimization trajectories.
