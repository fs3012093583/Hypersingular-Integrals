# GP-VNN reproducibility materials

**A Gram-preconditioned variational neural network method for hypersingular integral equations**

Shun Fan, Haoyun Li, and Chaolang Hu, School of Mathematics, Sichuan University.

## Data

- data/inputs contains the Gauss–Legendre nodes, quadrature weights, pointwise right-hand-side values, and independently computed finite-part data moments for the nine test problems.
- data/reported/combined_raw_metrics.csv contains the 135 training records used in the accuracy and Adam timing tables.
- data/reported/ablation.json, gradient_matched.json, and dimension.json contain the measurements from the loss and test-dimension studies.
- data/reported/direction_coefficients.csv and model_summary.csv contain the error spectrum and energy fractions for the second-order sine plot.
- data/reported/numerical_checks.json and independent_audit.json contain the numerical diagnostics and refinement records.

Run the following script from this directory to recompute the relative L2 error medians, median absolute deviations, and mean time per Adam update from the archived measurements.

    python summarize.py

## Training

Install Python 3.10 or later with NumPy, mpmath, and PyTorch appropriate for the target device. The primary results were recorded using Python 3.10.20, PyTorch 2.5.1, and CUDA 12.1 on an NVIDIA GeForce RTX 4060 Ti. The ablation studies used PyTorch 2.12.0 on one Apple M2 Pro CPU thread.

    python run_case.py --family sine --order 2 --method gram --seed 20261030 --device cpu --output runs

Use the families exp, polynomial, and sine, orders 2, 4, and 6, and seeds 20261030, 20261031, and 20261032. The method names are pinn (PINN), mc (MC-PINN), product (Rezaei-Quad), equal (EW-VNN), and gram (GP-VNN). The default budget is 1000 Adam updates and at most 300 L-BFGS iterations. The wrapper selects the quadratic PINN cutoffs in the reported records and uses the original loss, differentiation, initialization, and optimizer implementations.

To reproduce the loss ablation, the gradient-matched comparison, or the test-dimension study, run the following commands.

    python paper_adjoint_modal/analysis/gram_validation/study.py ablation
    python paper_adjoint_modal/analysis/gram_validation/study.py gradient_matched
    python paper_adjoint_modal/analysis/gram_validation/study.py dimension

For the saved-state numerical checks, generate the nine final GP-VNN sine networks and the three sixth-order Rezaei-Quad sine networks, using --output results/hie_trig_compare_20261018/run_v1. Then run the following command.

    python paper_adjoint_modal/analysis/gram_validation/study.py checks

Training creates model and optimizer states in the requested local output directory. The reported timings refer to the recorded hardware and software settings.
