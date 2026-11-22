# Training Reliability Audit Plan

**Status** — Proposed experiment.

The audit asks whether the automatic differentiation error estimates calibrated on the initial network and the periodically updated derivative envelopes cover the observed divided-difference errors at intermediate Adam and L-BFGS states. In a completed fourth-order paired run, derivatives observed at update 100 exceeded the previous envelope, and some higher derivatives still exceeded it at update 200. These observations concern the derivative envelope. The audit will measure the divided-difference and solution errors separately.

The first audit covers the existing fourth- and sixth-order double-precision solvers and their existing manufactured solutions. Use an independent run for instrumentation and record audit costs separately from solver timings.

## 1. Save the states used by the optimizer

Save the network parameters, current envelope, calibration version, training nodes and weights, optimizer stage, and step count. Include initialization, the states before and after each envelope update, the midpoint between updates, the final Adam state, and trial and accepted L-BFGS states. Track L-BFGS function evaluations and accepted iterations separately.

Start with one seed. Verify that instrumentation preserves the parameter and loss sequences for the same deterministic configuration, then extend the audit to the three original seeds. Use the same network for the before-and-after envelope comparison. Run the high-precision audit separately from checkpoint collection and retain both timings.

## 2. Check the three sources of evidence

1. **Derivative error.** Build a Taylor-jet reference at 100 decimal digits from the stored parameters and coordinates. Use high precision for subtraction and scaling. Compare the observed automatic differentiation error with the current `eta` at independent audit points, including points outside the envelope calibration grid.
2. **Derivative envelope.** Compare the reference derivatives at each order with the current `B`. Report the observed coverage just after updates, between updates, and during L-BFGS. State the coverage in terms of the sampled points.
3. **Operator error.** For preselected target rows, retain the full training quadrature rule and compute the weighted absolute divided-difference error for the stored inputs. Compare it with the row indicator and budget. Report the contributions from analytic moments and accumulation separately.

Select rows near the smallest diagonal distances, in the interior, and near the endpoints. State the number of audited rows and states when reporting coverage. Fix the sampling rule before inspecting the measured errors. Increase the reference precision once more to check that the reported differences have stabilized.

## 3. Record failures and mismatches

Count states for which the indicator meets the budget, states for which the measured divided-difference error meets the budget, and cases in which the indicator accepts a state whose measured error exceeds the budget. Report separately the states that remain above budget at the highest candidate order. Retain cases in which the previous envelope is exceeded while the actual divided-difference error stays within budget.

If false acceptance occurs, identify whether it comes from automatic differentiation calibration, the remainder envelope, or another arithmetic contribution. Compare possible changes such as more frequent updates, a larger early-training envelope, or additional error checks under the original budget. Include the additional costs in new paired trials and retain all audited checkpoints.

## 4. Audit parameter gradients separately

First check parameter gradients of the residual loss using independent high-precision directional derivatives with the branch selection fixed. Then measure the change in gradient under perturbations near branch switches. Report fixed-branch derivatives and differences with branch reselection separately. If finite differences are used as a supplementary check, verify convergence under step-size refinement.

## 5. Use the audit to decide the next benchmark

- If false acceptance occurs, diagnose and correct the indicator before extending the application.
- If the current evidence covers only initialization, assess intermediate training states before drawing conclusions about the trajectory.
- If the prespecified trajectory audit shows stable accuracy, compare on-demand implementations for fixed orders $p=0,1,2,3$ and optimized automatic differentiation backends.

Report empirical error control for the audited training trajectories, test points, and arithmetic precision.
