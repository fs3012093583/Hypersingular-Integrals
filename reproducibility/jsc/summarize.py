"""Recompute the manuscript summaries from the archived measurements."""

import csv
from collections import defaultdict
from itertools import product
import math
from pathlib import Path
from statistics import mean, median

DATA = Path(__file__).resolve().parent / "data/reported"
NORMS = {"exp": math.sqrt(math.sinh(2)), "polynomial": math.sqrt(2 / 105), "sine": 1.0}


def main():
    grouped = defaultdict(list)
    with (DATA / "combined_raw_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["family"], int(row["m"]), row["method"]
            grouped[key].append(row)
    methods = ("pinn", "mc", "product", "equal", "gram")
    expected_groups = set(product(NORMS, (2, 4, 6), methods))
    if grouped.keys() != expected_groups:
        raise ValueError("The archive must contain all 45 solver, order, and solution groups.")
    print("solution,m,method,median_relative_l2,mad_relative_l2,mean_adam_ms_per_step")
    for (family, order, method), rows in sorted(grouped.items()):
        if len(rows) != 3 or {int(row["seed"]) for row in rows} != {20261030, 20261031, 20261032}:
            raise ValueError(f"Unexpected training records for {(family, order, method)}.")
        errors = [float(row["solution_l2"]) / NORMS[family] for row in rows]
        median_error = median(errors)
        mad = median(abs(error - median_error) for error in errors)
        # The archived times are in seconds for 1000 updates.
        adam_time = mean(float(row["adam_seconds"]) for row in rows)
        print(f"{family},{order},{method},{median_error:.9g},{mad:.9g},{adam_time:.9g}")


if __name__ == "__main__":
    main()
