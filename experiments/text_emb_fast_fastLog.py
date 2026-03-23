from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess
import sys

from experiment import DEFAULT_DATASETS


MAX_CPUS = min(4, os.cpu_count() or 1)


def _run_dataset(dataset: str, results_dir: Path) -> tuple[str, int]:
    dataset_name = Path(dataset).stem
    output_file = results_dir / f"{dataset_name}_comparison_results.csv"

    cmd = [
        sys.executable,
        "experiments/experiment.py",
        "--datasets",
        dataset,
        "--output",
        str(output_file),
    ]

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    print(f"[START] {dataset}")
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode == 0:
        print(f"[DONE ] {dataset} -> {output_file}")
    else:
        print(f"[FAIL ] {dataset} (exit code {completed.returncode})")

    return dataset, completed.returncode


def main() -> None:
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_CPUS) as executor:
        futures = [executor.submit(_run_dataset, dataset, results_dir) for dataset in DEFAULT_DATASETS]
        for future in as_completed(futures):
            dataset, returncode = future.result()
            if returncode != 0:
                failures.append(dataset)

    if failures:
        print("\nDatasets with failures:")
        for dataset in failures:
            print(f"- {dataset}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

