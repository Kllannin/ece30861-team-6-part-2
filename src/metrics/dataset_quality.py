import os
import time
import shutil
import stat
import tempfile
import subprocess
from typing import Tuple, List
import pandas as pd

def _remove_readonly(func, path, _):
    """Helper to clear readonly flag on Windows when deleting .git files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def dataset_quality(dataset_name: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Evaluates the quality of a Hugging Face dataset or a GitHub repo dataset.
    Returns a quality score and execution time.

    Args:
        dataset_name (str): Hugging Face dataset name (e.g., "imdb")
                           OR GitHub repo URL (e.g., "https://github.com/zalandoresearch/fashion-mnist").
        verbosity (int): 0 = silent, 1 = info, 2 = debug.
        log_queue (multiprocessing.Queue): Centralized log queue.

    Returns:
        Tuple[float, float]: (dataset quality score, execution time in seconds).
    """
    start_time = time.perf_counter()
    pid = os.getpid()
    score = 0.0  # default for failures
    split: str = "train"

    dataset_name = (dataset_name or "").strip()
    if not dataset_name:
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] dataset_quality: no dataset provided -> score=0.0")
        time_taken_second = time.perf_counter() - start_time
        return score, time_taken_second
    try:
        df: pd.DataFrame = pd.DataFrame()

        # Case 1: GitHub repo
        if dataset_name.startswith("http") and "github.com" in dataset_name:
            tmp_dir = tempfile.mkdtemp()
            if verbosity >= 1 and log_queue:
                log_queue.put(f"[{pid}] Cloning GitHub repo {dataset_name} into {tmp_dir}...")

            result = subprocess.run(
                ["git", "clone", "--depth", "1", dataset_name, tmp_dir],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
            
            # Try to detect a dataset file inside the repo
            candidates: List[str] = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    if f.endswith((".csv", ".json", ".parquet")):
                        candidates.append(os.path.join(root, f))
            if not candidates:
                raise ValueError(f"No supported dataset file found in GitHub repo: {dataset_name}")

            dataset_file = candidates[0]
            if verbosity >= 1 and log_queue:
                log_queue.put(f"[{pid}] Found dataset file {dataset_file}")

            if dataset_file.endswith(".csv"):
                df = pd.read_csv(dataset_file)
            elif dataset_file.endswith(".json"):
                df = pd.read_json(dataset_file, lines=True)
            elif dataset_file.endswith(".parquet"):
                df = pd.read_parquet(dataset_file)

            # Safe cleanup
            try:
                shutil.rmtree(tmp_dir, onerror=_remove_readonly)
            except Exception as e:
                if verbosity >= 1 and log_queue:
                    log_queue.put(f"[{pid}] [WARNING] Failed to cleanup {tmp_dir}: {e}")

        # Case 2: Hugging Face dataset
        else:
            if verbosity >= 1 and log_queue:
                log_queue.put(f"[{pid}] Loading dataset '{dataset_name}' (split: {split})...")
            try:
                from datasets import load_dataset  # type: ignore
            except Exception as e:
                raise ImportError("The 'datasets' package is not installed.") from e

            hf_dataset = load_dataset(dataset_name, split=split)
            df = hf_dataset.to_pandas()

        # Run quality checks
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] Dataset loaded with {len(df)} rows. Starting checks...")

        passed_checks: List[str] = []
        failed_checks: List[str] = []

        checks = {
            "row_count > 0": len(df) > 0,
            "no_missing_values": df.isnull().sum().sum() == 0,
            "no_duplicates": not df.duplicated().any(),
        }

        if "text" in df.columns:
            checks["no_empty_text"] = (df["text"].astype(str).str.strip() != "").all()

        if "label" in df.columns:
            value_counts = df["label"].value_counts(normalize=True)
            checks["balanced_labels"] = (value_counts.min() >= 0.05)

        for check, passed in checks.items():
            if passed:
                passed_checks.append(check)
            else:
                failed_checks.append(check)

        score = len(passed_checks) / len(checks) if checks else 0.0

        if verbosity >= 1 and log_queue:
            log_queue.put(
                f"[{pid}] Quality check complete. "
                f"Passed: {len(passed_checks)}/{len(checks)}. Score: {score:.2f}"
            )
        if verbosity >= 2 and failed_checks and log_queue:
            log_queue.put(f"[{pid}] [DEBUG] Failed checks: {', '.join(failed_checks)}")

    except Exception as e:
        if verbosity > 0 and log_queue:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] evaluating dataset '{dataset_name}': {e}")
        score = 0.0

    time_taken_second = time.perf_counter() - start_time
    return score, time_taken_second

if __name__ == "__main__":
    from queue import SimpleQueue
    log_queue = SimpleQueue()

    # Hugging Face test
    hf_dataset = "imdb"  # HF datasets expect just the repo name
    hf_score, hf_time = dataset_quality(hf_dataset, verbosity=1, log_queue=log_queue)
    print(f"Hugging Face dataset test ({hf_dataset}):")
    print(f"  Score: {hf_score:.2f}, Time: {hf_time:.2f}s\n")

    # GitHub test
    gh_dataset = "https://github.com/zalandoresearch/fashion-mnist"
    gh_score, gh_time = dataset_quality(gh_dataset, verbosity=1, log_queue=log_queue)
    print(f"GitHub dataset test ({gh_dataset}):")
    print(f"  Score: {gh_score:.2f}, Time: {gh_time:.2f}s\n")