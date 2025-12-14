import os
import time
import stat
from typing import Tuple, List
import pandas as pd
import requests

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
    score = 0.5  # Default neutral score
    split: str = "train[:500]" # Use a small slice to improve latency

    dataset_name = (dataset_name or "").strip()
    if not dataset_name:
        score = 0.5 # No dataset to inspect, stay neutral -> 0.5
        if verbosity >= 1 and log_queue:
            log_queue.put(
                f"[{pid}] dataset_quality: no dataset provided; "
                f"using default score={score:.2f}"
            )
        
        time_taken_second = time.perf_counter() - start_time
        return score, time_taken_second
    
    try:
        df: pd.DataFrame = pd.DataFrame()

        # Case 1: URLs (must not use git CLI, handle GitHub via HTTP API)
        if dataset_name.startswith("http"):
            if "github.com" in dataset_name:
                try:
                    parts = dataset_name.rstrip("/").split("/")
                    owner, repo = parts[3], parts[4]

                    # Try common default branches in order
                    for ref in ("main", "master"):
                        r = requests.get(
                            f"https://api.github.com/repos/{owner}/{repo}/contents?ref={ref}",
                            timeout=5,
                        )
                        if r.status_code == 200 and isinstance(r.json(), list):
                            # Pick the first small CSV file
                            for it in r.json():
                                name = (it.get("name") or "").lower()
                                if it.get("type") == "file" and name.endswith(".csv"):
                                    df = pd.read_csv(it.get("download_url"), nrows=500)
                                    break
                            break
                except Exception as e:
                    if verbosity >= 1 and log_queue:
                        log_queue.put(f"[{pid}] dataset_quality: GitHub inspection failed: {e}")

            # No dataset to inspect, stay neutral -> 0.5
            if df.empty:
                if verbosity >= 1 and log_queue:
                    log_queue.put(
                        f"[{pid}] dataset_quality: URL detected ('{dataset_name}'). "
                        "No small CSV found via API; returning score 0.5."
                    )
                time_taken_second = time.perf_counter() - start_time
                return 0.5, time_taken_second
            
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

        # Optional text specific check
        if "text" in df.columns:
            checks["no_empty_text"] = (df["text"].astype(str).str.strip() != "").all()

        # Optional label balance check
        if "label" in df.columns:
            value_counts = df["label"].value_counts(normalize=True)
            if not value_counts.empty:
                checks["balanced_labels"] = (value_counts.min() >= 0.05)

        # Evaluate all checks and record results
        for check, passed in checks.items():
            if passed:
                passed_checks.append(check)
            else:
                failed_checks.append(check)

        # Quality score = fraction of checks passed with generous minimum
        score = len(passed_checks) / len(checks) if checks else 0.0
        score = max(score, 0.6)  # Higher minimum score

        if verbosity >= 1 and log_queue:
            log_queue.put(
                f"[{pid}] Quality check complete. "
                f"Passed: {len(passed_checks)}/{len(checks)}. Score: {score:.2f}"
            )
        if verbosity >= 2 and failed_checks and log_queue:
            log_queue.put(f"[{pid}] [DEBUG] Failed checks: {', '.join(failed_checks)}")

    except Exception as e:
        if verbosity > 1 and log_queue:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] evaluating dataset '{dataset_name}': {e}")
        score = 0.5 # On error assign neutral score -> 0.5

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