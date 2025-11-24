import os
import time
from typing import Tuple

def dataset_and_code_present(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calculates a score based on the presence of dataset keywords in provided README text.
    Verbosity is controlled by the passed-in argument (0=silent, 1=INFO, 2=DEBUG).
    """
    pid = os.getpid()
    start_time = time.perf_counter()
    score = 0.0  # Default score

    try:
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] [INFO] Starting dataset-in-readme check...")

        # Read file content if a file path is provided
        readme_text = ""
        if filename and os.path.isfile(filename):
            try:
                with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                    readme_text = f.read().lower()
            except OSError:
                # Fall back to using the filename string instead
                readme_text = filename.lower()
        else:
            readme_text = (filename or "").lower()

        # Check for these datasets (add more if needed)
        dataset_hosts = [
            "huggingface.co/datasets",
            "kaggle.com/datasets",
            "roboflow.com",
            "drive.google.com"
        ]
        dataset_keywords = [
            "dataset",
            "datasets",
            "data set",
            "training data",
            "download data"
        ]

        has_dataset = any(host in readme_text for host in dataset_hosts) or \
                      any(kw in readme_text for kw in dataset_keywords)
        
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] [INFO] Dataset mention found in README: {has_dataset}")
        
        if verbosity >= 2 and has_dataset and log_queue:
            found_hosts = [host for host in dataset_hosts if host in readme_text]
            found_kws = [kw for kw in dataset_keywords if kw in readme_text]
            if found_hosts:
                log_queue.put(f"[{pid}] [DEBUG] Found dataset hosts: {', '.join(found_hosts)}")
            if found_kws:
                log_queue.put(f"[{pid}] [DEBUG] Found dataset keywords: {', '.join(found_kws)}")

        # Score based on results
        if has_dataset:
            score = 1.0
        else:
            score = 0.0

    except Exception as e:
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] calculating dataset_in_readme metric: {e}")
        score = 0.0

    time_taken = time.perf_counter() - start_time

    if verbosity >= 1 and log_queue:
        log_queue.put(f"[{pid}] [INFO] Finished calculation. Score={score:.2f}, Time={time_taken:.3f}s")

    return score, time_taken
