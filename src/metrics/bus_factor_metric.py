'''import os
import time
import re
from typing import Tuple

def bus_factor_metric(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calculates a proxy for the bus factor score by searching for contributor
    information within a README file's text.
    NOTE: This is an approximation and less accurate than the API-based method.
    Verbosity is controlled by the passed-in argument (0=silent, 1=INFO, 2=DEBUG).

    Args:
        readme_content (str): The full text content of the README file.
        verbosity (int): The verbosity level (0, 1, or 2).
        log_queue (multiprocessing.Queue): The queue for centralized logging.

    Returns:
        A tuple containing:
        - The bus factor score (1.0 if contributor info is mentioned, 0.0 otherwise).
        - The total time spent (float).
    """
    pid = os.getpid()
    start_time = time.perf_counter()
    score = 0.0

    try:
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [INFO] Starting bus factor check based on README content...")

        readme_text = filename.lower()

        # Keywords that suggest contributor information is present. This can be expanded.
        contributor_keywords = [
            "contributor", "contributors", "author", "authors",
            "team", "maintainer", "maintained by", "developed by", "credits"
        ]

        # Check if any keywords are present. This is a simple proxy for the bus factor.
        found_mention = any(keyword in readme_text for keyword in contributor_keywords)

        if found_mention:
            score = 1.0
            if verbosity >= 1: # Informational
                log_queue.put(f"[{pid}] [INFO] Found mention of contributors in README -> Score = 1.0")
            if verbosity >= 2: # Debug
                found_kws = [kw for kw in contributor_keywords if kw in readme_text]
                log_queue.put(f"[{pid}] [DEBUG] Found keywords: {', '.join(found_kws)}")
        else:
            score = 0.0
            if verbosity >= 1: # Informational
                log_queue.put(f"[{pid}] [INFO] No mention of contributors found in README -> Score = 0.0")

    except Exception as e:
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [CRITICAL ERROR] calculating bus factor from README: {e}")
        score = 0.0

    time_taken = time.perf_counter() - start_time

    if verbosity >= 1: # Informational
        log_queue.put(f"[{pid}] [INFO] Finished calculation. Score={score:.2f}, Time={time_taken:.3f}s")

    return score, time_taken'''

from typing import Tuple
import os, time, re

def bus_factor_metric(readme_path: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """Return (score in [0,1], elapsed_seconds)."""
    pid = os.getpid()
    t0 = time.perf_counter()
    score = 0.0

    try:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [INFO] Bus factor: reading {readme_path!r}")

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()

        keywords = ["contributor", "contributors", "author", "authors",
                    "team", "maintainer", "maintained by", "developed by", "credits"]

        score = 1.0 if any(k in text for k in keywords) else 0.0

    except Exception as e:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] bus factor: {e}")
        score = 0.0

    dt = time.perf_counter() - t0
    if verbosity >= 1:
        log_queue.put(f"[{pid}] [INFO] Bus factor done in {dt:.3f}s (score={score:.2f})")
    return score, dt
