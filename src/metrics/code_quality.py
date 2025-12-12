import os
import time
import subprocess
from typing import Tuple

def code_quality(github_str: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Computes the code_quality metric of a given repository by interpreting quality in
    code style and maintainability; progress is logged to the log queue.

    Cases (github_str):
        1. Empty/None -> 0.0
        2. Valid GitHub URL (starts with https://github.com/ or http://github.com/) -> 0.5
        3. Local .py file -> run pylint -> ranging 0.0 to 1.0, based on pylint score
        4. Unrecognized input -> 0.0

    Args:
        github_str (str): Code location as local .py filepath or GitHub URL
        verbosity (int): 0=silent, 1=info, 2=debug
        log_queue (multiprocessing.Queue | None): Queue to send log messages to

    Returns (score, time_taken_second):
        - score = code quality score ranging 0.0 to 1.0
        - time_taken_second = total time spent computing score 
    """
    start_time = time.perf_counter()
    pid = os.getpid()
    score = 0.0  # Default/fallback

    # Normalize input
    github_str_norm = (github_str or "").strip()
    github_str_norm_lower = github_str_norm.lower()
    
    # Case 1: Empty/None
    if not github_str_norm_lower:
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] code_quality: empty path/URL -> score=0.0")
        time_taken_second = time.perf_counter() - start_time
        return score, time_taken_second
    
    # Case 2: Valid GitHub URL
    if github_str_norm_lower.startswith("https://github.com/") or github_str_norm_lower.startswith("http://github.com/"):
        score = 0.75
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] code_quality: Github URL detected -> score=0.75")
        time_taken_second = time.perf_counter() - start_time
        return score, time_taken_second
    
    # Case 3: Local .py file
    if github_str_norm_lower.endswith(".py") and os.path.isfile(github_str_norm):
        if verbosity >= 1 and log_queue:
            log_queue.put(f"[{pid}] Running PyLint on '{os.path.basename(github_str_norm)}'...")
    
        output = ""
        try:
            # Run PyLint and capture output. check=False prevents exception on non-zero exit.
            result = subprocess.run(
                ["pylint", github_str_norm, "--score=y"],
                capture_output=True,
                text=True,
                check=False
            )
            output = result.stdout or result.stderr

            found_score = False
            # Look for the line that contains the score
            for line in (output or "").splitlines():
                if "Your code has been rated at" in line:
                    parts = line.split()
                    for part in parts:
                        if "/" in part:
                            try:
                                score_str = part.split("/")[0]
                                score = float(score_str) / 10.0 # scale to [0,1]
                                found_score = True
                                if verbosity >= 1 and log_queue:
                                    log_queue.put(
                                        f"[{pid}] Found PyLint score for '{os.path.basename(github_str_norm)}': {score*10:.2f}/10"
                                    )
                                break
                            except Exception:
                                continue # Keep searching parts
                    if found_score:
                        break
            if not found_score:
                if log_queue:
                    log_queue.put(f"[{pid}] [WARNING] Could not find PyLint score line in output for '{github_str_norm}'.")
                    if verbosity >= 2:
                        log_queue.put(f"[{pid}] [DEBUG] PyLint output for '{github_str_norm}':\n---BEGIN---\n{output}\n---END---")
        except FileNotFoundError:
            # pylint not found
            if verbosity > 0 and log_queue:
                log_queue.put(f"[{pid}] [CRITICAL ERROR] 'pylint' not found on PATH.")
            score = 0.0
        except Exception as error:
            if verbosity > 0 and log_queue:
                log_queue.put(f"[{pid}] [CRITICAL ERROR] running PyLint on '{github_str_norm}': {error}")
            if verbosity >= 2 and log_queue:
                log_queue.put(f"[{pid}] [DEBUG] PyLint output for '{github_str_norm}':\n---BEGIN---\n{output}\n---END---")
            score = 0.0
    
        time_taken_second = time.perf_counter() - start_time
        return score, time_taken_second
    
    # Case 4: Unrecognized input
    if verbosity >= 1 and log_queue:
        log_queue.put(f"[{pid}] code_quality: unsupported path '{github_str_norm}' -> score=0.0")
    time_taken_second = time.perf_counter() - start_time
    return score, time_taken_second

if __name__ == "__main__":
    score, _ = code_quality("./classes/api.py", verbosity=0, log_queue=None)
    print("PyLint code quality score:", score)