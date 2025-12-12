import os
import time
from typing import Tuple, Dict

def calculate_size_score(model_size_bytes: int, verbosity: int, log_queue) -> Tuple[dict, float]:
    """
    Calculates a score based on the size of a model file, logging to a queue.
    Verbosity is controlled by the passed-in argument (0=silent, 1=INFO, 2=DEBUG).
    """
    pid = os.getpid()
    start_time = time.perf_counter()
    
    scores: Dict[str, float] = {}  # <-- ensure it's always defined
    
    try:
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [INFO] Starting size score calculation for model of {model_size_bytes} bytes...")

        # Handle edge cases: None, 0, or negative sizes
        if model_size_bytes is None or model_size_bytes <= 0:
            if verbosity >= 1:
                log_queue.put(f"[{pid}] [WARNING] Invalid model size ({model_size_bytes}), assuming small model for default scores")
            # Assume it's a very small model if size is unknown - give higher scores
            # This assumes unknown size models are likely small/optimized
            scores = {
                "raspberry_pi": 1.0,
                "jetson_nano": 1.0,
                "desktop_pc": 1.0,
                "aws_server": 1.0
            }
            time_taken = time.perf_counter() - start_time
            return scores, time_taken

        size_gb = model_size_bytes / (1024 * 1024 * 1024)
        
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [INFO] Model size: {size_gb:.2f} GB")
        
        scores: Dict[str, float] = {}
        
        # Raspberry Pi
        if size_gb <= 0.1: # 100MB
            scores["raspberry_pi"] = 1.0
        elif size_gb <= 0.5: # 500MB
            scores["raspberry_pi"] = 0.5
        else:
            scores["raspberry_pi"] = 0.0

        if verbosity >= 2: # Debug
            log_queue.put(f"[{pid}] [DEBUG] Raspberry Pi score: {scores['raspberry_pi']}")

        # Jetson Nano
        if size_gb <= 0.5: # 500MB
            scores["jetson_nano"] = 1.0
        elif size_gb <= 2:
            scores["jetson_nano"] = 0.5
        else:
            scores["jetson_nano"] = 0.0

        if verbosity >= 2: # Debug
            log_queue.put(f"[{pid}] [DEBUG] Jetson Nano score: {scores['jetson_nano']}")

        # Desktop PC
        if size_gb <= 5:
            scores["desktop_pc"] = 1.0
        elif size_gb <= 10:
            scores["desktop_pc"] = 0.5
        else:
            scores["desktop_pc"] = 0.0

        if verbosity >= 2: # Debug
            log_queue.put(f"[{pid}] [DEBUG] Desktop PC score: {scores['desktop_pc']}")

        # AWS Server
        scores["aws_server"] = 1.0

        if verbosity >= 2: # Debug
            log_queue.put(f"[{pid}] [DEBUG] AWS Server score: {scores['aws_server']}")
        
        # Final score is the average of all platform scores
        final_score = sum(scores.values()) / len(scores) if scores else 0.0

    except Exception as e:
        if verbosity >= 1:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] calculating size score: {e}")
        final_score = 0.0

    time_taken = time.perf_counter() - start_time 

    if verbosity >= 1: # Informational
        log_queue.put(f"[{pid}] [INFO] Finished calculation. Average Score={final_score:.2f}, Time={time_taken:.4f}s")

    return scores, time_taken

def main():
    """
    Main function for direct testing of this metric.
    This is not called when the script is run by the main metric_caller.
    """
    print("This script is intended to be called as a metric by the main runner.")
    print("To test it, you would need to create a dummy file and a dummy queue.")

if __name__ == "__main__":
    main()


'''
# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example model sizes in bytes
    model_sizes = [
        ("small_model", 50*1024*1024),      # 50MB
        ("medium_model", 600*1024*1024),    # 600MB
        ("large_model", 8*1024*1024*1024),  # 8GB
        ("huge_model", 12*1024*1024*1024),  # 12GB
    ]

    for name, size in model_sizes:
        scores, latency = calculate_size_score(size)
        print(f"{name} -> Scores: {scores}, Latency: {latency} ms")
'''
