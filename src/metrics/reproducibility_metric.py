import sys
import os
import time
from typing import Tuple

# --- Import Setup ---
# This block gets the project root onto the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now we can import the function from the other file in the 'metrics' directory
from .ai_llm_generic_call import process_file_and_get_response

def reproducibility_metric(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calls an LLM to evaluate if demo code in the model card works out-of-box.
    
    Scoring:
    - 1.0: Demo code works out-of-box
    - 0.5: Demo code works with agent debugging/minor fixes
    - 0.0: Demo code doesn't work at all or is missing
    
    Args:
        filename (str): The absolute path to the README/model card file (.md or .txt).
        verbosity (int): The verbosity level (0=silent, 1=INFO, 2=DEBUG).
        log_queue (multiprocessing.Queue): The queue to send log messages to.

    Returns:
        A tuple containing:
        - The reproducibility score as a float (0.0, 0.5, or 1.0).
        - The total time spent (float).
    """
    start_time = time.time()
    pid = os.getpid()  # Get process ID for clear log messages

    instruction = """Evaluate the reproducibility of the demo code in this model card/README.

Analyze if the provided demo code or usage examples would work out-of-box. Consider:
1. Is demo/usage code present?
2. Are all necessary imports/dependencies shown?
3. Are there clear, runnable code examples?
4. Would the code work without modification?
5. Are there any obvious errors or missing information?

Respond with ONLY ONE OF THESE VALUES (nothing else):
- 1.0 if demo code is present and would work out-of-box
- 0.5 if demo code is present but would need debugging/minor fixes
- 0.0 if demo code is missing or completely non-functional

ONLY PROVIDE A SINGLE NUMBER (1.0, 0.5, or 0.0), NO OTHER TEXT:

"""

    score = 0.0  # Default to 0.0 for failure cases
    llm_response_str = None

    try:
        if verbosity >= 1:  # Informational
            log_queue.put(f"[{pid}] [INFO] Calling LLM for reproducibility check on '{os.path.basename(filename)}'...")
            
        llm_response_str = process_file_and_get_response(filename, instruction, "gemma3:1b")

        # Safely convert the LLM's string response to a float
        if llm_response_str is not None:
            raw_score = float(llm_response_str.strip())
            
            # Normalize to valid scores (0.0, 0.5, 1.0)
            if raw_score >= 0.9:
                score = 1.0
            elif raw_score >= 0.4:
                score = 0.5
            else:
                score = 0.0
                
            if verbosity >= 2:  # Debug
                log_queue.put(f"[{pid}] [DEBUG] LLM raw response: {raw_score}, normalized to: {score}")
        else:
            if verbosity >= 1:  # Informational
                log_queue.put(f"[{pid}] [WARNING] Received no response from LLM for reproducibility metric.")

    except (ValueError, TypeError):
        if verbosity >= 1:  # Informational
            log_queue.put(f"[{pid}] [WARNING] Could not convert LLM response '{llm_response_str}' to a float.")
        score = 0.0  # Ensure score is 0 on conversion failure
    except Exception as e:
        # Log any other critical error before the process terminates
        if verbosity > 0:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] in reproducibility_metric: {e}")
        raise  # Re-raise the exception to be caught by the worker

    end_time = time.time()
    time_taken = end_time - start_time
    
    if verbosity >= 1:  # Informational
        log_queue.put(f"[{pid}] [INFO] Reproducibility metric done in {time_taken:.3f}s (score={score:.2f})")
    
    return score, time_taken
