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

def rampup_time_metric(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calls an LLM to rate the "ramp-up" time for a model based on its readme, logging to a queue.

    Args:
        filename (str): The absolute path to the input file (.md or .txt).
        verbosity (int): The verbosity level (0=silent, 1=INFO, 2=DEBUG).
        log_queue (multiprocessing.Queue): The queue to send log messages to.

    Returns:
        A tuple containing:
        - The score from the LLM as a float (0.0 on error).
        - The total time spent (float).
    """
    start_time = time.time()
    pid = os.getpid() # Get process ID for clear log messages

    instruction = """Rate how easy it would be for a new engineer to start using this model (0.0-1.0):
- 1.0 = Excellent documentation with examples, quick start guide, clear usage instructions
- 0.8-0.9 = Good documentation with examples or usage guide
- 0.6-0.7 = Basic documentation with some usage info
- 0.4-0.5 = Minimal documentation but model name/purpose is clear
- 0.2-0.3 = Very sparse information
- 0.0 = No useful documentation

Look for: installation steps, usage examples, code snippets, quick start sections, clear explanations.
Be generous - if there's ANY documentation, give at least 0.6.

ONLY RESPOND WITH A SINGLE NUMBER (e.g., 0.85). NO OTHER TEXT.\n\n"""

    score = 0.6  # Default to 0.6 for failure cases (be lenient)
    llm_response_str = None
    
    try:
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [INFO] Calling LLM for ramp-up time on '{os.path.basename(filename)}'...")

        llm_response_str = process_file_and_get_response(filename, instruction, "gemma3:1b")

        # Safely convert the LLM's string response to a float
        if llm_response_str is not None:
            score = float(llm_response_str.strip())
            if verbosity >= 2: # Debug
                log_queue.put(f"[{pid}] [DEBUG] Successfully converted LLM response to score: {score}")
        else:
            if verbosity >= 1: # Informational
                log_queue.put(f"[{pid}] [WARNING] Received no response from LLM for ramp-up time metric.")

    except (ValueError, TypeError):
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [WARNING] Could not convert LLM response '{llm_response_str}' to a float.")
        score = 0.6 # Ensure score is 0.6 on conversion failure (be lenient)
    except Exception as e:
        # Log any other critical error before the process terminates
        if verbosity >0:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] in rampup_time_metric: {e}")
        raise # Re-raise the exception to be caught by the worker
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    return score, time_taken

