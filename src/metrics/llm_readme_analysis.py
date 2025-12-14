import json
import os
import time
import re
import requests
from typing import Any, Dict, Tuple

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

def ollama_analyze_readme(readme_text: str) -> Dict[str, Any]:
    """Analyze README text with Ollama and return a structured JSON assessment."""
    readme_text = (readme_text or "")[:12000]

     # Build prompt for JSON formatted readme analysis via Ollama 3.2:3b
    json_request = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        "stream": False,
        "format": "json",
        "prompt": (
            "Return ONLY valid JSON. No markdown.\n"
            "Output keys exactly: summary, claims, has_benchmarks, risk_flags, confidence_score, confidence_justification.\n"
            "summary: 1-2 sentences.\n"
            "claims: array of strings (NOT JSON-encoded strings) describing performance/reproducibility claims.\n"
            "has_benchmarks: boolean.\n"
            "risk_flags: only include risks explicitly in the README; otherwise return an empty array.\n"
            "confidence_score: a FLOAT 0.0..1.0.\n"
            "confidence_justification: 1-2 sentences justifying the score given based on what evidence was found (or what was missing).\n\n"
            f"README:\n{readme_text}\n"
        ),
        "options": {
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.1")),
            "num_predict": int(os.getenv("LLM_MAX_TOKENS", "350")),
        },
    } 

    try:
        # Generate Ollama response
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=json_request, timeout=20)
        r.raise_for_status()
        data = r.json()

        # Collect response as raw string
        response_raw = data.get("response", "")
        try:
            # Attempt to extract JSON object from raw string
            return json.loads(response_raw)
        except (json.JSONDecodeError, TypeError):
            # Fallback to regex search for JSON object extraction
            match = re.search(r'\{.*\}', str(response_raw), re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
    
    except requests.RequestException as e:
        # Network/timeout error
        return {
            "summary": "Error: Could not connect to Ollama.",
            "claims": [],
            "has_benchmarks": False,
            "risk_flags": [f"Connection failed: {type(e).__name__}"],
            "confidence_score": 0.0,
            "confidence_justification": "N/A"
        }
    
    except Exception as e:
        # JSON decoding error
        return {
            "summary": "Error: LLM response contains no valid JSON.",
            "claims": [],
            "has_benchmarks": False,
            "risk_flags": [f"JSON decoding failed: {type(e).__name__}"],
            "confidence_score": 0.0,
            "confidence_justification": "N/A"
        }

def llm_readme_analysis(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """Run the LLM README analysis: load README, call analysis, log results, and record latency."""
    start = time.perf_counter()
    pid = os.getpid()

    if not filename or not os.path.isfile(filename):
        if verbosity >= 1:
            log_queue.put(f"[{pid}] README missing: skipping LLM analysis.")
        return 0.0, time.perf_counter() - start
    
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        readme_text = f.read()

    analysis = ollama_analyze_readme(readme_text)

    if verbosity >= 1:
        log_queue.put(
            f"[LLM README ANALYSIS] File={filename}\n"
            + json.dumps(analysis, indent=2, ensure_ascii=False)
        )

    confidence = analysis.get("confidence_score", 0.0)

    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0

    return score, time.perf_counter() - start