import json
import sys

_DEVICE_KEYS = ("raspberry_pi", "jetson_nano", "desktop_pc", "aws_server")

def _normalize_size_score(val):
    if isinstance(val, dict) and all(k in val for k in _DEVICE_KEYS):
        return {k: float(val[k]) for k in _DEVICE_KEYS}
    try:
        f = float(val)
        return {k: f for k in _DEVICE_KEYS}
    except Exception:
        return {k: 0.0 for k in _DEVICE_KEYS}


def build_model_output(
    name,
    category,
    scores,
    latency
):
    output = {
    "name":name,
    "category":category.upper(),
    "net_score":scores.get("net_score", 0.00),
    "net_score_latency":latency.get("net_score_latency", 0),
    "ramp_up_time":scores.get("rampup_time_metric", 0.00),
    "ramp_up_time_latency":latency.get("rampup_time_metric", 0),
    "bus_factor":scores.get("bus_factor_metric", 0.00),
    "bus_factor_latency":latency.get("bus_factor_metric", 0),
    "performance_claims":scores.get("performance_claims_metric", 0.00),
    "performance_claims_latency":latency.get("performance_claims_metric", 0),
    "license":scores.get("calculate_license_score", 0.00),
    "license_latency":latency.get("calculate_license_score", 0),
    "size_score": _normalize_size_score(scores.get("calculate_size_score", 0.00)),
    "size_score_latency":latency.get("calculate_size_score", 0),
    "dataset_and_code_score":scores.get("dataset_and_code_present", 0.00),
    "dataset_and_code_score_latency":latency.get("dataset_and_code_present", 0),
    "dataset_quality":scores.get("dataset_quality", 0.00),
    "dataset_quality_latency":latency.get("dataset_quality", 0),
    "code_quality":scores.get("code_quality", 0.00),
    "code_quality_latency":latency.get("code_quality", 0),
    "reproducibility":scores.get("reproducibility_metric", 0.00),
    "reproducibility_latency":latency.get("reproducibility_metric", 0),
    "reviewedness":scores.get("reviewedness_metric", 0.00),
    "reviewedness_latency":latency.get("reviewedness_metric", 0),
    "tree_score":scores.get("treescore_metric", 0.00),
    "tree_score_latency":latency.get("treescore_metric", 0),
}
    #print to stdout
    sys.stdout.write(json.dumps(output) + "\n")

# testing
if __name__ == "__main__":
    scores = {
        "net_score": 0.82,
        "rampup_time_metric": 0.75,
        "bus_factor_metric": 0.60,
        "performance_claims_metric": 0.80,
        "calculate_license_score": 1.00,
        "calculate_size_score": 0.90,
        "dataset_and_code_present": 0.90,
        "dataset_quality": 0.85,
        "code_quality": 0.70,
    }

    latency = {
        "net_score_latency": 1001,
        "rampup_time_metric": 123,
        "bus_factor_metric": 88,
        "performance_claims_metric": 110,
        "calculate_license_score": 95,
        "calculate_size_score": 200,
        "dataset_and_code_present": 130,
        "dataset_quality": 115,
        "code_quality": 140,
    }

    build_model_output(
        name="google/gemma-3-270m",
        category="MODEL",
        scores=scores,
        latency=latency,
    )