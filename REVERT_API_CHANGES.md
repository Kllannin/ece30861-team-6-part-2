# How to Revert API Changes

## What Changed:
Modified `/artifact/model/{id}/rate` endpoint in `api/main.py` (around line 705-755) to:
- Call actual Phase 1 metrics via `run.py` subprocess
- Parse JSON output and return real metric scores
- Replace hardcoded placeholder values (all 0.5)

## To Revert Back to Placeholder Version:

Replace the `/artifact/model/{id}/rate` endpoint function (lines ~705-820) with:

```python
@app.get("/artifact/model/{id}/rate", tags=["baseline"])
async def get_model_rate(
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization"),
):
    """
    Dummy rating that matches the ModelRating schema exactly.
    """

    stored = ARTIFACTS.get(id)
    if not stored or stored["metadata"].get("type") != "model":
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    meta = stored["metadata"]

    # everything is fake but structurally correct
    return {
        "name": meta["name"],
        "category": "model",
        "net_score": 0.5,
        "net_score_latency": 0.01,
        "ramp_up_time": 0.5,
        "ramp_up_time_latency": 0.01,
        "bus_factor": 0.5,
        "bus_factor_latency": 0.01,
        "performance_claims": 0.5,
        "performance_claims_latency": 0.01,
        "license": 0.5,
        "license_latency": 0.01,
        "dataset_and_code_score": 0.5,
        "dataset_and_code_score_latency": 0.01,
        "dataset_quality": 0.5,
        "dataset_quality_latency": 0.01,
        "code_quality": 0.5,
        "code_quality_latency": 0.01,
        "reproducibility": 0.5,
        "reproducibility_latency": 0.01,
        "reviewedness": 0.5,
        "reviewedness_latency": 0.01,
        "tree_score": 0.5,
        "tree_score_latency": 0.01,
        "size_score": {
            "raspberry_pi": 0.5,
            "jetson_nano": 0.5,
            "desktop_pc": 0.5,
            "aws_server": 0.5,
        },
        "size_score_latency": 0.01,
    }
```

## Quick Revert Command:
```bash
git checkout api/main.py
```
(If you have the original in git)
