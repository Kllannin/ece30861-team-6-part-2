from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

artifacts = {}  # (artifact_type, id) -> artifact data

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tracks")
def get_tracks():
    return {
        "plannedTracks": [
            "Performance track",
            # "Access control track",  # add if/when you want
        ]
    }

@app.delete("/reset")
def reset_registry(x_authorization: str = Header(..., alias="X-Authorization")):
    artifacts.clear()
    return {"status": "reset"}
