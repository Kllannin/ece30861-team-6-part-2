# MVP backend
from fastapi import FastAPI, UploadFile, File, Header
import uuid, os

app = FastAPI(title="MVP Registry")

# In-memory "database"
ARTIFACTS = {}

# Make sure a storage folder exists
os.makedirs("storage", exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    # Accept a .zip file, save to disk, store basic info in memory -> return a tiny JSON record.
    artifact_id = str(uuid.uuid4())

    # Save zip bytes to storage/<id>.zip
    data = await file.read()
    with open(f"storage/{artifact_id}.zip", "wb") as f:
        f.write(data)

    record = {
        "id": artifact_id,
        "filename": file.filename,
        "net_score": None # to be implemented
    }
    ARTIFACTS[artifact_id] = record
    return record

@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    # Return the JSON record if found; otherwise report not found.
    rec = ARTIFACTS.get(artifact_id)
    if rec is None:
        return {"error": "not found"}
    return rec


@app.get("/tracks")
def get_tracks():
    return {
        "plannedTracks": [
            "Performance track",
            # Add "Access control track" here later if you decide to do it:
            # "Access control track",
        ]
    }

@app.delete("/reset")
def reset_registry(x_authorization: str = Header(..., alias="X-Authorization")):
    # You can later check x_authorization value if you want,
    # but for MVP we just require that it exists.

    ARTIFACTS.clear()  # wipe in-memory registry

    return {"status": "reset"}
