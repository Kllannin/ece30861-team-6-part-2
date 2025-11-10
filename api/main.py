# MVP backend
from fastapi import FastAPI, UploadFile, File, Header
import uuid, os, shutil

app = FastAPI(title="MVP Registry")

STORAGE_DIR = "/storage"        # use container-level folder (not relative)
os.makedirs(STORAGE_DIR, exist_ok=True)

# In-memory "database"
ARTIFACTS = {}

# Make sure a storage folder exists

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    # Accept a .zip file, save to disk, store basic info in memory -> return a tiny JSON record.
    artifact_id = str(uuid.uuid4())

    # Save zip bytes to storage/<id>.zip
    data = await file.read()

    file_path = os.path.join(STORAGE_DIR, f"{artifact_id}.zip")

    with open(file_path, "wb") as f:
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
            "Access control track"
        ]
    }

@app.delete("/reset")
def reset_registry(x_authorization: str | None = Header(None, alias="X-Authorization")):
    """Clear registry state and wipe /storage directory."""
    # 1) Clear in-memory registry
    ARTIFACTS.clear()

    # 2) Delete /storage and all contents
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)

    # 3) Recreate clean /storage folder
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # 4) Return success
    return {"status": "reset"}

@app.post("/system/reset")
def system_reset(x_authorization: str | None = Header(None, alias="X-Authorization")):
    """Alias for system reset used by tests; mirrors /reset behavior."""
    ARTIFACTS.clear()

    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)

    os.makedirs(STORAGE_DIR, exist_ok=True)

    return {"status": "reset"}
