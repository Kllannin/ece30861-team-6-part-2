# MVP backend
from fastapi import FastAPI, UploadFile, File, Header, Query, HTTPException, Body, Request
from typing import Optional, List, Dict, Any
import uuid, os, shutil, re
from urllib.parse import urlparse
from src.download_model import get_router as get_download_router
from pydantic import BaseModel

# Pydantic models for artifact operations
class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: str

class ArtifactData(BaseModel):
    url: str
    download_url: str = None

class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData

class ArtifactQuery(BaseModel):
    name: str = "*"
    types: List[str] = []

app = FastAPI(title="MVP Registry")

# Include the download router
download_router = get_download_router()
app.include_router(download_router)

# In-memory "database"
ARTIFACTS = {}

# Make sure a storage folder exists
os.makedirs("storage", exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_model(file: UploadFile = File(...), s3_bucket: str | None = Query(None), s3_key: str | None = Query(None)):
    # Accept a .zip file, save to disk, store basic info in memory -> return a tiny JSON record.
    # Optionally store S3 metadata if provided (bucket and key for artifact storage on S3).
    artifact_id = str(uuid.uuid4())

    # Save zip bytes to storage/<id>.zip (local fallback)
    data = await file.read()
    with open(f"storage/{artifact_id}.zip", "wb") as f:
        f.write(data)

    record = {
        "id": artifact_id,
        "filename": file.filename,
        "net_score": None # to be implemented
    }
    
    # If S3 metadata is provided, store it in the artifact record
    if s3_bucket and s3_key:
        record["s3_bucket"] = s3_bucket
        record["s3_key"] = s3_key
    
    ARTIFACTS[artifact_id] = record
    return record

@app.get("/artifacts")
def list_artifacts(
    regex: str | None = Query(None, description="Optional regex over filename"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Enumerate artifacts:
    - If regex is given, filter by regex over filename (later also model card text).
    - Support offset + limit so it scales to millions of models.
    """
    items = list(ARTIFACTS.values())

    if regex:
        pattern = re.compile(regex, re.IGNORECASE)
        items = [a for a in items if pattern.search(a["filename"])]

    total = len(items)
    page = items[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page,
    }

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
            #"Access control track"
        ]
    }

@app.delete("/reset")
def reset_registry(x_authorization: str | None = Header(None, alias="X-Authorization")):
    # You can later check x_authorization value if you want,
    # but for MVP we just require that it exists.

    # 1) Clear in-memory artifacts
    ARTIFACTS.clear()

    # 2) Delete the storage directory if it exists
    if os.path.exists("storage"):
        shutil.rmtree("storage")

    # 3) Recreate an empty storage directory
    os.makedirs("storage", exist_ok=True)


    return {"status": "reset"}
