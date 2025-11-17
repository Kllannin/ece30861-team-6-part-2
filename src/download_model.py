
"""
download_model.py

Provides a FastAPI APIRouter that implements model download functionality
for the registry. Supports downloading full packages or specific parts.

Behavior:
- Verifies an `X-Authorization` header (placeholder check).
- Looks up model/artifact metadata in the in-memory `api.main.ARTIFACTS`.
- Supports files stored locally under `storage/<id>.zip` or in S3 when
  `s3_bucket`/`s3_key` are present in the artifact record.
- Streams bytes to the client using StreamingResponse so large files
  don't need to be loaded into memory.

This module purposely keeps S3 usage optional: if `boto3` is not
installed and the artifact references S3, an informative error is
returned.
"""

import os
import typing
from fastapi import APIRouter, Header, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Generator
import zipfile

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _HAS_BOTO3 = True
except Exception:
    boto3 = None
    _HAS_BOTO3 = False

CHUNK_SIZE = 1024 * 64

router = APIRouter()


def _verify_access(x_authorization: Optional[str]):
    # Placeholder access check: require header present and non-empty.
    if not x_authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Authorization header")


def _get_artifact_record(artifact_id: str) -> dict:
    try:
        # ARTIFACTS is defined in `api.main` (MVP). Import lazily so this
        # module can be used in other contexts without importing the app.
        from api.main import ARTIFACTS
    except Exception:
        ARTIFACTS = {}

    rec = ARTIFACTS.get(artifact_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return rec


def _stream_local_file(path: str) -> Generator[bytes, None, None]:
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found on disk")
    def _iter():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    return _iter()


def _stream_zip_member(zip_path: str, member_name: str) -> Generator[bytes, None, None]:
    """Stream a specific file from within a ZIP archive.
    Note: Caller must validate member exists before calling this function."""
    z = zipfile.ZipFile(zip_path, "r")
    try:
        with z.open(member_name, "r") as member:
            while True:
                chunk = member.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
    finally:
        z.close()


def _stream_s3_object(bucket: str, key: str) -> Generator[bytes, None, None]:
    if not _HAS_BOTO3:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="boto3 not installed on server; cannot fetch from S3")

    s3 = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"]
        while True:
            chunk = body.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error fetching from S3: {e}")


@router.get("/download/{artifact_id}")
def download_artifact(
    artifact_id: str,
    part: Optional[str] = Query(None, description="Optional path inside package to download"),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization"),
):
    """Download a full package (when `part` is omitted) or a specific
    part/file inside the package. The artifact metadata is expected to
    include either local storage info (saved as `storage/<id>.zip`) or
    `s3_bucket` and `s3_key` keys for S3-based storage.
    """
    _verify_access(x_authorization)
    rec = _get_artifact_record(artifact_id)

    # If artifact has explicit s3 info, prefer that
    if rec.get("s3_bucket") and rec.get("s3_key"):
        bucket = rec["s3_bucket"]
        base_key = rec["s3_key"].rstrip("/")

        if part:
            key = f"{base_key}/{part.lstrip('/')}"
            generator = _stream_s3_object(bucket, key)
            return StreamingResponse(generator, media_type="application/octet-stream")
        else:
            # stream the package object
            generator = _stream_s3_object(bucket, base_key)
            return StreamingResponse(generator, media_type="application/octet-stream")

    # Fallback: expect a local zip stored as storage/<id>.zip
    local_zip = os.path.join("storage", f"{artifact_id}.zip")

    if part:
        # Validate member exists BEFORE creating StreamingResponse
        if not os.path.exists(local_zip):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Archive not found")
        
        try:
            with zipfile.ZipFile(local_zip, "r") as z:
                if part not in z.namelist():
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Member '{part}' not found in archive")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid ZIP archive")
        
        # Member exists, now stream it
        gen = _stream_zip_member(local_zip, part)
        return StreamingResponse(gen, media_type="application/octet-stream")
    else:
        # stream the archive file itself
        gen = _stream_local_file(local_zip)
        return StreamingResponse(gen, media_type="application/zip")


def get_router() -> APIRouter:
    """Return the FastAPI router so callers can include it into their app."""
    return router
