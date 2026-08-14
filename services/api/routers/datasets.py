"""Datasets, uploads and processing jobs.

Uploads are resumable because a field upload happens over whatever connection the site
has, and losing 40 GB of imagery to a dropped connection is not acceptable. A client
declares the file and its size, sends chunks in any order, and finalises; the server
verifies the total and the checksum before the upload counts as complete.

Original imagery is never modified. Files land under a content path derived from the
dataset, and processing writes derivatives elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import Dataset, Project, Role, UploadSession
from ..security import CurrentUser, require_role
from ..paths import storage_root

router = APIRouter(tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = "imagery"
    description: str = ""


class DatasetOut(BaseModel):
    id: int
    project_id: int
    name: str
    kind: str
    description: str
    file_count: int
    total_bytes: int
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadBegin(BaseModel):
    filename: str = Field(min_length=1, max_length=400)
    total_bytes: int = Field(ge=0)
    # Optional but strongly preferred: without it the server cannot tell a truncated
    # transfer from a complete one beyond checking the byte count.
    sha256: str = ""
    chunk_size: int = Field(default=8 * 1024 * 1024, ge=1024)


class UploadStatus(BaseModel):
    upload_id: str
    filename: str
    total_bytes: int
    received_bytes: int
    complete: bool
    missing_chunks: list[int]
    chunk_size: int


def _project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


def _safe_name(filename: str) -> str:
    """Reduce a client-supplied filename to a leaf name.

    A client controls this string, so a path separator or a parent reference in it
    would let an upload land anywhere on the server's filesystem.
    """
    leaf = Path(str(filename).replace("\\", "/")).name
    if not leaf or leaf in {".", ".."}:
        raise HTTPException(status_code=422, detail="Invalid filename.")
    return leaf


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(
    project_id: int, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> list[Dataset]:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.viewer)
    return list(db.scalars(select(Dataset).where(Dataset.project_id == project_id)))


@router.post("/projects/{project_id}/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(
    project_id: int, payload: DatasetCreate,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> Dataset:
    project = _project_or_404(db, project_id)
    require_role(db, user, project.organization_id, Role.pilot)

    dataset = Dataset(project_id=project_id, **payload.model_dump())
    db.add(dataset)
    db.flush()
    record(db, action="dataset_created", user_id=user.id,
           organization_id=project.organization_id, resource=f"dataset:{dataset.id}",
           detail={"name": dataset.name})
    db.commit()
    return dataset


# ---------------------------------------------------------------------------
# resumable upload
# ---------------------------------------------------------------------------


def _upload_dir(dataset_id: int, upload_id: str) -> Path:
    path = storage_root() / "uploads" / str(dataset_id) / upload_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dataset_dir(dataset_id: int) -> Path:
    path = storage_root() / "datasets" / str(dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expected_chunks(total_bytes: int, chunk_size: int) -> int:
    if total_bytes <= 0:
        return 0
    return (total_bytes + chunk_size - 1) // chunk_size


def _received(session: UploadSession) -> tuple[set[int], int]:
    directory = _upload_dir(session.dataset_id, session.upload_id)
    indices: set[int] = set()
    total = 0
    for part in directory.glob("*.part"):
        try:
            indices.add(int(part.stem))
        except ValueError:
            continue
        total += part.stat().st_size
    return indices, total


@router.post("/datasets/{dataset_id}/uploads", response_model=UploadStatus, status_code=201)
def begin_upload(
    dataset_id: int, payload: UploadBegin,
    user: CurrentUser, db: Annotated[Session, Depends(get_db)],
) -> UploadStatus:
    dataset = _dataset_or_404(db, dataset_id)
    project = _project_or_404(db, dataset.project_id)
    require_role(db, user, project.organization_id, Role.pilot)

    import uuid

    session = UploadSession(
        upload_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        filename=_safe_name(payload.filename),
        total_bytes=payload.total_bytes,
        chunk_size=payload.chunk_size,
        sha256=payload.sha256.lower().strip(),
        created_by=user.id,
    )
    db.add(session)
    db.commit()

    return UploadStatus(
        upload_id=session.upload_id, filename=session.filename,
        total_bytes=session.total_bytes, received_bytes=0, complete=False,
        missing_chunks=list(range(_expected_chunks(payload.total_bytes, payload.chunk_size))),
        chunk_size=session.chunk_size,
    )


@router.put("/uploads/{upload_id}/chunks/{index}", response_model=UploadStatus)
async def upload_chunk(
    upload_id: str, index: int, user: CurrentUser,
    db: Annotated[Session, Depends(get_db)], chunk: UploadFile = File(...),
) -> UploadStatus:
    """Accept one chunk. Chunks may arrive in any order and may be re-sent."""
    session = db.scalar(select(UploadSession).where(UploadSession.upload_id == upload_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found.")
    dataset = _dataset_or_404(db, session.dataset_id)
    project = _project_or_404(db, dataset.project_id)
    require_role(db, user, project.organization_id, Role.pilot)

    expected = _expected_chunks(session.total_bytes, session.chunk_size)
    if index < 0 or (expected and index >= expected):
        raise HTTPException(status_code=422, detail=f"Chunk index out of range (0..{expected - 1}).")

    target = _upload_dir(session.dataset_id, upload_id) / f"{index}.part"
    body = await chunk.read()
    target.write_bytes(body)

    indices, received = _received(session)
    return UploadStatus(
        upload_id=upload_id, filename=session.filename, total_bytes=session.total_bytes,
        received_bytes=received, complete=len(indices) == expected,
        missing_chunks=sorted(set(range(expected)) - indices), chunk_size=session.chunk_size,
    )


@router.get("/uploads/{upload_id}", response_model=UploadStatus)
def upload_status(
    upload_id: str, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> UploadStatus:
    """What is still missing, so a resumed client sends only that."""
    session = db.scalar(select(UploadSession).where(UploadSession.upload_id == upload_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found.")
    dataset = _dataset_or_404(db, session.dataset_id)
    project = _project_or_404(db, dataset.project_id)
    require_role(db, user, project.organization_id, Role.viewer)

    expected = _expected_chunks(session.total_bytes, session.chunk_size)
    indices, received = _received(session)
    return UploadStatus(
        upload_id=upload_id, filename=session.filename, total_bytes=session.total_bytes,
        received_bytes=received, complete=len(indices) == expected,
        missing_chunks=sorted(set(range(expected)) - indices), chunk_size=session.chunk_size,
    )


@router.post("/uploads/{upload_id}/complete")
def complete_upload(
    upload_id: str, user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    """Assemble the chunks, verify, and only then count the file as uploaded.

    Verification is the point of this endpoint: a byte count catches truncation, and
    the checksum catches corruption. Without both, a damaged capture would be
    discovered only when reconstruction failed days later.
    """
    session = db.scalar(select(UploadSession).where(UploadSession.upload_id == upload_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found.")
    dataset = _dataset_or_404(db, session.dataset_id)
    project = _project_or_404(db, dataset.project_id)
    require_role(db, user, project.organization_id, Role.pilot)

    expected = _expected_chunks(session.total_bytes, session.chunk_size)
    indices, received = _received(session)
    missing = sorted(set(range(expected)) - indices)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Upload incomplete: {len(missing)} chunk(s) missing, first is {missing[0]}.",
        )

    directory = _upload_dir(session.dataset_id, upload_id)
    destination = _dataset_dir(session.dataset_id) / session.filename
    digest = hashlib.sha256()
    written = 0
    with destination.open("wb") as handle:
        for index in range(expected):
            part = (directory / f"{index}.part").read_bytes()
            digest.update(part)
            handle.write(part)
            written += len(part)

    actual = digest.hexdigest()
    if written != session.total_bytes:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=f"Size mismatch: declared {session.total_bytes} bytes, assembled {written}.",
        )
    if session.sha256 and actual != session.sha256:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail="Checksum mismatch: the assembled file does not match the declared sha256.",
        )

    shutil.rmtree(directory, ignore_errors=True)
    session.completed_at = datetime.now(timezone.utc)
    session.actual_sha256 = actual
    dataset.file_count += 1
    dataset.total_bytes += written

    record(db, action="upload_completed", user_id=user.id,
           organization_id=project.organization_id, resource=f"dataset:{dataset.id}",
           detail={"filename": session.filename, "bytes": written, "sha256": actual})
    db.commit()

    return {
        "upload_id": upload_id, "filename": session.filename, "bytes": written,
        "sha256": actual, "checksum_verified": bool(session.sha256),
        "path": str(destination), "dataset_file_count": dataset.file_count,
    }
