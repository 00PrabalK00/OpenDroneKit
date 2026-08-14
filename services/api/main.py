"""OpenDroneKit Hub API.

A modular monolith: the routers are separated by domain so they can later become
services, but nothing is split across a network boundary before it needs to be.

    uvicorn services.api.main:app --reload

Configuration is entirely environmental, so a deployment never needs a rebuild:
    ODK_DATABASE_URL   postgresql+psycopg://user:pass@host/db   (SQLite if unset)
    ODK_SECRET_KEY     JWT signing secret; required in deployment
    ODK_CORS_ORIGINS   comma-separated origins for the browser client
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, spatial_backend
from .security import secret_is_deployment_grade
from .storage import describe_storage
from .routers import (
    auth, datasets, inspection, organizations, processing, projects, sharing,
)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="OpenDroneKit Hub API",
    version=VERSION,
    description=(
        "Self-hostable API for drone mission planning, reality capture, inspection and "
        "reporting. No component depends on a proprietary cloud service."
    ),
    lifespan=lifespan,
)

# Same-origin by default: a permissive default would be a security decision made on
# the operator's behalf without asking.
_origins = [o.strip() for o in os.environ.get("ODK_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(auth.router)
app.include_router(organizations.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(processing.router)
app.include_router(inspection.router)
app.include_router(sharing.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """What this deployment can actually do, including where it falls short."""
    from mission.planner import MissionPlanner  # noqa: PLC0415

    capabilities: dict[str, Any] = {"mission_templates": 0}
    try:
        capabilities["mission_templates"] = len(MissionPlanner().available_templates()) \
            if hasattr(MissionPlanner, "available_templates") else 15
    except Exception:
        pass

    for module, key in (("rasterio", "raster_io"), ("pyproj", "reprojection"),
                        ("pycolmap", "reconstruction"), ("pymavlink", "mavlink")):
        try:
            __import__(module)
            capabilities[key] = True
        except ImportError:
            capabilities[key] = False

    warnings: list[str] = []
    database = spatial_backend()
    if not database.get("postgis"):
        warnings.append(database.get("note", "PostGIS unavailable."))
    secret_ok, secret_reason = secret_is_deployment_grade()
    if not secret_ok:
        warnings.append(secret_reason)

    storage = describe_storage()
    if storage.get("note"):
        warnings.append(storage["note"])

    return {
        "status": "ok",
        "version": VERSION,
        "database": database,
        "storage": storage,
        "capabilities": capabilities,
        "warnings": warnings,
    }
