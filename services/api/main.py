"""OpenDroneKit Hub API.

A modular monolith: the routers are separated by domain so they can later become
services, but nothing is split across a network boundary before it needs to be.

    uvicorn services.api.main:app --reload

Configuration is entirely environmental, so a deployment never needs a rebuild:
    ODK_DATABASE_URL   postgresql+psycopg://user:pass@host/db   (SQLite if unset)
    ODK_SECRET_KEY     JWT signing secret; required in deployment
    ODK_CORS_ORIGINS   comma-separated origins for the browser client
    ODK_CORS_LOOPBACK  "0" to stop allowing http(s)://127.0.0.1|localhost:<any port>
    ODK_JOB_BACKEND    "thread" (default) or "celery" for processing jobs
    ODK_AI_WORKER      "inline" (default) or "external" for AI jobs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .db import init_db
from .observability import (
    LOGGER, METRICS, ObservabilityMiddleware, configure_logging,
    database_readiness, observability_contract, storage_readiness,
)
from .security import secret_is_deployment_grade
from .routers import (
    annotations, auth, datasets, events, fleet, inspection, organizations, processing, projects, resources,
    sharing, tiles,
)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    init_db()
    # AI jobs accepted by a previous inline process would otherwise wait forever.
    from .ai_jobs import resume_pending  # noqa: PLC0415

    resumed = resume_pending()
    if resumed:
        LOGGER.info("resumed pending AI jobs", extra={"event": "ai.resumed", "jobs": resumed})
    LOGGER.info("API started", extra={"event": "api.started"})
    yield
    LOGGER.info("API stopped", extra={"event": "api.stopped"})


app = FastAPI(
    title="OpenDroneKit Hub API",
    version=VERSION,
    description=(
        "Self-hostable API for drone mission planning, reality capture, inspection and "
        "reporting. No component depends on a proprietary cloud service."
    ),
    lifespan=lifespan,
)

# Explicit origins come from ODK_CORS_ORIGINS. Loopback origins on any port are also
# allowed by default: the desktop shell serves the Hub page from 127.0.0.1 on an
# ephemeral port, and without this every call it made to a local API was blocked by
# the browser's preflight. Loopback is the same machine, and the API authenticates
# with bearer tokens rather than cookies, so this admits no ambient credentials.
# ODK_CORS_LOOPBACK=0 turns it off for a deployment that wants same-origin only.
LOOPBACK_ORIGIN_REGEX = r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$"
_origins = [o.strip() for o in os.environ.get("ODK_CORS_ORIGINS", "").split(",") if o.strip()]
_loopback = os.environ.get("ODK_CORS_LOOPBACK", "1").strip().lower() not in {"0", "false", "no"}
if _origins or _loopback:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_origin_regex=LOOPBACK_ORIGIN_REGEX if _loopback else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(ObservabilityMiddleware)

app.include_router(auth.router)
app.include_router(organizations.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(processing.router)
app.include_router(inspection.router)
app.include_router(resources.router)
app.include_router(annotations.router)
app.include_router(sharing.router)
app.include_router(fleet.router)
app.include_router(events.router)
app.include_router(tiles.router)


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
    database = database_readiness()
    if not database.get("postgis"):
        warnings.append(database.get("note", "PostGIS unavailable."))
    secret_ok, secret_reason = secret_is_deployment_grade()
    if not secret_ok:
        warnings.append(secret_reason)

    storage = storage_readiness()
    if storage.get("note"):
        warnings.append(storage["note"])

    ready = bool(database.get("ready") and storage.get("ready"))

    return {
        "status": "ok" if ready else "degraded",
        "version": VERSION,
        "database": database,
        "storage": storage,
        "capabilities": capabilities,
        "observability": observability_contract(),
        "warnings": warnings,
    }


@app.get("/health/live", tags=["system"])
def health_live() -> dict[str, Any]:
    """Process liveness only; it deliberately says nothing about dependencies."""
    from .observability import PROCESS_STARTED  # noqa: PLC0415

    return {
        "status": "alive", "process_uptime_s": round(max(0.0, time.time() - PROCESS_STARTED), 3),
        "scope": "this_api_worker",
    }


@app.get("/health/ready", tags=["system"])
def health_ready() -> JSONResponse:
    """Readiness based on live calls to the configured database and storage backend."""
    database = database_readiness()
    storage = storage_readiness()
    ready = bool(database.get("ready") and storage.get("ready"))
    payload = {
        "status": "ready" if ready else "not_ready",
        "database": database, "storage": storage,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/metrics", tags=["system"], response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Prometheus text for this worker; the body declares its non-aggregated scope."""
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")
