"""Exceptions raised by the OpenDroneKit SDK.

Every SDK call either returns a real result or raises one of these with the reason in
plain words. Nothing is returned "empty but successful" in place of a refusal.
"""

from __future__ import annotations


class OdkError(Exception):
    """Base class for every error the SDK raises on purpose."""


class Refused(OdkError):
    """OpenDroneKit declined the request and said why.

    The message is the same text the desktop application shows for the same refusal:
    a missing dependency, an input that cannot support the result (an arbitrary-scale
    model asked for metres, a photograph with no scale), a missing dataset, and so on.
    """

    def __init__(self, reason: str, *, payload: dict | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        #: Whatever structured detail came with the refusal (may be empty).
        self.payload = dict(payload or {})


class JobFailed(OdkError):
    """A background job (a reconstruction, say) ended failed or cancelled."""

    def __init__(self, job: dict) -> None:
        self.job = dict(job)
        status = self.job.get("status", "failed")
        why = self.job.get("error") or self.job.get("message") or "no reason recorded"
        super().__init__(f"{self.job.get('name', 'job')} {status}: {why}")


class PluginError(OdkError):
    """A plugin could not be found, built or run, or returned something unusable.

    Raised to the caller who asked for that plugin; it never takes the host down.
    """

    def __init__(self, plugin: str, stage: str, reason: str) -> None:
        self.plugin = str(plugin)
        self.stage = str(stage)
        self.reason = str(reason)
        super().__init__(f"Plugin {self.plugin} failed at {self.stage}: {self.reason}")
