"""Run the real Hub API under uvicorn on a loopback port for the duration of a test.

For clients that must be exercised over real HTTP -- the Python SDK and the browser
Hub -- rather than through Starlette's in-process transport.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


@contextmanager
def live_api():
    import uvicorn

    from services.api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
