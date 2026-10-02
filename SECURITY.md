# Security policy

## Supported versions

OpenDroneKit has not reached 1.0 and has no maintained release branches. Security fixes
are made on `main` only. If you run an older checkout, the fix is to update to the latest
`main`.

| Version | Supported |
|---|---|
| latest `main` | yes |
| anything older | no |

## Reporting a vulnerability

**Please do not open a public issue, discussion or pull request for a security problem.**

Report it privately through GitHub:

1. Go to the repository's [Security tab](https://github.com/00PrabalK00/OpenDroneKit/security).
2. Choose **Report a vulnerability**. This opens a private security advisory visible only
   to you and the maintainer ([@00PrabalK00](https://github.com/00PrabalK00)).

If the **Report a vulnerability** button is not there, private vulnerability reporting
has not been enabled on the repository yet. In that case open a public issue that says
only that you have a security report and asks for a private channel, with no details,
and the maintainer will follow up.

A useful report includes:

- the commit you tested (`git rev-parse HEAD`) and your OS;
- which component is affected (desktop shell, Hub API, worker, flight/MAVLink, training
  or model loading, deployment files);
- steps to reproduce, and what an attacker gains;
- whether it needs local access, network access to a deployed Hub, or access to the
  aircraft's telemetry link.

## What to expect

This is a small project maintained in spare time, so there is no guaranteed response
time. The intent is to acknowledge a report within a couple of weeks, to keep you
informed while it is investigated, and to credit you in the advisory and the changelog
unless you prefer not to be named. Please give a reasonable window for a fix before
disclosing publicly, and coordinate the disclosure date through the advisory.

## Scope notes specific to OpenDroneKit

These describe how the software is meant to behave, so you can tell a vulnerability from
a design decision.

### Desktop shell (`app/shell.py`)

- The UI is served by a small HTTP server bound explicitly to `127.0.0.1` on an
  ephemeral port. It serves the static files under `app/web/`, the offline tile cache
  (`tiles/{z}/{x}/{y}.png`) and `odk-runtime.json`. It has no authentication: it is
  assumed that other processes running as the same user on the same machine are trusted.
  Anything that makes this server reachable from another machine, or lets it serve files
  outside `app/web/` and the tile cache, **is** in scope.
- The desktop `Api` is exposed to the page through pywebview's JS bridge. Script injection
  into the page (for example through a project name, a report field or an imported file)
  that reaches that bridge is in scope.
- The in-process Hub service also binds to `127.0.0.1` (port 8000 when free).
  `ODK_HUB=0` disables it.

### Hub / HTTP API (`services/api/`)

- Sessions are JWTs signed with `ODK_SECRET_KEY`. Without it, a random per-process key is
  used (sessions die on restart) and the API's health report carries a warning; a key
  shorter than 32 bytes is stretched by hashing and also warned about. Running a public deployment
  without setting `ODK_SECRET_KEY` is a configuration error, not a vulnerability.
- Passwords are hashed with bcrypt (pre-hashed beyond 72 bytes so long passphrases are
  not truncated). API keys, share links and webhooks are part of the API surface.
- CORS allows loopback origins on any port so the desktop shell can reach the Hub
  (`ODK_CORS_LOOPBACK=0` disables this); other origins come from `ODK_CORS_ORIGINS`.
- In scope: authentication or authorisation bypass, one project or tenant reading
  another's data, path traversal through uploads or artifact downloads, disclosure of
  server filesystem paths or secrets in responses, SSRF through webhooks or the cloud
  reconstruction endpoint, and injection into the database layer.
- The compose and Helm files under `infrastructure/` are examples for self-hosting.
  They require you to set credentials in an uncommitted `.env`; anything in them that is
  insecure by default when used as documented is in scope.

### Flight and vehicle control (MAVLink)

- OpenDroneKit can arm, take off, upload missions, geofences and rally points, and
  command RTL or abort on a real aircraft. **A report that could make an aircraft do
  something its operator did not command, or stop it obeying an abort or RTL, is treated
  as the highest priority.** That includes a crafted mission or plan file that uploads
  something different from what the UI showed, preflight checks that pass when they
  should not, or telemetry that hides a failure.
- MAVLink itself is generally unauthenticated unless MAVLink 2 message signing is set up
  on the link. Weaknesses of the protocol or of a particular radio are out of scope, but
  OpenDroneKit making them worse (for example by opening an unexpected listening port)
  is in scope.
- Nothing in this project replaces the pilot's responsibility or the autopilot's own
  failsafes. See [docs/PILOT_GUIDE.md](docs/PILOT_GUIDE.md).

### Models and training

- Model weights are not distributed with the repository. When weights are installed
  through `training/register.py`, their sha256 is recorded, and the runtime hashes the
  installed file and reports a mismatch if it was replaced. A way to make a modified
  model load as "verified", or to make a mismatch go unreported, is in scope.
- The training and export scripts load PyTorch checkpoints with
  `torch.load(..., weights_only=False)`, which can execute code from the file. Only use
  checkpoints you produced or trust. The application runtime loads ONNX only.

### Out of scope

- Findings that require an attacker who already controls your user account or machine.
- Missing hardening headers on the loopback-only desktop server.
- Denial of service against a development server started with `--reload`.
- Vulnerabilities in third-party dependencies with no demonstrated effect on
  OpenDroneKit (please report those upstream; a note to us is still welcome if we pin an
  affected version).

## Keeping secrets out of the repository

Never commit `.env` files, `ODK_SECRET_KEY`, S3 keys, `KAGGLE_API_TOKEN`,
`ROBOFLOW_API_KEY`, cloud tokens or trained weights. If you find a committed secret in
the history, report it privately as above.
