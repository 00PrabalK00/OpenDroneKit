# Getting help

OpenDroneKit is maintained in spare time by one person. There is no paid support and no
guaranteed response time, but questions and bug reports are welcome.

## Where to ask

| You want to... | Go to |
|---|---|
| Install, run or use the toolkit | The guides first: [Installation](docs/INSTALLATION.md), [User guide](docs/USER_GUIDE.md), [UI guide](docs/UI_GUIDE.md), [Pilot guide](docs/PILOT_GUIDE.md). The website, [opendronekit.com](https://opendronekit.com), links to them too. |
| Ask a question or discuss an idea | [GitHub Discussions](https://github.com/00PrabalK00/OpenDroneKit/discussions), if enabled for the repository; otherwise open an issue. |
| Report a bug or a wrong refusal | [New issue: bug report](https://github.com/00PrabalK00/OpenDroneKit/issues/new/choose) |
| Request a feature | [New issue: feature request](https://github.com/00PrabalK00/OpenDroneKit/issues/new/choose) |
| Report a security problem | Privately, as described in [SECURITY.md](SECURITY.md). Never in a public issue. |
| Report a conduct problem | Privately, as described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). |

Search existing issues before opening a new one.

## Check this first

Many problems are a missing optional dependency, which OpenDroneKit reports as a refusal
with a reason rather than a crash. Before reporting:

```bash
python main.py --check-deps     # names every missing or too-old package
python -m pytest -rs tests/test_honesty.py   # a quick sanity run; -rs lists skip reasons
```

[docs/INSTALLATION.md](docs/INSTALLATION.md#optional-dependencies-and-what-you-lose-without-each)
lists what each optional dependency gives you. "Dense MVS: no" without a CUDA COLMAP build,
and model-backed detectors refusing when no weights are installed, are expected behaviour.

## What to include in a report

- **Version**: the commit you are on (`git rev-parse --short HEAD`) and branch.
- **OS** and version (for example Windows 11 23H2, Ubuntu 22.04).
- **Environment**: `python --version`, how you installed (the `odk` conda env from
  `environment.yml`, a venv, `OpenDroneKit.bat`), and the output of
  `python main.py --check-deps`.
- **Which interface**: classic (`python main.py`), cockpit (`--ui cockpit`), the Hub in a
  browser, `run_pipeline.py`, or the HTTP API.
- **What you did, what you expected, what happened**, including the exact refusal or
  error text.
- **Logs**: startup failures are written to `final_toolkit_outputs/logs/startup-error.log`
  (or `<ODK_DATA_DIR>/logs/` if you set `ODK_DATA_DIR`). For a pipeline run, the
  `summary.json` from its output folder is often the most useful file.
- For flight issues: autopilot and firmware version, whether it was SITL or a real
  aircraft, and the connection string type (serial, UDP, TCP).

Please remove anything private before attaching files: site coordinates you do not want
public, imagery, tokens, and `.env` contents.
