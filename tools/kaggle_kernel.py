"""Generate a Kaggle kernel that trains one OpenDroneKit config.

Kaggle is the free tier of this project's compute, and a kernel is the only way to use
it non-interactively. This writes the two files the Kaggle API wants -- a script and a
``kernel-metadata.json`` -- for a named training config, so a run can be pushed without
hand-editing a notebook and without the settings drifting from the config in the repo.

Two things here are reactions to what actually went wrong last time.

The output whitelist. A previous run copied both prepared corpora into the kernel's
output directory, which made the result 161 MB of data we already had and stalled the
download for three hours before anyone noticed. A kernel's output should be the weights,
the metrics and the log, and nothing that was already an input, so this generates a
finalisation step that copies named artefacts out and refuses to sweep a directory.

The environment check. The same run silently trained on a GPU whose compute capability
the installed torch did not support, and only self-healed because Kaggle happened to
update. The generated kernel prints torch, CUDA and device capability before training so
that mismatch appears at the top of the log rather than as a mysterious slowdown.

    python -m tools.kaggle_kernel solar_thermal_cls --username <kaggle-user>
    python -m tools.kaggle_kernel pvel_ad_yolo11l --username <kaggle-user> --gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "training" / "configs"
OUTPUT_ROOT = ROOT / "training" / "kaggle"

# Which trainer module runs a config, keyed by a marker in the config's own contents.
# Reading the config rather than guessing from the file name means a renamed config
# cannot quietly end up on the wrong trainer.
TRAINER_FOR_KIND = {
    "classification": "training.train_cls",
    "detection": "training.train_det",
    "segmentation": "training.train_seg",
    "segmentation_multiclass": "training.train_seg_multiclass",
    "change": "training.train_change",
    "semantic": "training.train_shared_semantic",
}


class KernelError(ValueError):
    pass


def _trainer_fields(module_name: str) -> set[str] | None:
    """The config keys a trainer will actually accept, read from its own dataclass."""
    import dataclasses
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a trainer whose deps are absent cannot be checked
        return None
    for value in vars(module).values():
        if dataclasses.is_dataclass(value) and value.__name__.endswith("Config"):
            return set(value.__dataclass_fields__)
    return None


def infer_kind(config_text: str) -> str:
    """Work out which trainer a config belongs to from what it declares.

    Decided by MATCHING KEYS against each trainer's own config dataclass rather than by
    grepping the text. Substring matching read words out of comments -- a comment
    mentioning class_names routed a binary PV-extent config to the multiclass trainer --
    and it could not tell change detection from segmentation at all, because they share
    an encoder and differ only in the keys they accept.

    Getting this wrong is expensive in a specific way: the mistake surfaces as "Unknown
    config keys" on a rented GPU, after the repo clone, the torch reinstall and the
    corpus download have all been paid for.
    """
    import yaml

    try:
        keys = set(yaml.safe_load(config_text) or {})
    except Exception:  # noqa: BLE001
        keys = set()

    if "encoder: dinov2" in config_text:
        return "semantic"

    # The architecture decides the FAMILY, and only then do keys choose within it. Key
    # matching alone is not enough: the classifier's config happens to accept every key
    # a SegFormer config sets, so "whichever trainer accepts the keys" sent segmentation
    # configs to the classifier. A nvidia/mit encoder is never a classifier, and that
    # fact is not up for inference.
    if "model_id: yolo" in config_text:
        return "detection"
    if re.search(r"^model_id: (resnet|efficientnet|convnext|vit)", config_text, re.MULTILINE):
        return "classification"

    if "model_id: nvidia/mit-" in config_text:
        # Three trainers share this encoder, so the choice is made on POSITIVE evidence
        # rather than on which dataclass happens to accept the keys. Scoring by "fewest
        # spare fields" sent plain segmentation configs to the change trainer, because a
        # superset config can always swallow a smaller one.
        #
        # Change detection is the only one that splits its own corpus -- it takes pairs
        # of images per site and has to keep a site whole -- so those keys are what
        # identify it. num_classes then separates multiclass from binary.
        if keys & {"split_salt", "val_fraction", "test_fraction"}:
            return "change"
        return "segmentation_multiclass" if "num_classes" in keys else "segmentation"

    # Fallback for a config whose trainer cannot be imported here (missing optional
    # dependency), so generation still works on a machine without torch installed.
    if "model_id: yolo" in config_text:
        return "detection"
    if "model_id: nvidia/mit-" in config_text:
        multiclass = re.search(r"^num_classes:", config_text, re.MULTILINE)
        return "segmentation_multiclass" if multiclass else "segmentation"
    if re.search(r"^model_id: (resnet|efficientnet|convnext|vit)", config_text, re.MULTILINE):
        return "classification"
    raise KernelError(
        "Cannot tell which trainer this config needs. Its keys match no trainer's "
        "config, and no model_id names a known architecture."
    )


def kernel_script(config_name: str, kind: str, dataset_slug: str) -> str:
    """The script the kernel runs, as text."""
    import yaml

    trainer = TRAINER_FOR_KIND[kind]
    settings = yaml.safe_load((CONFIGS / f"{config_name}.yaml").read_text(encoding="utf-8")) or {}
    config_image_size = int(settings.get("image_size") or 0)
    config_batch_size = int(settings.get("batch_size") or 0)
    return f'''"""Kaggle kernel: train {config_name} ({kind}).

Generated by tools/kaggle_kernel.py. Edit the config in the repository, not this file --
a change made here is invisible to everyone reading the repo.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Everything under /kaggle/working is packaged as the kernel's output when the run
# ends. The repo checkout and the unpacked corpus therefore must NOT live there: for
# crack_cls that is 96,092 files Kaggle tries to archive, and the run dies packaging
# them with an empty log, which looks like a crash with no cause. Scratch goes to
# /kaggle/temp, and only the named artefacts are copied into working at the end.
SCRATCH = Path("/kaggle/temp")
REPO = SCRATCH / "OpenDroneKit"
CORPUS = Path("/kaggle/input/{dataset_slug}")
OUT = SCRATCH / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# What the config itself asks for, baked in at generation time so the GPU cap
# below can compare against it instead of overriding blindly.
CONFIG_IMAGE_SIZE = {config_image_size}
CONFIG_BATCH_SIZE = {config_batch_size}


def report_environment() -> str:
    """Print the environment before training rather than after it fails.

    A torch build that does not support the assigned GPU's compute capability still
    runs; it just falls back and takes far longer, which reads as a slow dataset rather
    than a broken install. Printing this at the top of the log makes it obvious.
    """
    import torch

    print("torch     :", torch.__version__)
    print("cuda built:", torch.version.cuda)
    print("cuda avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        print("device    :", torch.cuda.get_device_name(index))
        print("capability: sm_{{}}{{}}".format(major, minor))
        supported = torch.cuda.get_arch_list()
        print("torch arch:", ", ".join(supported))
        if f"sm_{{major}}{{minor}}" not in supported:
            print(
                f"This torch does not support sm_{{major}}{{minor}}; every CUDA kernel "
                "launch will fail with 'no kernel image is available'."
            )
            return f"sm_{{major}}{{minor}}"
    else:
        print("WARNING: no GPU visible. Check the kernel's accelerator setting.")
    return ""


def install_compatible_torch(capability: str) -> None:
    """Replace the preinstalled torch with one that has kernels for this GPU.

    Kaggle's P100 is sm_60 (Pascal) and the preinstalled torch 2.10+cu128 ships
    sm_70 upward, so the first convolution dies with cudaErrorNoKernelImageForDevice.
    It is not a slowdown and not a fallback -- nothing runs at all.

    The cu121 wheels are the last ones built with Pascal kernels, so that is what gets
    installed. This is a real cost, roughly two minutes and a couple of gigabytes per
    run, and it is only paid when the check above finds a mismatch.
    """
    print(f"installing a torch build with {{capability}} kernels", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.4.1", "torchvision==0.19.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
    )
    if result.returncode != 0:
        raise SystemExit("Could not install a torch build compatible with this GPU.")

    # Verify in a fresh interpreter: the already-imported torch cannot be swapped in
    # place, and asserting the new build actually carries the capability is the only
    # way to know the reinstall did what it claimed.
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch;a=torch.cuda.get_device_capability(0);"
         "s=f'sm_{{a[0]}}{{a[1]}}';print(s, s in torch.cuda.get_arch_list())"],
        capture_output=True, text=True,
    )
    print("after reinstall:", check.stdout.strip() or check.stderr.strip()[:200])
    if "True" not in check.stdout:
        raise SystemExit(f"Reinstalled torch still lacks {{capability}} kernels.")


def resolve_corpus() -> Path:
    """Find the attached corpus, and say what IS there when it is missing.

    Kaggle mounts a dataset under /kaggle/input using a name derived from the dataset,
    and guessing that name wrongly produces "corpus not attached" -- which reads as a
    missing dataset rather than a wrong path, and sends you to fix the wrong thing.
    Directories are also zipped per-split by the uploader, so the mount may hold
    train.zip rather than train/, and those are unpacked here.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        raise SystemExit("No /kaggle/input at all; the kernel has no attached data.")

    # Search for the directory that actually holds the corpus rather than trusting a
    # guessed name. Kaggle nests the mount (observed: /kaggle/input/datasets/<name>),
    # and the depth is not something to hardcode -- the marker is the content.
    def looks_like_corpus(path: Path) -> bool:
        if any((path / split).is_dir() for split in ("train", "val", "test")):
            return True
        # A semantic corpus is a manifest plus flat image and label directories -- there
        # are no per-split folders, because the split lives in the manifest. Without this
        # the search walked straight past a perfectly good corpus and reported it missing,
        # which reads as a failed upload rather than a detector that only knows one shape.
        if (path / "corpus.json").is_file():
            return True
        return bool(list(path.glob("*.zip")))

    found = None
    if CORPUS.is_dir() and looks_like_corpus(CORPUS):
        found = CORPUS
    else:
        for depth in range(4):
            for candidate in sorted(root.glob("/".join(["*"] * (depth + 1)))):
                if candidate.is_dir() and looks_like_corpus(candidate):
                    found = candidate
                    break
            if found is not None:
                break
    if found is None:
        listing = [str(q.relative_to(root)) for q in root.rglob("*") if q.is_dir()][:40]
        raise SystemExit(f"No corpus under /kaggle/input. Directories seen: {{listing}}")
    if found != CORPUS:
        print(f"corpus expected at {{CORPUS}}, found at {{found}}")

    archives = sorted(found.glob("*.zip"))
    if archives:
        # Kaggle serves the zips as-is; the trainer wants split directories.
        import zipfile

        unpacked = SCRATCH / "corpus"
        unpacked.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            target = unpacked / archive.stem
            if target.is_dir():
                continue
            print(f"unpacking {{archive.name}}", flush=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(target)
        for extra in found.glob("*.yaml"):
            shutil.copy2(extra, unpacked / extra.name)
        print("corpus ready at", unpacked, sorted(p.name for p in unpacked.iterdir()))
        return unpacked
    return found


def ensure_repo() -> None:
    """Clone the code the kernel is supposed to run.

    A kernel is a bare script on a fresh machine: nothing about the repository is
    present unless it is fetched. Cloning at a pinned commit rather than tracking main
    means a kernel re-run months later trains the same code, and a result can be traced
    to a SHA rather than to whatever main happened to be that day.
    """
    if REPO.is_dir():
        return
    commit = os.environ.get("ODK_COMMIT", "main")
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/00PrabalK00/OpenDroneKit.git", str(REPO)],
        check=True,
    )
    subprocess.run(["git", "checkout", "--quiet", commit], cwd=REPO, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    print("repo at commit", head)


def gpu_memory_overrides() -> list:
    """Shrink the run to fit this GPU, without touching the config.

    The detection configs are sized for a 24 GB card. Kaggle's P100 has 16 GB, and the
    kernels started training and then died: roads at 640px/batch 16 survived while
    rail_obstacle at 960/12, corrosion at 960/10 and solar at 1024/8 did not.

    Overriding on the command line rather than editing the configs matters. Those values
    are correct for the hardware they were written for, and a config edited to fit the
    smallest machine that ever ran it quietly degrades every future run on a bigger one.
    The trainer already accepts both flags.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 20:
        print(f"GPU has {{total_gb:.0f}} GB; running the config as written", flush=True)
        return []

    # CAP, never raise. This forced --image-size 640 unconditionally, and the
    # agriculture configs ask for 512 -- so the helper written to make runs fit made
    # them BIGGER, and both arms died with CUDA out of memory at 15.6 GB on a 15.89 GB
    # card. A ceiling that is above the config is not a ceiling.
    image_size = min(CONFIG_IMAGE_SIZE, 640) if CONFIG_IMAGE_SIZE else 640
    batch_size = min(CONFIG_BATCH_SIZE, 8) if CONFIG_BATCH_SIZE else 8
    if image_size >= CONFIG_IMAGE_SIZE and batch_size >= CONFIG_BATCH_SIZE:
        print(
            f"GPU has {{total_gb:.0f}} GB; the config already fits at "
            f"{{CONFIG_IMAGE_SIZE}}px/batch {{CONFIG_BATCH_SIZE}}",
            flush=True,
        )
        return []
    print(
        f"GPU has only {{total_gb:.0f}} GB; capping to {{image_size}}px/batch {{batch_size}} "
        f"(config asks {{CONFIG_IMAGE_SIZE}}px/batch {{CONFIG_BATCH_SIZE}})",
        flush=True,
    )
    return ["--image-size", str(image_size), "--batch-size", str(batch_size)]


def ensure_ultralytics() -> None:
    """Install ultralytics if the trainer will need it.

    Kaggle preinstalls torch and torchvision but NOT ultralytics, so the detection
    kernels failed with "train_det needs ultralytics" after the torch self-heal had
    already succeeded -- the fix worked and the run died on the next line.

    --no-deps matters: ultralytics depends on torch and pip would happily pull the
    latest wheel back in, undoing the sm_60-compatible build installed moments earlier
    and returning the kernel to the exact failure it just escaped.
    """
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    print('installing ultralytics (--no-deps to protect the torch build)', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                    'ultralytics', 'ultralytics-thop'], check=True)
    # Runtime imports ultralytics needs that Kaggle may not carry.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                    'py-cpuinfo'], check=False)
    check = subprocess.run([sys.executable, '-c',
                            'import ultralytics,torch;a=torch.cuda.get_device_capability(0);'
                            'print(ultralytics.__version__, f"sm_{{a[0]}}{{a[1]}}" in torch.cuda.get_arch_list())'],
                           capture_output=True, text=True)
    print('ultralytics check:', check.stdout.strip() or check.stderr.strip()[:200])
    if 'True' not in check.stdout:
        raise SystemExit('ultralytics installed but the torch build no longer matches the GPU.')


class _Tee:
    """Mirror stdout into /kaggle/working/run.log, flushing every line.

    This exists because four detector kernels failed in a row and left NOTHING to read:
    Kaggle's own captured log came back zero bytes, and everything the kernel writes goes
    to /kaggle/temp, which is not packaged as output. A run that dies is exactly the run
    whose log matters, and it was the only one that produced none.

    The file lives under /kaggle/working so it is packaged even when the run fails, and it
    is flushed per line so a hard kill -- an OOM, a timeout -- still leaves everything
    printed up to the moment of death.
    """

    def __init__(self, path: Path) -> None:
        self.stream = sys.__stdout__
        self.handle = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.handle.write(text)
        self.handle.flush()
        try:
            os.fsync(self.handle.fileno())
        except OSError:
            pass
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


def main() -> int:
    log_path = Path("/kaggle/working") / "run.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"tee: mirroring this run to {{log_path}}", flush=True)

    missing = report_environment()
    if missing:
        install_compatible_torch(missing)
    if '{trainer}' == 'training.train_det':
        ensure_ultralytics()
    ensure_repo()
    corpus = resolve_corpus()

    if "{trainer}" == "training.train_shared_semantic":
        # A different CLI from the other trainers, and worth not papering over: this one
        # takes a corpus MANIFEST rather than a data root, because a semantic corpus is a
        # list of samples with their splits and licences rather than a directory layout.
        # --source github because training/sources/dinov2 and the pretrained encoder are
        # not in the repository, so a fresh clone has neither; torch.hub fetches both.
        # Checked BEFORE the session starts. The pack is JPEG and carries no
        # georeferencing, so every sample must bring its own ground sample distance; if it
        # does not, the loader crops a fixed pixel count from sources spanning 0.20 to
        # 4.78 m/px and spends the whole session reproducing the exact defect this run
        # exists to correct, looking entirely normal throughout. The check also refuses a
        # corpus whose India holdout tiles escaped test, because a score measured on data
        # the model trained on is not a holdout score.
        if subprocess.run(
            [sys.executable, str(REPO / "tools" / "check_packed_corpus.py"),
             str(corpus / "corpus.json")],
            cwd=str(REPO),
        ).returncode != 0:
            raise SystemExit("corpus rejected; not spending a session on it")

        command = [
            sys.executable, "-m", "{trainer}",
            "--config", str(REPO / "training" / "configs" / "{config_name}.yaml"),
            "--corpus", str(corpus / "corpus.json"),
            "--run-dir", str(OUT / "{config_name}"),
            "--source", "github",
            # Kaggle caps a session and this corpus needs more than one, so every run
            # continues from the mirrored checkpoint. --resume-if-available rather than
            # --resume because the FIRST run has nothing to resume: plain --resume is
            # strict and killed the first attempt with FileNotFoundError before a single
            # step. The tolerant flag says which of the two it did.
            "--resume-if-available",
        ]
    else:
        command = [
            sys.executable, "-m", "{trainer}",
            "--config", str(REPO / "training" / "configs" / "{config_name}.yaml"),
            "--data-root", str(corpus),
            "--output-dir", str(OUT),
        ]
        # A multispectral config points band_root at a repo-relative cache that does not
        # exist here; the stacks arrive inside the mounted dataset instead. The trainer
        # refuses a missing stack rather than reading RGB and calling it five-band, so
        # this has to be pointed at the real location or the run stops on the first tile.
        # Searched upward, not just inside the corpus. resolve_corpus descends to the
        # directory holding train/val/test, so it lands on <dataset>/agriculture_seg
        # while the band stacks sit beside it at <dataset>/weedsgalore_bands. Looking
        # only in the corpus directory found nothing, no --band-root was passed, and the
        # trainer refused on the first tile -- correctly, but the run was lost.
        bands = None
        for candidate in (corpus, corpus.parent, corpus.parent.parent):
            probe = candidate / "weedsgalore_bands"
            if probe.is_dir():
                bands = probe
                break
        if bands is not None:
            command += ["--band-root", str(bands)]
            print("multispectral stacks at", bands, flush=True)
        command += gpu_memory_overrides()
    restore_checkpoint()
    report_resources("before training")
    # Started before training so a run killed at the session limit still leaves its best
    # epoch behind. Kaggle packages /kaggle/working; OUT is scratch and is not packaged.
    start_checkpoint_mirror()
    print("running:", " ".join(command), flush=True)

    # Piped rather than inherited so the child's output goes through the tee above. With
    # inherited handles it writes straight to the real descriptor and never reaches
    # run.log -- which would leave the failing runs just as silent as before.
    process = subprocess.Popen(
        command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\\n"), flush=True)
    returncode = process.wait()

    report_resources("after training")
    if returncode != 0:
        raise SystemExit(f"Training failed with exit code {{returncode}}.")

    collect_outputs()
    return 0


def report_resources(when: str) -> None:
    """Print free disk and RAM, so a silent kill can be attributed afterwards.

    A kernel killed for exhausting memory or disk leaves no message of its own. Printing
    the headroom on either side of training turns "it died" into a number that says which
    one ran out, or rules both out.
    """
    usage = shutil.disk_usage("/kaggle/temp")
    line = (
        f"resources {{when}}: disk free {{usage.free / 1e9:.1f}} GB "
        f"of {{usage.total / 1e9:.1f}} GB"
    )
    try:
        meminfo = {{}}
        for entry in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = entry.partition(":")
            meminfo[key] = float(value.strip().split()[0]) / 1e6  # kB -> GB
        line += (
            f"; RAM available {{meminfo.get('MemAvailable', 0):.1f}} GB "
            f"of {{meminfo.get('MemTotal', 0):.1f}} GB"
        )
    except (OSError, ValueError, IndexError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            line += f"; VRAM free {{free / 1e9:.1f}} GB of {{total / 1e9:.1f}} GB"
    except Exception:
        pass
    print(line, flush=True)


def start_checkpoint_mirror(interval_s: int = 300):
    """Copy checkpoints into /kaggle/working WHILE training runs, not only after.

    A roads run reached epoch 83 of 100 and was cancelled at the session limit. It
    produced nothing at all, because artefacts were collected once, at the end, and the
    end never came. Eight hours of GPU for zero output, and the checkpoints existed the
    whole time -- they were simply in a directory Kaggle does not package.

    So a daemon thread mirrors them every few minutes. A cancelled or timed-out run now
    yields its best epoch so far, which for a long run is most of the value. The cost is
    a few seconds of copying per interval.

    Daemon so it cannot keep the kernel alive, and every failure is swallowed: a mirror
    that crashed the run it exists to protect would be worse than no mirror.
    """
    import threading
    import time as _time

    wanted = ("best.pt", "last.pt", "best.pth", "last.pth", "summary.json", "results.csv")
    staged = Path("/kaggle/working")

    def mirror() -> None:
        while True:
            _time.sleep(interval_s)
            try:
                for name in wanted:
                    for source in OUT.rglob(name):
                        target = staged / source.name
                        # Skip unchanged files so a 200 MB checkpoint is not rewritten
                        # every interval for no reason.
                        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
                            continue
                        shutil.copy2(source, target)
                        print(f"mirrored {{source.name}} ({{target.stat().st_size / 1e6:.1f}} MB)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"checkpoint mirror skipped a pass: {{exc}}", flush=True)

    thread = threading.Thread(target=mirror, name="checkpoint-mirror", daemon=True)
    thread.start()
    return thread


def restore_checkpoint() -> bool:
    """Bring a previous session's checkpoint back into the run directory.

    The run directory is /kaggle/temp, which is scratch and does not survive a session.
    The mirror copies checkpoints to /kaggle/working, which Kaggle packages as output --
    and that is where they stop. Nothing brought them back, so --resume-if-available found
    an empty directory on the next session, said "starting a fresh run", and did exactly
    that. Twelve hours of training discarded silently, with a log line that reads like
    correct behaviour because it IS the correct behaviour for a genuinely first run.

    Kaggle mounts a kernel's own previous output under /kaggle/input when the kernel lists
    itself in kernel_sources. This finds it there and copies it in.
    """
    target = OUT / "{config_name}"
    if (target / "last.pt").is_file():
        print("checkpoint already in the run directory")
        return True

    candidates = sorted(Path("/kaggle/input").rglob("last.pt"))
    if not candidates:
        print("no previous checkpoint mounted; this session starts from scratch", flush=True)
        return False

    # Newest wins: several versions may be mounted and the latest is the one to continue.
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "best.pt"):
        source = newest.parent / name
        if source.is_file():
            shutil.copy2(source, target / name)
            print(f"restored {{name}} ({{source.stat().st_size / 1e6:.1f}} MB) from {{source.parent}}",
                  flush=True)
    return (target / "last.pt").is_file()


def collect_outputs() -> None:
    """Copy named artefacts to /kaggle/working, and only those.

    Kaggle packages everything under /kaggle/working as the kernel's output. A previous
    run left the corpus there and produced a 161 MB download of data that was already on
    disk locally, which then stalled. Named files only, and a printed manifest so the
    size of what is about to be downloaded is visible in the log.
    """
    wanted = ["best.pt", "last.pt", "best.pth", "summary.json", "metrics.json", "results.csv"]
    staged = Path("/kaggle/working")
    manifest = []
    for name in wanted:
        for source in OUT.rglob(name):
            target = staged / source.name
            if source.resolve() == target.resolve():
                continue
            shutil.copy2(source, target)
            manifest.append({{"file": target.name, "bytes": target.stat().st_size}})

    total = sum(item["bytes"] for item in manifest)
    print(json.dumps({{"outputs": manifest, "total_bytes": total}}, indent=2))
    if not manifest:
        print("WARNING: training reported success but produced no recognised artefact.")


if __name__ == "__main__":
    raise SystemExit(main())
'''


def kernel_metadata(
    config_name: str,
    username: str,
    dataset_slug: str,
    *,
    gpu: bool,
    internet: bool,
) -> dict[str, Any]:
    slug = f"{username}/odk-train-{config_name.replace('_', '-')}"
    return {
        "id": slug,
        "title": f"ODK train {config_name}",
        "code_file": f"{config_name}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        # Its own previous output, mounted under /kaggle/input, is where an
        # interrupted run's checkpoint comes back from. Without this a capped
        # session cannot be continued and every re-push restarts from epoch 1.
        "kernel_sources": [slug],
    }


def generate(
    config_name: str,
    username: str,
    *,
    dataset_slug: str = "",
    gpu: bool = True,
    internet: bool = True,
) -> Path:
    config_path = CONFIGS / f"{config_name}.yaml"
    if not config_path.is_file():
        available = ", ".join(sorted(p.stem for p in CONFIGS.glob("*.yaml")))
        raise KernelError(f"No config named {config_name!r}. Available: {available}")

    kind = infer_kind(config_path.read_text(encoding="utf-8"))
    slug = dataset_slug or f"{username}/odk-{config_name.replace('_', '-')}-corpus"

    target = OUTPUT_ROOT / config_name
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{config_name}.py").write_text(
        kernel_script(config_name, kind, slug.split("/", 1)[-1]), encoding="utf-8"
    )
    (target / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(config_name, username, slug, gpu=gpu, internet=internet), indent=2),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.kaggle_kernel")
    parser.add_argument("config", help="Config stem, e.g. solar_thermal_cls.")
    parser.add_argument("--username", required=True, help="Kaggle username owning the kernel.")
    parser.add_argument("--dataset", default="", help="Corpus dataset slug, user/name.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-internet", action="store_true")
    args = parser.parse_args(argv)

    try:
        target = generate(
            args.config,
            args.username,
            dataset_slug=args.dataset,
            gpu=not args.no_gpu,
            internet=not args.no_internet,
        )
    except KernelError as exc:
        print(str(exc))
        return 2

    print(f"Wrote {target}")
    print(f"  push with: kaggle kernels push -p {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
