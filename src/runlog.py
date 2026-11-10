"""Run registry: every training run goes through this file.

Comparing and sharing results only works if every run records the same things in the same
place. ``start_run`` creates ``runs/<run_id>/`` containing:

    config.yaml    the exact config used, after command-line overrides and sweep expansion
    env.json       seed, versions (python, torch, PyG), device, platform, git commit, command
    history.csv    one row per epoch, written by ``run.log_epoch``
    metrics.json   final *validation* metrics, written by ``run.finish``
    status.txt     running | finished | failed

The training script may add its own files to ``run.dir`` (best.pt, val_preds.npz, ...).

The test set is locked. ``log_epoch`` and ``finish`` refuse metric names that start with
"test". Only ``python -m src.evaluate --final`` computes test metrics, and it must call
``assert_final_test_allowed()`` first, which checks that Checkpoint 2 has passed.

Usage in src/train.py:

    cfg = runlog.load_config(args.config, overrides=args.set)
    for variant in runlog.expand_sweep(cfg):
        for seed in seeds:
            with runlog.start_run(variant, seed, smoke=args.smoke) as run:
                runlog.set_seed(seed)
                ...
                run.log_epoch(epoch, train_loss=..., val_acc=...)
                ...
                run.finish(best_epoch=..., val_acc=...)

Run ``python -m src.runlog`` to print a table of all runs.

Agents: use this module; don't rewrite it. If you need a feature, add it without changing the
files it writes, because the analyst and reviewer depend on them.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
SMOKE_DIR = RUNS_DIR / "_smoke"
REPORTS_DIR = REPO_ROOT / "reports"

# ----------------------------------------------------------------------------- config

_SCI_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read a YAML config and apply command-line overrides.

    ``overrides`` look like ``["optim.lr=0.001", "model.hidden_dim=128"]``. Each value is parsed
    as YAML, so numbers become numbers and ``[1, 2]`` becomes a list.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("name", path.stem)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return _fix_sci_notation(cfg)


def _fix_sci_notation(node: Any, where: str = "") -> Any:
    # PyYAML reads 5e-4 as the *string* "5e-4" (YAML 1.1 wants 5.0e-4). Convert and warn.
    if isinstance(node, dict):
        return {k: _fix_sci_notation(v, f"{where}.{k}" if where else str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_fix_sci_notation(v, f"{where}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, str) and _SCI_FLOAT.match(node.strip()):
        print(f"[runlog] note: {where}: '{node}' read as text by YAML; using {float(node)}. "
              f"Write it as {float(node):.1e} with a decimal point to avoid this.")
        return float(node)
    return node


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    """``set_by_path(cfg, "optim.lr", 0.01)`` sets ``cfg["optim"]["lr"] = 0.01``."""
    node = cfg
    *parents, last = dotted_key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def expand_sweep(cfg: dict) -> list[dict]:
    """Turn a config with a ``sweep:`` block into one config per combination.

    ``sweep: {model.hidden_dim: [32, 64], optim.lr: [0.01, 0.001]}`` gives 4 configs, named
    ``<name>__hidden_dim-32__lr-0.01`` and so on, each with ``sweep_values`` recording its
    choices. A config without ``sweep`` is returned unchanged, as a one-item list.
    """
    sweep = cfg.get("sweep") or {}
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    if not sweep:
        return [base]
    keys = list(sweep)
    for k in keys:
        if not isinstance(sweep[k], list):
            raise ValueError(f"sweep.{k} must be a list of values, got {sweep[k]!r}")
    variants = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        v = copy.deepcopy(base)
        for k, val in zip(keys, values):
            set_by_path(v, k, val)
        tag = "__".join(f"{k.split('.')[-1]}-{val}" for k, val in zip(keys, values))
        v["name"] = f"{base['name']}__{tag}"
        v["sweep_values"] = dict(zip(keys, values))
        variants.append(v)
    return variants


# ----------------------------------------------------------------------------- environment

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU) so a run can be repeated."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Slower, but makes GPU results repeatable bit-for-bit where PyTorch supports it.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto"):
    """Pick a device. "auto" uses an NVIDIA GPU if present, then Apple's MPS, then the CPU.

    Some PyG operations aren't implemented on MPS. If you see an MPS error, set
    ``train.device: cpu`` in the config.
    """
    import torch

    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_platform() -> str:
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    if "SLURM_JOB_ID" in os.environ:
        return "slurm"  # CCAST or another cluster
    return "local"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _repo_relative(arg: str) -> str:
    try:
        return str(Path(arg).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return arg


def describe_environment(seed: int | None = None, smoke: bool = False) -> dict:
    """Versions, platform, device and git commit. Every run's env.json uses it, and so should
    src/evaluate.py, so the final test records the code version it ran at."""
    import torch

    try:
        import torch_geometric

        pyg = torch_geometric.__version__
    except ImportError:
        pyg = None
    commit = _git("rev-parse", "HEAD")
    where = detect_platform()
    return {
        "seed": seed,
        "smoke": smoke,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        # Paths are stored relative to the repo so run records don't reveal personal folders.
        "command": " ".join([Path(sys.executable).name, *(_repo_relative(a) for a in sys.argv)]),
        "platform": where,
        "hostname": socket.gethostname() if where == "slurm" else None,  # useful on clusters only
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": pyg,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "git_dirty": bool(_git("status", "--porcelain")) if commit else None,
    }


# ----------------------------------------------------------------------------- runs

def _to_py(v: Any) -> Any:
    """Turn tensors and NumPy scalars into plain Python numbers for JSON/CSV."""
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def _refuse_test_keys(metrics: dict) -> None:
    bad = [k for k in metrics if k.lower().startswith("test")]
    if bad:
        raise ValueError(
            f"Refusing to record {bad}: the test set is locked until Checkpoint 2 passes. "
            "Training records validation metrics only; test metrics come from "
            "`python -m src.evaluate --final`."
        )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


class Run:
    """One training run (one config × one seed). Create it with ``start_run``."""

    def __init__(self, cfg: dict, seed: int, smoke: bool = False):
        name = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(cfg.get("name", "run")))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{name}_s{seed}_{uuid.uuid4().hex[:4]}"
        self.dir = (SMOKE_DIR if smoke else RUNS_DIR) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg, self.seed, self.smoke = cfg, seed, smoke
        self._history: list[dict] = []
        self._t0 = time.time()
        self._done = False
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        _write_json(self.dir / "env.json", describe_environment(seed, smoke))
        self._set_status("running")
        print(f"[runlog] run {self.run_id} → {self.dir.relative_to(REPO_ROOT)}")

    def _set_status(self, status: str) -> None:
        (self.dir / "status.txt").write_text(status + "\n")

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        """Record one row of history.csv, for example ``run.log_epoch(5, train_loss=0.3, val_acc=0.8)``."""
        _refuse_test_keys(metrics)
        self._history.append({"epoch": epoch, **{k: _to_py(v) for k, v in metrics.items()}})
        fields: list[str] = []
        for row in self._history:
            fields += [k for k in row if k not in fields]
        with open(self.dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._history)

    def finish(self, **metrics: Any) -> None:
        """Record final validation metrics (for example, the values at the best epoch) and mark the run finished."""
        _refuse_test_keys(metrics)
        out = {k: _to_py(v) for k, v in metrics.items()}
        out["seconds"] = round(time.time() - self._t0, 1)
        _write_json(self.dir / "metrics.json", out)
        self._set_status("finished")
        self._done = True

    def fail(self, error: BaseException | str) -> None:
        (self.dir / "error.txt").write_text(f"{type(error).__name__ if isinstance(error, BaseException) else 'Error'}: {error}\n")
        self._set_status("failed")
        self._done = True

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fail(exc)
        elif not self._done:
            self.fail("run ended without calling run.finish()")
        return False  # never hide the exception


def start_run(cfg: dict, seed: int, smoke: bool = False) -> Run:
    return Run(cfg, seed, smoke=smoke)


def list_runs(status: str | None = "finished", include_smoke: bool = False) -> list[dict]:
    """Load every run record. ``status=None`` returns runs in any state."""
    dirs: list[Path] = []
    if RUNS_DIR.exists():
        dirs += sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if include_smoke and SMOKE_DIR.exists():
        dirs += sorted(p for p in SMOKE_DIR.iterdir() if p.is_dir())
    records = []
    for d in dirs:
        st = (d / "status.txt").read_text().strip() if (d / "status.txt").exists() else "unknown"
        if status is not None and st != status:
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {}
        env = json.loads((d / "env.json").read_text()) if (d / "env.json").exists() else {}
        metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        records.append({
            "run_id": d.name, "dir": str(d), "status": st, "name": cfg.get("name"),
            "seed": env.get("seed"), "platform": env.get("platform"),
            "config": cfg, "env": env, "metrics": metrics,
        })
    return records


# ----------------------------------------------------------------------------- final test lock

def assert_final_test_allowed() -> list[str]:
    """Raise unless Checkpoint 2 passed and the final test has not run yet.

    Returns the config names the reviewer approved (the ``APPROVED_CONFIGS:`` line).
    """
    review = REPORTS_DIR / "reviews" / "checkpoint-2.md"
    if not review.exists():
        raise PermissionError(
            "The test set is locked: reports/reviews/checkpoint-2.md does not exist. "
            "Ask the reviewer for Checkpoint 2 first."
        )
    text = review.read_text()
    verdicts = re.findall(r"^VERDICT:\s*(\w+)\s*$", text, flags=re.M)
    if not verdicts or verdicts[-1].upper() != "PASS":
        raise PermissionError(
            f"The test set is locked: Checkpoint 2 verdict is {verdicts[-1] if verdicts else 'missing'}. "
            "Fix the findings in the review and ask for a re-review."
        )
    if (REPORTS_DIR / "final_test.json").exists():
        raise PermissionError(
            "The final test has already run (reports/final_test.json). It runs once, so test "
            "results can't influence later choices. To check reproducibility, use "
            "`python -m src.evaluate --final --reproduce`."
        )
    approved = re.findall(r"^APPROVED_CONFIGS:\s*(.+)$", text, flags=re.M)
    names = [n.strip() for n in approved[-1].split(",") if n.strip()] if approved else []
    if not names:
        raise PermissionError("Checkpoint 2 passed but lists no APPROVED_CONFIGS. Ask the reviewer to add them.")
    return names


# ----------------------------------------------------------------------------- CLI

def _summary() -> None:
    runs = list_runs(status=None)
    if not runs:
        print("No runs yet in runs/.")
        return
    print(f"{'status':9} {'name':40} {'seed':>4}  {'platform':8} metrics")
    for r in runs:
        m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["metrics"].items()}
        print(f"{r['status']:9} {str(r['name'])[:40]:40} {str(r['seed']):>4}  {str(r['platform']):8} {m}")
    counts = {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})}
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items()))


if __name__ == "__main__":
    _summary()
