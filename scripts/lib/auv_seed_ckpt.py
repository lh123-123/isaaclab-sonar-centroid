#!/usr/bin/env python3
"""Resolve/verify SB3 checkpoints by (task, seed). Never cross seeds."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "scripts" / "auv_v2_paper_run_lock.tsv"


def load_lock() -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    if not LOCK_PATH.exists():
        return out
    for line in LOCK_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        task, seed, run = parts
        p = Path(run)
        if not p.is_absolute():
            p = REPO / p
        out[(task, str(seed))] = p
    return out


def locked_run_dir(task: str, seed: str) -> Path | None:
    return load_lock().get((task, str(seed)))


def append_lock(task: str, seed: str, run_dir: Path) -> None:
    seed = str(seed)
    existing = load_lock()
    if (task, seed) in existing:
        return
    rel = run_dir.resolve()
    try:
        rel_s = str(rel.relative_to(REPO))
    except ValueError:
        rel_s = str(rel)
    with LOCK_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{task}\t{seed}\t{rel_s}\n")


def sonar_sig(run_dir: Path) -> tuple[str, str, str] | None:
    y = run_dir / "params" / "env.yaml"
    if not y.exists():
        return None
    text = y.read_text()
    idx = text.find("oceansim_sonar_cfg:")
    if idx < 0:
        return None
    block = text[idx : idx + 800]

    def grab(key: str) -> str | None:
        m = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", block, re.M)
        return m.group(1) if m else None

    r, h, w = grab("range_res"), grab("height"), grab("width")
    if r is None or h is None or w is None:
        return None
    return (r, h, w)


def _run_seed(run_dir: Path) -> str | None:
    y = run_dir / "params" / "agent.yaml"
    if not y.exists():
        return None
    for line in y.read_text().splitlines():
        if line.startswith("seed:"):
            return line.split(":", 1)[1].strip()
    return None


def _step_of(ckpt: Path) -> int:
    parts = ckpt.name.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return -1


def ckpt_run_dir(ckpt: Path) -> Path:
    return ckpt.resolve().parent


def ckpt_seed(ckpt: Path) -> str | None:
    return _run_seed(ckpt_run_dir(ckpt))


def iter_ckpts(task: str, seed: str, run_dir: Path | None = None):
    seed = str(seed)
    if run_dir is not None:
        roots = [run_dir]
    else:
        root = REPO / "logs" / "sb3" / task
        if not root.exists():
            return
        roots = [p for p in root.iterdir() if p.is_dir()]
    for run in roots:
        if _run_seed(run) != seed:
            continue
        for z in run.glob("model_*_steps.zip"):
            yield z, _step_of(z)


def latest_ckpt(task: str, seed: str, run_dir: Path | None = None) -> Path | None:
    seed = str(seed)
    if run_dir is None:
        locked = locked_run_dir(task, seed)
        if locked is not None:
            sig = sonar_sig(locked)
            best: Path | None = None
            best_n = -1
            best_mtime = -1.0
            root = REPO / "logs" / "sb3" / task
            if not root.exists():
                return None
            for run in root.iterdir():
                if not run.is_dir() or _run_seed(run) != seed:
                    continue
                if sig is not None and sonar_sig(run) != sig:
                    continue
                for z, n in iter_ckpts(task, seed, run_dir=run):
                    m = z.stat().st_mtime
                    if n > best_n or (n == best_n and m > best_mtime):
                        best, best_n, best_mtime = z, n, m
            return best
    best: Path | None = None
    best_n = -1
    best_mtime = -1.0
    for z, n in iter_ckpts(task, seed, run_dir=run_dir):
        m = z.stat().st_mtime
        if n > best_n or (n == best_n and m > best_mtime):
            best, best_n, best_mtime = z, n, m
    return best


def verify(ckpt: Path, seed: str) -> None:
    got = ckpt_seed(ckpt)
    if got is None:
        raise SystemExit(f"SEED CHECK FAIL: no params/agent.yaml seed for {ckpt}")
    if got != str(seed):
        raise SystemExit(
            f"SEED CHECK FAIL: checkpoint seed={got} != requested seed={seed}\n  {ckpt}"
        )


def _cmd_latest(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else None
    ckpt = latest_ckpt(args.task, str(args.seed), run_dir=run_dir)
    if ckpt is None:
        return 0
    verify(ckpt, str(args.seed))
    print(ckpt)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"SEED CHECK FAIL: missing file {ckpt}")
    verify(ckpt, str(args.seed))
    print(ckpt.resolve())
    return 0


def _cmd_lock(args: argparse.Namespace) -> int:
    run = Path(args.run_dir)
    if not run.is_dir():
        raise SystemExit(f"LOCK FAIL: not a directory {run}")
    got = _run_seed(run)
    if got != str(args.seed):
        raise SystemExit(f"LOCK FAIL: run seed={got} != requested {args.seed}\n  {run}")
    append_lock(args.task, str(args.seed), run)
    print(run.resolve())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_lat = sub.add_parser("latest", help="Print latest zip for this task+seed")
    p_lat.add_argument("--task", required=True)
    p_lat.add_argument("--seed", required=True)
    p_lat.add_argument("--run-dir", default=None, help="Restrict to one run directory")
    p_lat.set_defaults(func=_cmd_latest)

    p_ver = sub.add_parser("verify", help="Exit non-zero if checkpoint seed mismatches")
    p_ver.add_argument("--checkpoint", required=True)
    p_ver.add_argument("--seed", required=True)
    p_ver.set_defaults(func=_cmd_verify)

    p_lock = sub.add_parser("lock", help="Record canonical run dir for task+seed (no overwrite)")
    p_lock.add_argument("--task", required=True)
    p_lock.add_argument("--seed", required=True)
    p_lock.add_argument("--run-dir", required=True)
    p_lock.set_defaults(func=_cmd_lock)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
