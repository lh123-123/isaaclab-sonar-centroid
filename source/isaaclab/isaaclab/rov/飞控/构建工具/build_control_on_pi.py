"""Build the combined patch in a new isolated tree. Never accesses the vehicle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import time
import zlib
import base64

COMMIT = "dbe792162d06cab66c3475fd5556bf7a120f119e"
ROOT = Path("/home/liang/rov_firmware_build_20261004")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepared-name", choices=("prepared_control_20261004", "prepared_independent_20261004",
                                                   "prepared_control_20261005", "prepared_independent_20261005",
                                                   "prepared_tolerance_20261005", "prepared_horizontal_20261007", "prepared_level_target_20261007",
                                                   "prepared_current_guard_20261008", "prepared_current_guard_20261008_v2",
                                                   "prepared_gentle_attitude_20261008",
                                                   "prepared_gentle_depth_20261008",
                                                   "prepared_depth_bottom_guard_20261008",
                                                   "prepared_full_trigger_release_20261013",
                                                   "prepared_full_trigger_release_safe_20261013"),
                        default="prepared_control_20261004")
    parser.add_argument("--resume", action="store_true", help="reuse only a hash-verified task-owned source tree")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--verified-cache-path", action="store_true",
                        help="use the prepared tree at the prior compiler path, only after the guarded cache migration")
    parser.add_argument("--clean", action="store_true", help="discard only compiler outputs after interrupted power/reboot")
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    if bundle.parent != ROOT or not bundle.name.startswith("control_"):
        raise RuntimeError("Unexpected isolated bundle path")
    original = ROOT / "source"
    destination = ROOT / 'control_1715/source' if args.verified_cache_path else bundle / "source"
    if args.verified_cache_path and (not args.resume or not (bundle / 'cache_migration.json').is_file()):
        raise RuntimeError('Verified cache migration evidence required')
    manifest = json.loads((bundle / args.prepared_name / "manifest.json").read_text())
    if manifest["source_commit"] != COMMIT:
        raise RuntimeError("Unexpected pinned source")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=original).decode().strip()
    if actual != COMMIT:
        raise RuntimeError("Full source commit mismatch")
    patch = bundle / args.prepared_name / "ardusub_rov_control.patch"
    if digest(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Patch hash mismatch")
    for name, expected in manifest["baseline_sha256"].items():
        stored = bundle / "baseline_dbe79216" / name
        official = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=original)
        if digest(stored) != expected or hashlib.sha256(official).hexdigest() != expected:
            raise RuntimeError("Baseline differs from actual pinned Git: " + name)
    if destination.exists() and not args.resume:
        raise RuntimeError("Refusing to overwrite an existing build tree")
    prepared_already = args.resume and all((destination / name).is_file() and
        digest(destination / name) == expected for name, expected in manifest['source_sha256'].items())
    if not prepared_already:
        if not args.resume:
            shutil.copytree(original, destination, symlinks=True,
                            ignore=lambda src, names: ["build"] if Path(src) == original and "build" in names else [])
            # Restore official baseline files in the COPY, never the old build.
            for name in manifest["baseline_sha256"]:
                shutil.copyfile(bundle / "baseline_dbe79216" / name, destination / name)
        else:
            # Resume an interrupted preparation only if every baseline file is
            # exact. Never apply the patch over a mixed/unknown source state.
            for name, expected in manifest['baseline_sha256'].items():
                if digest(destination / name) != expected:
                    raise RuntimeError('Interrupted copy is not exact baseline: ' + name)
        old_helper = destination / "libraries/AP_Motors/ROV_DeadzonePWM.h"
        if old_helper.exists():
            known_helpers = {manifest['source_sha256']['libraries/AP_Motors/ROV_DeadzonePWM.h'],
                # Reviewed original task deadzone-only helper (older range guard).
                '954a249355e6378ac9f12ded0d9bc239c0b633787b1d6c6b9d6ac8518613c6b8'}
            if digest(old_helper) not in known_helpers:
                raise RuntimeError("Unexpected old helper; original preserved")
            old_helper.unlink()  # Only duplicate in the new, task-owned build tree.
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=destination, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=destination, check=True)
    for name, expected in manifest["source_sha256"].items():
        if digest(destination / name) != expected:
            raise RuntimeError("Prepared source payload mismatch: " + name)
    env = os.environ.copy()
    env["PATH"] = str(ROOT / "venv/bin") + ":" + str(ROOT / "toolchain/gcc-arm-none-eabi-10-2020-q4-major/bin") + ":" + env["PATH"]
    python = str(ROOT / "venv/bin/python")
    status = bundle / "build_status.json"
    status.write_text(json.dumps({"status": "BUILDING", "source_commit": COMMIT}))
    with (bundle / f"build_{int(time.time())}.log").open("x") as log:
        commands = [[python, "waf", "configure", "--board", "Pixhawk1", "--no-submodule-update"]]
        if args.clean:
            commands.append([python, "waf", "clean"])
        commands.append([python, "waf", "sub", f"-j{args.jobs}"])
        for command in commands:
            try:
                subprocess.run(command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                status.write_text(json.dumps({"status": "BUILD_FAILED", "source_commit": COMMIT,
                                              "exit_code": exc.returncode, "command": command}, indent=2) + "\n")
                raise
    for name, expected in manifest["source_sha256"].items():
        if digest(destination / name) != expected:
            status.write_text(json.dumps({"status": "SOURCE_CHANGED_DURING_BUILD", "path": name}))
            raise RuntimeError("Source changed during build: " + name)
    apj = destination / "build/Pixhawk1/bin/ardusub.apj"
    image = json.loads(apj.read_text())
    binary = zlib.decompress(base64.b64decode(image["image"]))
    if image["board_id"] != 9 or len(binary) != image["image_size"] or len(binary) > 2080768:
        raise RuntimeError("Built image board/size mismatch")
    status.write_text(json.dumps({"status": "BUILT_NOT_FLASHED", "source_commit": COMMIT,
                                 "apj": str(apj), "board_id": image["board_id"],
                                 "image_size": len(binary), "apj_sha256": digest(apj),
                                 "patch_sha256": digest(patch)}, indent=2) + "\n")
    print(status.read_text(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
