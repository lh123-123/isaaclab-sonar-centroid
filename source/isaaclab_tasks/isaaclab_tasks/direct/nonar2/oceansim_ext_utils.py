# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve OceanSim extension paths inside conda / pip Isaac Sim installs."""

from __future__ import annotations

import os
import sys
import sysconfig


def resolve_isaacsim_exts_user_folder() -> str:
    """Return absolute path to ``isaacsim/extsUser`` (where OceanSim lives).

    Search order:
    1. ``ISAACSIM_EXTENSIONS_FOLDER`` environment variable
    2. Directory adjacent to the installed ``isaacsim`` Python package
    3. Current interpreter ``site-packages/isaacsim/extsUser``
    """
    env = os.environ.get("ISAACSIM_EXTENSIONS_FOLDER", "").strip()
    if env and os.path.isdir(env):
        path = os.path.abspath(env)
        os.environ.setdefault("ISAACSIM_EXTENSIONS_FOLDER", path)
        return path

    candidates: list[str] = []
    try:
        import isaacsim

        candidates.append(os.path.join(os.path.dirname(os.path.abspath(isaacsim.__file__)), "extsUser"))
    except Exception:
        pass

    try:
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            candidates.append(os.path.join(purelib, "isaacsim", "extsUser"))
    except Exception:
        pass

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            path = os.path.abspath(candidate)
            os.environ.setdefault("ISAACSIM_EXTENSIONS_FOLDER", path)
            return path

    legacy = os.path.abspath(
        os.path.join(
            os.path.dirname(sys.executable),
            "..",
            "..",
            "..",
            "site-packages",
            "isaacsim",
            "extsUser",
        )
    )
    if os.path.isdir(legacy):
        os.environ.setdefault("ISAACSIM_EXTENSIONS_FOLDER", legacy)
        return legacy

    raise FileNotFoundError(
        "OceanSim extsUser folder not found. Set ISAACSIM_EXTENSIONS_FOLDER to your "
        "isaacsim/extsUser path (e.g. "
        "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extsUser)."
    )


def isaacsim_exts_folder(exts_user_folder: str) -> str:
    """``site-packages/isaacsim/exts`` from ``.../isaacsim/extsUser``."""
    isaacsim_root = os.path.dirname(os.path.abspath(exts_user_folder))
    return os.path.join(isaacsim_root, "exts")


def ensure_oceansim_kit_extensions(exts_user_folder: str) -> None:
    """Enable Isaac Sim kit extensions required by OceanSim imaging sonar."""
    import omni.kit.app

    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    # OceanSim itself is loaded from extsUser via importlib; kit registry may not list it.
    if not ext_mgr.is_extension_enabled("isaacsim.sensors.camera"):
        ext_mgr.set_extension_enabled_immediate("isaacsim.sensors.camera", True)

    camera_ext_dir = os.path.join(isaacsim_exts_folder(exts_user_folder), "isaacsim.sensors.camera")
    if os.path.isdir(camera_ext_dir) and camera_ext_dir not in sys.path:
        sys.path.insert(0, camera_ext_dir)

    oceansim_root = oceansim_extension_root(exts_user_folder)
    sensor_py = os.path.join(oceansim_root, "isaacsim", "oceansim", "sensors", "ImagingSonarSensor.py")
    if not os.path.isfile(sensor_py):
        raise FileNotFoundError(f"OceanSim ImagingSonarSensor.py not found at {sensor_py}")


def oceansim_extension_root(exts_user_folder: str | None = None) -> str:
    """Path to ``isaacsim.oceansim`` under extsUser."""
    root = exts_user_folder or resolve_isaacsim_exts_user_folder()
    return os.path.join(root, "isaacsim.oceansim")
