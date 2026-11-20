"""Convert a CAD STEP/STP file to USD using the Isaac Sim Hoops converter.

Usage:
    conda run -n isaaclab python cad_to_usd.py INPUT.STEP OUTPUT.usd [config.json]
"""

import json
import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])
config_path = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else ""

if not os.path.isfile(input_path):
    raise SystemExit(f"Input not found: {input_path}")

from isaacsim.simulation_app import SimulationApp

app = SimulationApp({"headless": True, "extra_args": ["--enable", "omni.kit.converter.hoops_core"]})

import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
for _ in range(100):
    app.update()
    if ext_manager.is_extension_enabled("omni.kit.converter.hoops_core"):
        break

from omni.kit.converter.hoops_core import get_instance, is_format_supported

print("FORMAT_SUPPORTED:", is_format_supported(input_path))

file_format_args = {}
if config_path and os.path.isfile(config_path):
    with open(config_path) as f:
        data = json.load(f)
        for k, v in data.items():
            file_format_args[k] = str(v).lower() if isinstance(v, bool) else str(v)


async def convert():
    cad = get_instance()
    progress = await cad.create_converter_task(input_path, output_path, file_format_args)
    print("RESULT:", progress)
    return progress


result = app.run_coroutine(convert(), run_until_complete=True)
print("DONE RESULT:", result)
print("OUTPUT_EXISTS:", os.path.exists(output_path))
print("OUTPUT:", output_path)

app.close()
