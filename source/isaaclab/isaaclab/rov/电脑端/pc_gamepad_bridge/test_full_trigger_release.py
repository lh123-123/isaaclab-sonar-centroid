"""Integration-shaped regression for the full-release depth latch."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pc_gamepad_bridge.test_release_depth import STUBS, _control_depth_function


ROOT = Path(__file__).resolve().parent / "firmware"
SOURCE = ROOT / "prepared_full_trigger_release_safe_20261013" / "source"

STUBS_WITH_CAP = STUBS.replace(
    "    bool active=true;",
    "    bool active=true; bool hold_active=false;\n"
    "    void set_depth_hold_active(bool value) { hold_active=value; }",
).replace(
    "struct Channel { float get_control_in() const { return 500.0f; } };",
    "struct Channel { float control=500.0f;\n"
    "    float get_control_in() const { return control; }\n"
    "    float get_control_mid() const { return 500.0f; } };",
)

SCENARIO = r'''
int main() {
    ModeAlthold mode;
    sub.climb_rate=-30.0f;
    mode.throttle.control=300.0f;
    mode.control_depth();
    assert(mode.position.climb_calls==1);

    // The commanded climb is already zero inside THR_DZ, but the pilot has
    // not finished releasing the trigger. No position latch yet, but the
    // original zero-climb DH cap must remain active for electrical safety.
    sub.climb_rate=0.0f;
    mode.throttle.control=550.0f;
    mode.inertial_nav.altitude=-210.0f;
    mode.control_depth();
    assert(mode.position.slew_calls==0);
    assert(mode.position.climb_calls==2);
    assert(mode.motors.hold_active);

    // At the true neutral input, capture this tick's measured depth once.
    mode.throttle.control=500.0f;
    mode.inertial_nav.altitude=-216.0f;
    mode.control_depth();
    assert(mode.position.slew_calls==1);
    assert(near(mode.position.slew_target,-216.0f));
    assert(mode.motors.hold_active);
    mode.inertial_nav.altitude=-220.0f;
    mode.control_depth();
    assert(near(mode.position.slew_target,-216.0f));
}
'''


@unittest.skipUnless(shutil.which("g++"), "Host C++ compiler unavailable")
class FullTriggerReleaseTests(unittest.TestCase):
    def test_target_waits_for_true_release_while_cap_stays_active(self):
        function = _control_depth_function(
            (SOURCE / "ArduSub" / "mode_althold.cpp").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="rov-full-release-") as folder:
            source = Path(folder) / "test.cpp"
            exe = Path(folder) / "test.exe"
            source.write_text(STUBS_WITH_CAP + function + SCENARIO, encoding="utf-8")
            compiled = subprocess.run(
                ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                 "-I" + str(SOURCE / "ArduSub"), str(source), "-o", str(exe)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
