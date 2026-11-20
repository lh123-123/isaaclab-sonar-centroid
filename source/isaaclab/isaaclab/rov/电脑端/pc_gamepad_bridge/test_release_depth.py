"""Exercise the generated depth adapter and real latch, never hardware or PID tuning.

The native position controller is represented by an API recorder: this proves
which target API the generated adapter calls, not physical braking performance.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent / "firmware"
sys.path.insert(0, str(ROOT))
from prepare_independent_bundle import sources


def _control_depth_function(source):
    """Compile the generator's function verbatim rather than a behavioral copy."""
    signature = "void ModeAlthold::control_depth()"
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    for index in range(opening + 1, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start:index + 1]
    raise AssertionError("Generated control_depth function is incomplete")


STUBS = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "ROV_DepthHold.h"
#define MIN(a,b) std::min((a),(b))
#define MAX(a,b) std::max((a),(b))
static float constrain_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}
static bool near(float a, float b) { return std::fabs(a-b) < 0.002f; }
class Mode { public: enum class Number { ALT_HOLD=2, DEPTH_ONLY=22, MANUAL=19 }; };
struct Pilot {
    bool on=true;
    ROV_DepthHold::Target target;
    bool enabled() const { return on; }
    ROV_DepthHold::Target &depth_target() { return target; }
    void reset_depth() { target.reset(); }
};
struct ParametersG2 { Pilot rov_pilot; };
struct Parameters {
    float surface_depth=-5.0f, surface_max_throttle=0.2f, pilot_speed_up=150.0f;
};
struct Motors {
    bool active=true;
    float max_throttle=1.0f;
    bool armed() const { return active; }
    void set_max_throttle(float v) { max_throttle=v; }
};
struct Position {
    float desired=-200.0f, slew_target=0.0f, climb_target=0.0f;
    int slew_calls=0, climb_calls=0, updates=0;
    float get_pos_desired_U_cm() const { return desired; }
    void set_pos_desired_U_cm(float v) { desired=v; }
    void set_alt_target_with_slew_cm(float v) { slew_target=v; ++slew_calls; }
    void D_set_pos_target_from_climb_rate_cms(float v) { climb_target=v; ++climb_calls; }
    void D_update_controller() { ++updates; }
};
struct InertialNav {
    float altitude=-200.0f;
    float get_position_z_up_cm() const { return altitude; }
};
struct Ahrs {
    uint32_t reset_ms=0;
    float reset_down_m=0.0f;
    uint32_t getLastPosDownReset(float &delta) const { delta=reset_down_m; return reset_ms; }
};
struct Channel { float get_control_in() const { return 500.0f; } };
struct Sub {
    bool fresh=true;
    float climb_rate=0.0f;
    Mode::Number control_mode=Mode::Number::ALT_HOLD;
    struct { bool at_surface=false, at_bottom=false; } ap;
    bool rov_pilot_input_fresh() const { return fresh; }
    float get_pilot_desired_climb_rate(float) const { return climb_rate; }
    float get_pilot_speed_dn() const { return 150.0f; }
};
static Sub sub;
class ModeAlthold {
public:
    ParametersG2 g2;
    Parameters g;
    Motors motors;
    Position position;
    Position *position_control=&position;
    InertialNav inertial_nav;
    Ahrs ahrs;
    Channel throttle;
    Channel *channel_throttle=&throttle;
    void control_depth();
};
'''


SCENARIOS = r'''
static void begin_motion(ModeAlthold &mode, float rate, float altitude=-200.0f) {
    sub.climb_rate=rate;
    mode.inertial_nav.altitude=altitude;
    mode.control_depth();
    assert(mode.position.climb_calls==1);
    assert(near(mode.position.climb_target, rate));
    assert(mode.position.slew_calls==0);
}
static void release_and_reject_disturbance() {
    ModeAlthold mode;
    begin_motion(mode, -35.0f);
    sub.climb_rate=0;
    mode.inertial_nav.altitude=-214.0f;
    mode.control_depth();
    assert(mode.position.slew_calls==1);
    assert(near(mode.position.slew_target, -214.0f));
    assert(mode.position.climb_calls==1); // Never integrate climb(0) after release.
    mode.inertial_nav.altitude=-227.0f;
    for (int i=0; i<40; ++i) { mode.control_depth(); }
    assert(mode.position.slew_calls==41);
    assert(near(mode.position.slew_target, -214.0f));
    assert(mode.position.updates==42);
}
static void stale_captures_once() {
    ModeAlthold mode;
    begin_motion(mode, 35.0f);
    sub.fresh=false; // Old nonzero manual demand must not remain active.
    mode.inertial_nav.altitude=-186.0f;
    mode.control_depth();
    assert(near(mode.position.slew_target, -186.0f));
    mode.inertial_nav.altitude=-177.0f;
    for (int i=0; i<40; ++i) { mode.control_depth(); }
    assert(near(mode.position.slew_target, -186.0f));
    assert(mode.position.climb_calls==1);
}
static void surface_and_bottom_safety() {
    ModeAlthold surface;
    begin_motion(surface, 35.0f, -10.0f);
    sub.climb_rate=0;
    sub.ap.at_surface=true;
    surface.inertial_nav.altitude=-1.0f;
    surface.control_depth();
    assert(surface.position.slew_calls==1);
    assert(surface.position.slew_target<=surface.g.surface_depth);
    sub.ap.at_surface=false;
    surface.inertial_nav.altitude=-8.0f;
    surface.control_depth();
    assert(surface.position.slew_target<=surface.g.surface_depth);
    sub=Sub{};
    ModeAlthold bottom;
    begin_motion(bottom, -35.0f);
    sub.climb_rate=0;
    sub.ap.at_bottom=true;
    bottom.inertial_nav.altitude=-214.0f;
    bottom.control_depth();
    assert(bottom.position.slew_calls==1);
    assert(bottom.position.slew_target>=-204.0f); // Existing 10cm bottom clearance.
    sub.ap.at_bottom=false;
    bottom.inertial_nav.altitude=-208.0f;
    bottom.control_depth();
    assert(bottom.position.slew_target>=-204.0f);
}
static void symmetric_release_and_mode_handoff() {
    for (const float rate : {-35.0f,35.0f}) {
        sub=Sub{};
        ModeAlthold mode;
        begin_motion(mode, rate);
        const float release_altitude=-200.0f+(rate>0 ? 14.0f : -14.0f);
        sub.climb_rate=0;
        mode.inertial_nav.altitude=release_altitude;
        mode.control_depth();
        assert(near(mode.position.slew_target, release_altitude));
        sub.control_mode=Mode::Number::DEPTH_ONLY;
        mode.inertial_nav.altitude=release_altitude+8.0f;
        mode.control_depth();
        assert(near(mode.position.slew_target, release_altitude));
        sub.control_mode=Mode::Number::ALT_HOLD;
        mode.inertial_nav.altitude=release_altitude-8.0f;
        mode.control_depth();
        assert(near(mode.position.slew_target, release_altitude));
    }
}
static void reset_frame_and_coincident_release() {
    ModeAlthold mode;
    begin_motion(mode, -35.0f);
    sub.climb_rate=0;
    mode.inertial_nav.altitude=-214.0f;
    mode.control_depth();
    mode.ahrs.reset_ms=100;
    mode.ahrs.reset_down_m=1.0f;
    mode.inertial_nav.altitude=-314.0f;
    mode.control_depth();
    assert(near(mode.position.slew_target, -314.0f));
    mode.control_depth();
    assert(near(mode.position.slew_target, -314.0f)); // Same reset never shifts twice.
    sub.climb_rate=-35.0f;
    mode.control_depth();
    mode.ahrs.reset_ms=101;
    mode.ahrs.reset_down_m=1.0f;
    mode.inertial_nav.altitude=-430.0f;
    sub.climb_rate=0;
    mode.control_depth();
    assert(near(mode.position.slew_target, -430.0f)); // Capture is already in new frame.
}
static void disarmed_and_stock_bypass() {
    ModeAlthold mode;
    begin_motion(mode, -35.0f);
    const int updates=mode.position.updates;
    mode.motors.active=false;
    sub.climb_rate=0;
    mode.control_depth();
    assert(mode.position.updates==updates);
    assert(mode.position.slew_calls==0);
    mode.motors.active=true;
    mode.inertial_nav.altitude=-250.0f;
    mode.control_depth();
    assert(mode.position.slew_calls==1);
    assert(near(mode.position.slew_target,-250.0f));
    sub=Sub{};
    ModeAlthold stock;
    stock.g2.rov_pilot.on=false;
    stock.control_depth();
    assert(stock.position.slew_calls==0 && stock.position.climb_calls==1);
    ModeAlthold other;
    sub.control_mode=Mode::Number::MANUAL;
    other.control_depth();
    assert(other.position.slew_calls==0 && other.position.climb_calls==1);
}
int main(int argc, char **argv) {
    assert(argc==2);
    switch (std::atoi(argv[1])) {
    case 1: release_and_reject_disturbance(); break;
    case 2: stale_captures_once(); break;
    case 3: surface_and_bottom_safety(); break;
    case 4: symmetric_release_and_mode_handoff(); break;
    case 5: reset_frame_and_coincident_release(); break;
    case 6: disarmed_and_stock_bypass(); break;
    default: assert(false);
    }
    std::cout << "Depth release adapter scenario passed\n";
}
'''


class DepthReleaseGeneratorTests(unittest.TestCase):
    def test_generated_depth_adapter_uses_position_slew_hold(self):
        _, updated = sources(ROOT / "baseline_dbe79216")
        function = _control_depth_function(updated["ArduSub/mode_althold.cpp"])
        self.assertIn("set_alt_target_with_slew_cm", function)
        self.assertIn("getLastPosDownReset", function)
        self.assertIn("depth_target()", function)
        self.assertIn("D_set_pos_target_from_climb_rate_cms", function)
        self.assertIn("D_update_controller", function)


@unittest.skipUnless(shutil.which("g++"), "Host C++ compiler unavailable")
class DepthReleaseAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, updated = sources(ROOT / "baseline_dbe79216")
        function = _control_depth_function(updated["ArduSub/mode_althold.cpp"])
        cls.directory = tempfile.TemporaryDirectory(prefix="rov-depth-release-")
        cls.addClassCleanup(cls.directory.cleanup)
        directory = Path(cls.directory.name)
        source = directory / "depth_release.cpp"
        cls.exe = directory / "depth_release.exe"
        source.write_text(STUBS + function + SCENARIOS, encoding="utf-8")
        result = subprocess.run(
            ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
             "-I" + str(ROOT), str(source), "-o", str(cls.exe)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def scenario(self, number):
        result = subprocess.run(
            [str(self.exe), str(number)], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Depth release adapter scenario passed", result.stdout)

    def test_release_uses_actual_position_api_and_does_not_track_disturbance(self):
        self.scenario(1)

    def test_stale_manual_input_captures_once(self):
        self.scenario(2)

    def test_surface_and_bottom_protection_persists_in_shared_target(self):
        self.scenario(3)

    def test_up_down_symmetry_and_independent_mode_handoff(self):
        self.scenario(4)

    def test_ekf_reset_and_same_tick_release_do_not_double_shift(self):
        self.scenario(5)

    def test_disarmed_reset_and_stock_mode_bypass(self):
        self.scenario(6)


if __name__ == "__main__":
    unittest.main()
