"""Offline execution of the generated release mixer, not a copied mixer model.

Only the two production mixing functions are extracted. Minimal host stubs supply
inputs, coefficients, limits and battery samples; no hardware or network is used.
This is not a complete firmware build, SITL or a physical stability acceptance.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from firmware.prepare_control_bundle import ROOT, build_sources


def extract_function(source, name):
    marker = "void AP_Motors6DOF::" + name + "()"
    start = source.index(marker)
    opening = source.index("{", start + len(marker))
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError("Unclosed production mixer function: " + name)


STUBS = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include "ROV_AttitudeAllocation.h"

#define AP_MOTORS_MAX_NUM_MOTORS 8
#define AP_BATTERY_ENABLED 1
enum sub_frame_t { SUB_FRAME_OTHER, SUB_FRAME_VECTORED, SUB_FRAME_VECTORED_6DOF };
inline bool is_zero(float value) { return std::fabs(value) < 1.0e-6f; }
inline float constrain_float(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}
struct Parameter {
    int value = 1;
    int get() const { return value; }
};
struct Limits {
    bool roll = false, pitch = false, yaw = false;
    bool throttle_lower = false, throttle_upper = false;
    void set_all(bool value) {
        roll = pitch = yaw = throttle_lower = throttle_upper = value;
    }
    void set_rpy(bool value) { roll = pitch = yaw = value; }
    void set_throttle(bool value) { throttle_lower = throttle_upper = value; }
};
struct AP_BattMonitor {
    bool available = true;
    float amps = 10;
    bool current_amps(float &result) const {
        result = amps;
        return available;
    }
};
static AP_BattMonitor battery_sample;
namespace AP {
    inline const AP_BattMonitor &battery() { return battery_sample; }
}

class AP_Motors6DOF {
public:
    int _active_frame_class = SUB_FRAME_VECTORED_6DOF;
    Parameter _attitude_priority;
    Limits limit;
    float _roll_in = 0, _roll_in_ff = 0;
    float _pitch_in = 0, _pitch_in_ff = 0;
    float _yaw_in = 0, _yaw_in_ff = 0;
    float _forward_in = 0, _lateral_in = 0, throttle_input = 0;
    float _throttle_thrust_max = 1, _max_throttle = 1;
    float _batt_current_max = 0, _batt_current_last = 10;
    float _batt_current_time_constant = 5, _dt_s = 0.02f;
    float _output_limited = 1;
    bool motor_enabled[8] = {true,true,true,true,true,true,true,true};
    float _motor_reverse[8] = {1,-1,-1,1,1,-1,-1,1};
    // The pinned SUB_FRAME_VECTORED_6DOF logical coefficients, unchanged.
    float _roll_factor[8] = {0,0,0,0,1,-1,1,-1};
    float _pitch_factor[8] = {0,0,0,0,-1,-1,1,1};
    float _yaw_factor[8] = {1,-1,-1,1,0,0,0,0};
    float _throttle_factor[8] = {0,0,0,0,-1,-1,-1,-1};
    float _forward_factor[8] = {-1,-1,1,1,0,0,0,0};
    float _lateral_factor[8] = {1,-1,1,-1,0,0,0,0};
    float _thrust_rpyt_out[8] = {};
    float get_throttle_bidirectional() const { return throttle_input; }
    void output_armed_stabilizing();
    void output_armed_stabilizing_vectored_6dof();
    void output_armed_stabilizing_vectored() {}
};

class StockMotors : public AP_Motors6DOF {
public:
    void output_armed_stabilizing();
    void output_armed_stabilizing_vectored_6dof();
};

static void require(bool condition, const char *message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
static bool near(float a, float b) { return std::fabs(a-b) < 2.0e-5f; }
static float realized(const AP_Motors6DOF &motor, const float *factor) {
    float sum = 0;
    for (int i=0; i<8; ++i) {
        sum += motor._thrust_rpyt_out[i] * motor._motor_reverse[i] * factor[i];
    }
    return sum / 4;
}
static void bounds(const AP_Motors6DOF &motor) {
    for (float output : motor._thrust_rpyt_out) {
        require(std::isfinite(output) && std::fabs(output)<=1.00001f,
                "Generated mixer exceeded normalized output bounds");
    }
}
'''


SCENARIOS = r'''
static void near_surface() {
    AP_Motors6DOF motor;
    motor.throttle_input = 0.8f;
    motor._max_throttle = 0.1f;
    motor.output_armed_stabilizing();
    require(near(realized(motor, motor._throttle_factor), 0.1f),
            "Near-surface limiter must retain its existing output reduction");
    require(motor.limit.throttle_upper,
            "Near-surface upper clipping was not reported to depth anti-windup");
    require(!motor.limit.throttle_lower,
            "Positive near-surface clipping should not be a lower limit");
}

static void current_feedback() {
    for (int direction : {-1,1}) {
        AP_Motors6DOF motor;
        motor._roll_in = 0.15f;
        motor._pitch_in = 0.1f;
        motor._yaw_in = 0.2f;
        motor.throttle_input = direction * 0.3f;
        motor._batt_current_max = 10;
        motor._batt_current_last = battery_sample.amps = 10;
        motor._output_limited = 0.5f;
        motor.output_armed_stabilizing();
        bounds(motor);
        require(near(realized(motor,motor._roll_factor),0.075f) &&
                near(realized(motor,motor._pitch_factor),0.05f) &&
                near(realized(motor,motor._yaw_factor),0.1f) &&
                near(realized(motor,motor._throttle_factor),direction*0.15f),
                "Current limiter's existing common output scaling changed");
        require(motor.limit.roll && motor.limit.pitch && motor.limit.yaw,
                "Current-limited attitude requests were not reported to anti-windup");
        require(direction>0 ? motor.limit.throttle_upper : motor.limit.throttle_lower,
                "Current-limited depth request was not reported to directional anti-windup");
    }
    AP_Motors6DOF unrestricted;
    unrestricted._roll_in = 0.15f;
    unrestricted._pitch_in = 0.1f;
    unrestricted._yaw_in = 0.2f;
    unrestricted.throttle_input = 0.3f;
    unrestricted._batt_current_max = 10;
    unrestricted._batt_current_last = battery_sample.amps = 10;
    unrestricted._output_limited = 1;
    unrestricted.output_armed_stabilizing();
    require(!unrestricted.limit.roll && !unrestricted.limit.pitch &&
            !unrestricted.limit.yaw && !unrestricted.limit.throttle_upper &&
            !unrestricted.limit.throttle_lower,
            "Unrestricted requests must not acquire current-limit flags");
}

static void attitude_priority() {
    for (int direction : {-1,1}) {
        AP_Motors6DOF motor;
        motor._roll_in = 0.4f;
        motor._pitch_in = 0.2f;
        motor._yaw_in = 0.3f;
        motor._forward_in = 1;
        motor._lateral_in = 1;
        motor.throttle_input = static_cast<float>(direction);
        motor.output_armed_stabilizing();
        bounds(motor);
        require(near(realized(motor,motor._roll_factor),0.4f) &&
                near(realized(motor,motor._pitch_factor),0.2f) &&
                near(realized(motor,motor._yaw_factor),0.3f),
                "Full translation must not dilute feasible rotation requests");
        require(near(realized(motor,motor._throttle_factor),direction*0.4f),
                "Lift should use the remaining vertical rotation headroom");
        const float forward = realized(motor,motor._forward_factor);
        const float lateral = realized(motor,motor._lateral_factor);
        require(forward>0 && forward<1 && near(forward,lateral),
                "Horizontal translation scaling must retain forward:lateral ratio");
        require(!motor.limit.roll && !motor.limit.pitch && !motor.limit.yaw,
                "Preserved attitude requests should not report rotational saturation");
        require(direction>0 ? motor.limit.throttle_upper : motor.limit.throttle_lower,
                "Reduced lift must report directional depth saturation");
    }
}

static bool same_limits(const Limits &a,const Limits &b) {
    return a.roll==b.roll && a.pitch==b.pitch && a.yaw==b.yaw &&
           a.throttle_lower==b.throttle_lower && a.throttle_upper==b.throttle_upper;
}
static void stock_fallback() {
    for (int variant=0; variant<4; ++variant) {
        AP_Motors6DOF release;
        release._attitude_priority.value = 0;
        release._roll_in = 0.4f;
        release._pitch_in = 0.2f;
        release._yaw_in = 0.3f;
        release._forward_in = 0.9f;
        release._lateral_in = -0.5f;
        release.throttle_input = variant==3 ? -0.8f : 0.8f;
        if (variant==1) { release._max_throttle = 0.1f; }
        if (variant>=2) {
            release._batt_current_max = 10;
            release._batt_current_last = battery_sample.amps = 10;
            release._output_limited = 0.5f;
        }
        StockMotors stock;
        static_cast<AP_Motors6DOF &>(stock) = release;
        release.output_armed_stabilizing();
        stock.output_armed_stabilizing();
        bounds(release);
        for (int i=0;i<8;++i) {
            require(near(release._thrust_rpyt_out[i],stock._thrust_rpyt_out[i]),
                    "Feature-disabled outputs diverged from pinned stock mixer");
        }
        require(same_limits(release.limit,stock.limit),
                "Feature-disabled saturation flags diverged from pinned stock mixer");
    }
}

int main(int argc,char **argv) {
    require(argc==2,"Provide one release mixer scenario");
    const std::string name = argv[1];
    if (name=="surface") { near_surface(); }
    else if (name=="current") { current_feedback(); }
    else if (name=="priority") { attitude_priority(); }
    else if (name=="stock") { stock_fallback(); }
    else { require(false,"Unknown release mixer scenario"); }
    std::cout << name << " release mixer scenario passed\n";
}
'''


@unittest.skipUnless(shutil.which("g++"), "Host C++ compiler unavailable")
class ReleaseMixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        original, updated = build_sources(ROOT / "baseline_dbe79216")
        motor_path = "libraries/AP_Motors/AP_Motors6DOF.cpp"
        function_names = ("output_armed_stabilizing",
                          "output_armed_stabilizing_vectored_6dof")
        production = "\n\n".join(extract_function(updated[motor_path], name)
                                   for name in function_names)
        stock = "\n\n".join(extract_function(original[motor_path], name)
                              for name in function_names)
        stock = stock.replace("AP_Motors6DOF::", "StockMotors::")
        allocation_path = "libraries/AP_Motors/ROV_AttitudeAllocation.h"
        (directory / "ROV_AttitudeAllocation.h").write_text(updated[allocation_path], encoding="utf-8")
        source = directory / "release_mixer.cpp"
        source.write_text(STUBS + production + stock + SCENARIOS, encoding="utf-8")
        cls.executable = directory / "release_mixer.exe"
        result = subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                                 str(source), "-o", str(cls.executable)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError("Generated production mixer did not compile:\n" +
                                 result.stdout + result.stderr)

    def run_scenario(self, name):
        result = subprocess.run([str(self.executable), name], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(name + " release mixer scenario passed", result.stdout)

    def test_near_surface_reports_upper_limit(self):
        self.run_scenario("surface")

    def test_current_limit_reports_attitude_and_depth_saturation(self):
        self.run_scenario("current")

    def test_full_translation_retains_feasible_attitude_requests(self):
        self.run_scenario("priority")

    def test_feature_disabled_preserves_pinned_stock_outputs_and_flags(self):
        self.run_scenario("stock")


if __name__ == "__main__":
    unittest.main()
