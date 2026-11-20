"""Execute actual release mixer with the user's observed reversed horizontal plant.

The plant encodes reported directions, not measured dynamics or compass accuracy.
Host compilation and this simple closed loop do not certify water stability.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from firmware.prepare_horizontal_direction import sources, MOTOR_CPP
from test_release_mixer import STUBS, extract_function


SCENARIOS = r'''
static float physical_yaw(const AP_Motors6DOF &m) {
    return -realized(m,m._yaw_factor);
}
static float physical_lateral(const AP_Motors6DOF &m) {
    return -realized(m,m._lateral_factor);
}
static void directions() {
    for (int priority : {0,1}) {
        for (int sign : {-1,1}) {
            AP_Motors6DOF m;
            m._attitude_priority.value=priority;
            m._yaw_in=sign*.3f;
            m.output_armed_stabilizing();
            require(near(physical_yaw(m),sign*.3f),"Yaw output still turns opposite to demand");
            m._yaw_in=0; m._lateral_in=sign*.4f;
            m.output_armed_stabilizing();
            require(near(physical_lateral(m),sign*.4f),"Lateral output still moves opposite to demand");
            bounds(m);
        }
    }
}
static void unchanged_axes() {
    for (int priority : {0,1}) {
        for (int axis=0;axis<4;++axis) {
            for (int sign : {-1,1}) {
                AP_Motors6DOF m;
                m._attitude_priority.value=priority;
                if(axis==0)m._forward_in=sign*.4f;
                if(axis==1)m.throttle_input=sign*.4f;
                if(axis==2)m._roll_in=sign*.4f;
                if(axis==3)m._pitch_in=sign*.4f;
                m._horizontal_reverse.value=0;
                m.output_armed_stabilizing();
                float before[8];
                std::copy(m._thrust_rpyt_out,m._thrust_rpyt_out+8,before);
                m._horizontal_reverse.value=1;
                m.output_armed_stabilizing();
                for(int i=0;i<8;++i)require(near(before[i],m._thrust_rpyt_out[i]),"Verified axis changed");
            }
        }
    }
}
static void restoring() {
    for(int priority : {0,1}) {
        for(float error : {-170.f,-20.f,-1.f,1.f,20.f,170.f}) {
            AP_Motors6DOF m;
            m._attitude_priority.value=priority;
            // A stationary body with positive native yaw needs negative torque.
            m._yaw_in=constrain_float(-error*.04f,-1,1);
            m.output_armed_stabilizing();
            require(error*physical_yaw(m)<0,"Heading correction reinforces error instead of restoring");
        }
    }
    for(int correction : {0,1}) {
        AP_Motors6DOF m;
        m._horizontal_reverse.value=correction;
        float heading=10,rate=0;
        for(int step=0;step<1000;++step) {
            m._yaw_in=constrain_float(-.04f*heading-.08f*rate,-1,1);
            m.output_armed_stabilizing();
            rate+=(20*physical_yaw(m)-2*rate)*.01f;
            heading+=rate*.01f;
            bounds(m);
        }
        // The deliberately overdamped model must remove at least 80% of its
        // initial error in 10 s; this is a sign check, not a vehicle settling spec.
        require(correction ? std::fabs(heading)<2.f : std::fabs(heading)>10,
                "Observed reversed plant did not switch from divergence to restoring feedback");
    }
}
static void saturation() {
    AP_Motors6DOF m;
    m._yaw_in=.3f; m._forward_in=1; m._lateral_in=1;
    m._roll_in=.4f; m._pitch_in=.2f; m.throttle_input=1;
    m.output_armed_stabilizing();
    bounds(m);
    require(near(physical_yaw(m),.3f),"Heading priority lost after direction correction");
    require(near(realized(m,m._forward_factor),physical_lateral(m)) && physical_lateral(m)>0,
            "Diagonal travel ratio/direction changed under saturation");
    require(m.limit.throttle_upper,"Depth saturation feedback lost");
    m._yaw_in=2;
    m.output_armed_stabilizing();
    require(m.limit.yaw,"Yaw saturation feedback lost");
    bounds(m);
}
static void invalid_parameter() {
    for(int invalid : {-1,2,127}) {
        AP_Motors6DOF m;
        m._horizontal_reverse.value=invalid;
        m._yaw_in=.3f; m._forward_in=.4f; m.throttle_input=.2f;
        m.output_armed_stabilizing();
        for(float v:m._thrust_rpyt_out) require(near(v,0),"Invalid geometry parameter must inhibit outputs");
        require(m.limit.yaw && m.limit.throttle_upper,"Invalid geometry did not report controller limits");
    }
}
int main(int argc,char **argv) {
    require(argc==2,"One scenario required");
    std::string name=argv[1];
    if(name=="directions")directions();
    else if(name=="unchanged")unchanged_axes();
    else if(name=="restoring")restoring();
    else if(name=="saturation")saturation();
    else if(name=="invalid")invalid_parameter();
    else require(false,"Unknown scenario");
}
'''


class HorizontalDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('Host C++ compiler unavailable')
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        _, updated = sources()
        functions = '\n'.join(extract_function(updated[MOTOR_CPP], name) for name in (
            'output_armed_stabilizing', 'output_armed_stabilizing_vectored_6dof'))
        stubs = STUBS.replace('Parameter _attitude_priority;',
                              'Parameter _attitude_priority; Parameter _horizontal_reverse;')
        (folder / 'ROV_AttitudeAllocation.h').write_text(
            updated['libraries/AP_Motors/ROV_AttitudeAllocation.h'], encoding='utf8')
        source = folder / 'horizontal.cpp'
        source.write_text(stubs+functions+SCENARIOS, encoding='utf8')
        cls.executable = folder / 'horizontal.exe'
        result = subprocess.run([compiler, '-std=c++11', '-Wall', '-Wextra', '-Werror',
                                 str(source), '-o', str(cls.executable)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout+result.stderr)

    def run_scenario(self, name):
        result = subprocess.run([str(self.executable), name], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_manual_yaw_and_lateral_follow_reported_physical_directions(self):
        self.run_scenario('directions')

    def test_forward_heave_roll_pitch_unchanged(self):
        self.run_scenario('unchanged')

    def test_heading_feedback_restores_instead_of_amplifying_error(self):
        self.run_scenario('restoring')

    def test_combined_motion_retains_priority_bounds_and_saturation_feedback(self):
        self.run_scenario('saturation')

    def test_invalid_direction_setting_inhibits_output(self):
        self.run_scenario('invalid')

    def test_release_preserves_deployed_pid_depth_pwm_and_other_sources(self):
        original, updated = sources()
        changed = {name for name in updated if original.get(name) != updated[name]}
        self.assertEqual(changed, {MOTOR_CPP, 'libraries/AP_Motors/AP_Motors6DOF.h'})


if __name__ == '__main__':
    unittest.main()
