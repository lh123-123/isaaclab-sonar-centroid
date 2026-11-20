"""Host-run the exact current-guard mixer and priority allocator."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from firmware.prepare_current_guard import sources, MOTOR, ALLOC
from test_release_mixer import STUBS, extract_function


SCENARIOS = r'''
static float max_output(const AP_Motors6DOF &m) {
    float value=0; for(float x:m._thrust_rpyt_out)value=std::max(value,std::fabs(x)); return value;
}
static void translation_first() {
    AP_Motors6DOF m; m._horizontal_reverse.value=1;
    m._roll_in=.2f; m.throttle_input=1; m._batt_current_max=15;
    m._batt_current_last=battery_sample.amps=15; m._output_limited=.5f;
    m.output_armed_stabilizing();
    require(near(m._output_limited,.5f),"At-threshold envelope must remain stable");
    require(max_output(m)<=.50001f,"Priority allocation exceeded current envelope");
    require(near(realized(m,m._roll_factor),.2f),"Feasible leveling correction was not preserved");
    require(near(realized(m,m._throttle_factor),.3f),"Translation did not use only remaining current envelope");
    require(!m.limit.roll && m.limit.throttle_upper,"Translation, not feasible attitude, must saturate first");
}
static void attitude_bounded() {
    AP_Motors6DOF m; m._horizontal_reverse.value=1;
    m._roll_in=.8f; m.throttle_input=.8f; m._batt_current_max=15;
    m._batt_current_last=battery_sample.amps=15; m._output_limited=.3f;
    m.output_armed_stabilizing();
    require(max_output(m)<=.30001f,"Attitude exceeded hard current envelope");
    require(m.limit.roll && m.limit.pitch && m.limit.throttle_upper,
            "Reduced attitude/depth was not fed back for anti-windup");
}
static void dynamic_limit() {
    AP_Motors6DOF m; m._horizontal_reverse.value=1;
    m._roll_in=.2f; m.throttle_input=1; m._batt_current_max=15;
    m._batt_current_time_constant=.5f; m._dt_s=.02f;
    m._batt_current_last=15; battery_sample.amps=18;
    for(int i=0;i<100;++i) { m.output_armed_stabilizing(); bounds(m); require(max_output(m)<=m._output_limited+.00001f,"Output escaped live envelope"); }
    const float reduced=m._output_limited;
    require(reduced<.5f,"Sustained overcurrent did not reduce output authority");
    battery_sample.amps=5;
    m.output_armed_stabilizing();
    require(m._output_limited>reduced && m._output_limited<1,"Recovery must be gradual, not a jump to full power");
}
static void disabled_is_transparent() {
    AP_Motors6DOF m; m._horizontal_reverse.value=1;
    m._roll_in=.2f; m.throttle_input=.4f; m._batt_current_max=0; m._output_limited=.1f;
    battery_sample.available=false; m.output_armed_stabilizing();
    require(near(m._output_limited,1) && max_output(m)>.1f,"Disabled native current limit changed outputs");
    battery_sample.available=true;
}
int main(int argc,char **argv) {
    require(argc==2,"scenario required"); const std::string s=argv[1];
    if(s=="priority")translation_first(); else if(s=="attitude")attitude_bounded();
    else if(s=="dynamic")dynamic_limit(); else if(s=="disabled")disabled_is_transparent();
    else require(false,"unknown scenario");
}
'''


@unittest.skipUnless(shutil.which('g++'), 'host compiler unavailable')
class CurrentGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary=tempfile.TemporaryDirectory(prefix='rov_current_')
        cls.addClassCleanup(temporary.cleanup)
        folder=Path(temporary.name)
        _, updated=sources()
        stubs=STUBS.replace('Parameter _attitude_priority;',
                            'Parameter _attitude_priority; Parameter _horizontal_reverse;')
        production='\n'.join(extract_function(updated[MOTOR], n) for n in
                            ('output_armed_stabilizing','output_armed_stabilizing_vectored_6dof'))
        (folder/'ROV_AttitudeAllocation.h').write_text(updated[ALLOC],encoding='utf8')
        (folder/'test.cpp').write_text(stubs+production+SCENARIOS,encoding='utf8')
        cls.exe=folder/'test.exe'
        result=subprocess.run(['g++','-std=c++11','-Wall','-Wextra','-Werror','-pedantic',
                               str(folder/'test.cpp'),'-o',str(cls.exe)],capture_output=True,text=True)
        if result.returncode:
            raise AssertionError(result.stdout+result.stderr)

    def scenario(self,name):
        result=subprocess.run([str(self.exe),name],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_translation_is_reduced_before_feasible_attitude(self): self.scenario('priority')
    def test_attitude_is_still_bounded_and_feedback_is_set(self): self.scenario('attitude')
    def test_overcurrent_reduces_and_recovers_smoothly(self): self.scenario('dynamic')
    def test_disabled_native_limit_is_transparent(self): self.scenario('disabled')


if __name__=='__main__': unittest.main()
