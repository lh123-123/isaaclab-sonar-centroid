"""Compile the candidate's actual final-output/test functions, with host I/O stubs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from firmware.prepare_output_envelope import ROOT, sources


def extract(source, name):
    marker = 'AP_Motors6DOF::' + name + '('
    location = source.index(marker)
    start = source.rfind('\n', 0, location) + 1
    opening = source.index('{', location)
    depth = 0
    for end in range(opening, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end+1]
    raise ValueError('Unclosed function: ' + name)


STUBS = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include "ROV_OutputEnvelope.h"
#include "ROV_DeadzonePWM.h"
#define AP_MOTORS_MAX_NUM_MOTORS 8
static uint32_t clock_us = 1000000;
namespace AP_HAL { inline uint32_t micros() { return clock_us; } }
struct Parameter { float value; Parameter(float v) : value(v) {} float get() const { return value; } };
struct Limits {
    bool roll=false, pitch=false, yaw=false, throttle_lower=false, throttle_upper=false;
    void set_all(bool v) { roll=pitch=yaw=throttle_lower=throttle_upper=v; }
    void set_rpy(bool v) { roll=pitch=yaw=v; }
    void set_throttle(bool v) { throttle_lower=throttle_upper=v; }
};
class AP_Motors6DOF {
public:
    enum class SpoolState { SHUT_DOWN, GROUND_IDLE, SPOOLING_UP, THROTTLE_UNLIMITED, SPOOLING_DOWN };
    SpoolState _spool_state = SpoolState::THROTTLE_UNLIMITED;
    bool is_armed=true;
    bool motor_enabled[8]={true,true,true,true,true,true,true,true};
    uint8_t _test_order[8]={1,2,3,4,5,6,7,8};
    float _thrust_rpyt_out[8]={};
    float _dt_s=0.02f;
    Parameter _rov_output_cap{20}, _rov_output_sum{1}, _rov_output_rise{0.5f};
    Parameter _startup_deadzone_pwm{50}, _startup_deadzone_input{0.01f};
    ROV_OutputEnvelope::Envelope<8> _rov_envelope;
    bool _rov_output_limited=false, _rov_test_active=false;
    uint32_t _rov_test_last_us=0;
    int16_t _rov_test_pwm[8]={};
    int16_t written[8]={1500,1500,1500,1500,1500,1500,1500,1500};
    int16_t pwm_min=1000, pwm_max=2000;
    Limits limit;
    bool armed() const { return is_armed; }
    int16_t get_pwm_output_min() const { return pwm_min; }
    int16_t get_pwm_output_max() const { return pwm_max; }
    void rc_write(uint8_t i, int16_t value) { written[i]=value; }
    int16_t calc_thrust_to_pwm(float value) const;
    void output_min();
    void output_to_motors();
    void rov_guard_outputs(int16_t *pwm, float dt, bool active);
    bool output_test_num(uint8_t motor, int16_t pwm);
    void _output_test_seq(uint8_t seq, int16_t pwm);
};
class BaselineMotors : public AP_Motors6DOF {
public: void output_to_motors();
};
static void require(bool condition, const char *message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
static float excursion(int16_t p,int lo=1000,int hi=2000) {
    return (p-1500)/float(p>=1500?hi-1500:1500-lo);
}
static void bounds(const AP_Motors6DOF &m) {
    float sum=0;
    for (int i=0;i<8;++i) {
        const float v=std::fabs(excursion(m.written[i],m.pwm_min,m.pwm_max));
        require(v<=m._rov_output_cap.get()/100.0f+1e-6f,"Final per-motor PWM cap exceeded");
        sum+=v;
    }
    require(sum<=m._rov_output_sum.get()+1e-5f,"Aggregate excursion budget exceeded");
}
static void full(AP_Motors6DOF &m) { for (int i=0;i<8;++i) { m._thrust_rpyt_out[i]=i%2?1:-1; } }
static void settle(AP_Motors6DOF &m) { for(int n=0;n<100;++n) { m.limit.set_all(false);m.output_to_motors();bounds(m); } }
'''

SCENARIOS = r'''
static void run_case(const std::string &name) {
    AP_Motors6DOF m;
    if(name=="baseline") {
        BaselineMotors old;full(old);old.output_to_motors();
        require(old.written[0]==1000&&old.written[1]==2000,"Old final-output path no longer matches the observed uncapped configuration");
        full(m);settle(m);bounds(m);
        require(std::abs(m.written[0]-1500)<std::abs(old.written[0]-1500),"Candidate did not eliminate uncapped request path");
    } else if(name=="caps") {
        full(m);m._rov_output_sum.value=8;settle(m);
        require(m.written[0]>=1400&&m.written[1]<=1600,"20 percent must bound final PWM, including deadzone offset");
        require(m._rov_output_limited&&m.limit.roll&&m.limit.pitch&&m.limit.yaw&&m.limit.throttle_upper&&m.limit.throttle_lower,"Final clipping was hidden from integrators");
    } else if(name=="sum") {
        full(m);m._rov_output_cap.value=100;settle(m);
        for(int p:m.written) { require(std::abs(p-1500)<=63,"All eight motors exceeded shared budget"); }
    } else if(name=="slew") {
        full(m);m.output_to_motors();
        for(int p:m.written) { require(std::abs(p-1500)<=5,"First enabled output jumped"); }
        settle(m);for(float &v:m._thrust_rpyt_out) { v=0; }m.output_to_motors();
        for(int p:m.written) { require(p==1500,"Neutral must be immediate"); }
    } else if(name=="reverse") {
        m._thrust_rpyt_out[0]=1;settle(m);m._thrust_rpyt_out[0]=-1;m.output_to_motors();
        require(m.written[0]==1500,"Reversal failed to cross neutral");m.output_to_motors();
        require(m.written[0]>=1495&&m.written[0]<=1500,"Reversal bypassed rise limit");
    } else if(name=="inactive") {
        full(m);settle(m);m.is_armed=false;m.output_to_motors();
        for(int p:m.written) { require(p==1500,"Disarm left propulsion active"); }
        m.is_armed=true;m._spool_state=AP_Motors6DOF::SpoolState::GROUND_IDLE;m.output_to_motors();
        for(int p:m.written) { require(p==1500,"Idle left propulsion active"); }
        m._spool_state=AP_Motors6DOF::SpoolState::THROTTLE_UNLIMITED;m.output_to_motors();
        for(int p:m.written) { require(std::abs(p-1500)<=5,"Rearm inherited previous output"); }
    } else if(name=="invalid") {
        const float nan=std::numeric_limits<float>::quiet_NaN();
        for(float value:{-1.0f,101.0f,nan}) {
            AP_Motors6DOF x;full(x);x._rov_output_cap.value=value;x.output_to_motors();
            for(int p:x.written) { require(p==1500,"Invalid configuration failed open"); }
        }
        full(m);m._dt_s=nan;m.output_to_motors();for(int p:m.written) { require(p==1500,"Invalid dt failed open"); }
    } else if(name=="zero") {
        full(m);settle(m);m._rov_output_cap.value=0;m.output_to_motors();
        for(int p:m.written) { require(p==1500,"Zero cap must inhibit, not disable protection"); }
    } else if(name=="asymmetric") {
        m.pwm_min=1100;full(m);m._rov_output_sum.value=8;settle(m);
        require(m.written[0]>=1420&&m.written[1]<=1600,"Asymmetric endpoint fraction incorrect");
    } else if(name=="disabled") {
        full(m);m.motor_enabled[0]=false;settle(m);require(m.written[0]==1500,"Disabled motor actuated");
    } else if(name=="small") {
        // Once ramped, small requests are unchanged (apart from integer quantization).
        m._startup_deadzone_pwm.value=0;m._thrust_rpyt_out[0]=0.08f;settle(m);
        require(m.written[0]==m.calc_thrust_to_pwm(0.08f),"Small feasible request changed after settling");
        require(!m._rov_output_limited&&!m.limit.roll,"Feasible steady request marked saturated");
    } else if(name=="tests") {
        m.output_test_num(0,2000);for(int p:m.written) { require(p==1500,"Test start jumped"); }
        for(int n=0;n<100;++n) {
            clock_us+=40000;m.output_test_num(n%8,n%2?2000:1000);bounds(m);
        }
        m.output_test_num(0,3000);for(int p:m.written) { require(p==1500,"Out-of-endpoint motor test failed open"); }
    } else if(name=="sequence") {
        m._output_test_seq(2,2000);
        for(int n=0;n<100;++n) { clock_us+=40000;m._output_test_seq(2,2000);bounds(m); }
        require(m.written[1]>1500&&m.written[1]<=1600,"Ordered motor test bypassed cap or mapping");
        m.is_armed=false;m._output_test_seq(2,2000);for(int p:m.written) { require(p==1500,"Disarmed motor test actuated"); }
    } else if(name=="same_tick") {
        // Eight simultaneous test requests share one real clock, not eight dt steps.
        for(int i=0;i<8;++i) { m.output_test_num(i,2000); }
        for(int p:m.written) { require(p==1500,"Same-tick test batch multiplied slew"); }
        clock_us+=20000;
        for(int i=0;i<8;++i) { m.output_test_num(i,2000); }
        for(int p:m.written) { require(std::abs(p-1500)<=5,"Batch multiplied available ramp time"); }
    } else if(name=="replay") {
        const int16_t recorded[5][8]={
            {1500,1500,1500,1500,1554,1446,1446,1554},
            {1444,1444,1444,1444,1556,1444,1443,1557},
            {1249,1249,1249,1249,1385,1556,1392,1667},
            {1091,1057,1170,1136,1365,1374,1446,1724},
            {1073,1073,1073,1073,1413,1282,1591,1715}};
        for(const auto &row:recorded) {
            for(int n=0;n<50;++n) {
                int16_t output[8];for(int i=0;i<8;++i) { output[i]=row[i]; }
                m.rov_guard_outputs(output,0.02f,true);
                for(int i=0;i<8;++i) { m.written[i]=output[i]; }bounds(m);
            }
        }
        require(m._rov_output_limited,"Recorded runaway request was not limited");
    } else if(name=="random") {
        uint32_t seed=1234567;
        for(int n=0;n<50000;++n) {
            for(float &v:m._thrust_rpyt_out) { seed=1664525U*seed+1013904223U;v=(int(seed%20001)-10000)/10000.0f; }
            m._rov_output_cap.value=(n%99+1);m._rov_output_sum.value=(n%80+1)/10.0f;
            m.output_to_motors();bounds(m);
        }
    } else { require(false,"Unknown scenario"); }
}
int main(int argc,char **argv) {
    require(argc==2,"Provide a scenario");run_case(argv[1]);std::cout<<argv[1]<<" passed\n";
}
'''


class OutputEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('g++'):
            raise AssertionError('C++ compiler required; protection tests may not silently skip')
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        cls.executable = directory / 'envelope.exe'
        previous, updated = sources()
        for name in ('ROV_OutputEnvelope.h', 'ROV_DeadzonePWM.h'):
            (directory/name).write_text(updated['libraries/AP_Motors/'+name], encoding='utf8')
        motor = updated['libraries/AP_Motors/AP_Motors6DOF.cpp']
        functions = '\n'.join(extract(motor, name) for name in
                              ('calc_thrust_to_pwm','output_min','output_to_motors',
                               'rov_guard_outputs','output_test_num','_output_test_seq'))
        baseline = extract(previous['libraries/AP_Motors/AP_Motors6DOF.cpp'], 'output_to_motors')
        baseline = baseline.replace('AP_Motors6DOF::', 'BaselineMotors::')
        source = directory / 'envelope.cpp'
        source.write_text(STUBS+functions+baseline+SCENARIOS, encoding='utf8')
        result = subprocess.run(['g++','-std=c++11','-Wall','-Wextra','-Werror','-pedantic',
                                 str(source),'-o',str(cls.executable)],capture_output=True,text=True,timeout=30)
        if result.returncode:
            raise AssertionError(result.stdout+result.stderr)

    def run_case(self, name):
        result = subprocess.run([str(self.executable),name],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('passed',result.stdout)

    def test_final_pwm_cap_and_anti_windup(self): self.run_case('caps')
    def test_previous_output_path_reproduces_missing_envelope(self): self.run_case('baseline')
    def test_aggregate_budget(self): self.run_case('sum')
    def test_rise_and_immediate_neutral(self): self.run_case('slew')
    def test_reversal_crosses_neutral(self): self.run_case('reverse')
    def test_disarm_idle_and_rearm(self): self.run_case('inactive')
    def test_invalid_config_and_timing_fail_closed(self): self.run_case('invalid')
    def test_zero_cap_inhibits_propulsion(self): self.run_case('zero')
    def test_asymmetric_pwm_endpoints(self): self.run_case('asymmetric')
    def test_disabled_motor_remains_neutral(self): self.run_case('disabled')
    def test_feasible_small_output_preserved(self): self.run_case('small')
    def test_raw_motor_tests_are_guarded(self): self.run_case('tests')
    def test_sequence_tests_are_guarded(self): self.run_case('sequence')
    def test_test_batch_cannot_multiply_slew_time(self): self.run_case('same_tick')
    def test_recorded_output_trace_bounded(self): self.run_case('replay')
    def test_fifty_thousand_random_cases(self): self.run_case('random')


if __name__ == '__main__':
    unittest.main()
