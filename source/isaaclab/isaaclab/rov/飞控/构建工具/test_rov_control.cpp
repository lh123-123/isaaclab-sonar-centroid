// Actual target/allocation kernels used by the firmware patch, not copies.
#include <cmath>
#include <cstddef>
// Match ArduPilot's single-precision policy even in the host regression build.
#define fmax(x,y) DO_NOT_USE_DOUBLE_MATHS()
#define fabs(x) DO_NOT_USE_DOUBLE_MATHS()
#define sqrt(x) DO_NOT_USE_DOUBLE_MATHS()
#include "ROV_PilotKernel.h"
#include "ROV_AttitudeAllocation.h"
#include "ROV_DeadzonePWM.h"
#include "ROV_DepthHold.h"
#undef fmax
#undef fabs
#undef sqrt
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

using ROV_PilotKernel::Targets;
using ROV_PilotKernel::Config;
using ROV_PilotKernel::wrap;
static bool near(float a, float b, float eps = 0.002f) { return std::fabs(a-b) <= eps; }
static constexpr float dt = 0.01f;
static auto step(Targets &t, float ri, float pi, float yi, float yaw = 0, bool fresh = true)
    -> ROV_PilotKernel::Output {
    return t.update(dt, 0, 0, yaw, ri, pi, yi, fresh, Config{});
}

static void roll_pitch_tolerance_keeps_fixed_reference() {
    using ROV_PilotKernel::tolerant_angle_target;
    assert(near(tolerant_angle_target(0, 0.4f, 0.5f), 0.4f));
    assert(near(tolerant_angle_target(0, -0.4f, 0.5f), -0.4f));
    assert(near(tolerant_angle_target(0, 0.6f, 0.5f), 0.5f));
    assert(near(tolerant_angle_target(0, -0.6f, 0.5f), -0.5f));
    assert(near(tolerant_angle_target(12, 11.8f, 0.5f), 11.8f));
    assert(near(tolerant_angle_target(12, 10, 0.5f), 11.5f));
    assert(near(tolerant_angle_target(12, 10, 0), 12));
    assert(near(tolerant_angle_target(12, 10, 6), 12));
    // The angle target is continuous at the boundary, avoiding a thrust step.
    assert(near(tolerant_angle_target(0, 0.5f, 0.5f),
                tolerant_angle_target(0, 0.50001f, 0.5f), 0.00002f));
}

static void release_returns_both_axes_to_level() {
    for (int r = -1; r <= 1; ++r) {
        for (int p = -1; p <= 1; ++p) {
            Targets t;
            t.enter(false, 0, 0, 123);
            for (int i=0; i<500; ++i) {
                const auto o = step(t, float(r), float(p), 0, 123);
                assert(o.valid && std::hypot(o.roll,o.pitch) <= 30.001f);
            }
            const auto before = t.output();
            const auto first = step(t, 0, 0, 0, 123);
            assert(std::fabs(first.roll-before.roll) <= 0.301f);
            assert(std::fabs(first.pitch-before.pitch) <= 0.301f);
            for (int i=0; i<500; ++i) { step(t, 0, 0, 0, 123); }
            assert(near(t.output().roll, 0) && near(t.output().pitch, 0));
            assert(near(t.output().yaw, 123));
        }
    }
}

static void entry_switch_and_outside_limit() {
    Targets t;
    t.enter(false, 45, -35, 179);
    assert(near(t.output().roll,45) && near(t.output().pitch,-35));
    auto first = step(t, 0, 0, 0, 179);
    assert(first.roll > 44.98f && first.pitch < -34.98f);
    // STABILIZE <-> ALT_HOLD preserves the old commanded target, not noisy AHRS.
    t.enter(true, -20, 20, -90);
    assert(near(t.output().roll, first.roll) && near(t.output().yaw,179));
    for (int i=0; i<600; ++i) { step(t,0,0,0,179); }
    assert(near(t.output().roll,0) && near(t.output().pitch,0));
    t.enter(false, -10, 15, -90);
    assert(near(t.output().roll,-10) && near(t.output().pitch,15));
    t.reset();
    assert(!t.output().valid);
    t.enter(true, 5, 8, 7);
    assert(near(t.output().pitch,8));
    // Moving the goal to the currently commanded angle must not reset a fast
    // trajectory to zero velocity in one tick.
    t.enter(false,0,0,0);
    for (int i=0; i<35; ++i) { step(t,0,1,0); }
    const auto moving=t.output();
    assert(moving.pitch_rate>25);
    const auto changed=step(t,0,moving.pitch/30.0f,0);
    assert(std::fabs(changed.pitch_rate-moving.pitch_rate)<=0.9001f);
}

static void yaw_release_wrap_reset_and_disturbance() {
    Targets t;
    t.enter(false, 0, 0, 179);
    float measured = 179;
    for (int i=0; i<250; ++i) {
        auto o = step(t,0,0,1,measured);
        assert(std::fabs(wrap(o.yaw-measured)) <= 12.001f);
        measured = wrap(measured + o.yaw_rate*dt*0.8f);
    }
    for (int i=0; i<80; ++i) {
        auto o = step(t,0,0,0,measured);
        measured = wrap(measured + o.yaw_rate*dt*0.8f);
    }
    const float held = t.output().yaw;
    assert(near(t.output().yaw_rate,0));
    // Neutral ticks must NOT capture water-current disturbance as a new heading.
    for (int i=0; i<200; ++i) { step(t,0,0,0,wrap(measured+20)); }
    assert(near(wrap(t.output().yaw-held),0));
    t.shift_yaw(30);
    assert(near(wrap(t.output().yaw-held),30));
    t.shift_yaw(std::numeric_limits<float>::quiet_NaN());
    assert(near(wrap(t.output().yaw-held),30));
}

static void yaw_release_captures_immediately_despite_inertia() {
    for (float direction : {-1.0f, 1.0f}) {
        Targets t;
        t.enter(false, 0, 0, 179);
        for (int i=0; i<80; ++i) { step(t,0,0,direction,179); }
        assert(std::fabs(t.output().yaw_rate)>40);
        auto released = step(t,0,0,0,-179);
        assert(near(released.yaw,-179) && near(released.yaw_rate,0));
        // The vehicle can overshoot, but the operator reference must not follow it.
        for (int i=0; i<100; ++i) { step(t,0,0,0,wrap(-179+direction*i*0.1f)); }
        assert(near(t.output().yaw,-179) && near(t.output().yaw_rate,0));
    }
}

static void stale_and_bad_data() {
    Targets t;
    for (int i=0; i<300; ++i) { step(t,0,1,1); }
    auto o = step(t,1,1,1,70,false);
    assert(o.valid && near(o.yaw_rate,0) && near(o.yaw,70));
    for (int i=0; i<500; ++i) { step(t,1,1,1,90,false); }
    assert(near(t.output().pitch,0) && near(t.output().roll,0));
    assert(near(t.output().yaw,70));
    for (float bad_dt : {0.0f, -1.0f, 0.2f, std::numeric_limits<float>::infinity()}) {
        o = t.update(bad_dt,0,0,0,0,0,0,true,Config{});
        assert(!o.valid);
        o = step(t,0,0,0,42);
        assert(o.valid && near(o.yaw,42));
    }
    Config c;
    c.yaw_accel = std::numeric_limits<float>::quiet_NaN();
    assert(!t.update(dt,0,0,0,0,0,0,true,c).valid);
    assert(!t.update(dt,0,std::numeric_limits<float>::infinity(),0,0,0,0,true,Config{}).valid);
}

static void random_target_envelope_and_neutral_final() {
    Targets t;
    std::mt19937 rng(104);
    std::uniform_real_distribution<float> input(-1,1);
    for (int i=0; i<40000; ++i) {
        const auto previous = t.output();
        auto o = step(t,input(rng),input(rng),input(rng),input(rng)*180);
        assert(o.valid && std::hypot(o.roll,o.pitch) <= 30.002f);
        assert(std::fabs(o.roll_rate)<=30 && std::fabs(o.pitch_rate)<=30);
        assert(o.yaw>=-180 && o.yaw<180 && std::fabs(o.yaw_rate)<=45);
        if (previous.valid) {
            assert(std::fabs(o.pitch-previous.pitch)<=0.302f);
            assert(std::fabs(o.roll-previous.roll)<=0.302f);
        }
    }
    for (int i=0; i<600; ++i) { step(t,0,0,0,25); }
    assert(near(t.output().roll,0) && near(t.output().pitch,0));
}

static void allocation_preserves_attitude_and_linear_ratios() {
    bool enabled[4] = {true,true,true,true};
    float rotation[4] = {0.5f,-0.5f,0.5f,-0.5f};
    float translation[4] = {1,1,1,1};
    auto result = ROV_AttitudeAllocation::allocate(rotation,translation,enabled,4);
    assert(result.valid && near(result.rotation_scale,1) && near(result.translation_scale,0.5f));
    assert(near(rotation[0],1) && near(rotation[1],0));
    // Roll moment is kept, rather than scaling roll and lift together.
    assert(near((rotation[0]-rotation[1]+rotation[2]-rotation[3])/4,0.5f));
    float saturated[4] = {2,-2,2,-2};
    result = ROV_AttitudeAllocation::allocate(saturated,translation,enabled,4);
    assert(near(result.rotation_scale,0.5f) && near(result.translation_scale,0));
    float diagonal[4] = {0,0,0,0};
    float linear[4] = {-2,0,0,2};
    result = ROV_AttitudeAllocation::allocate(diagonal,linear,enabled,4);
    assert(near(result.translation_scale,0.5f) && near(diagonal[0],-1) && near(diagonal[3],1));
    for (float motor : diagonal) {
        assert(ROV_DeadzonePWM::calculate(motor,1000,2000,50,0.01f)>=1000);
        assert(ROV_DeadzonePWM::calculate(-motor,1000,2000,50,0.01f)<=2000);
    }
}

static void allocation_random_bounds_and_invalid() {
    std::mt19937 rng(14);
    std::uniform_real_distribution<float> command(-3,3);
    bool enabled[4] = {true,true,true,false};
    for (int n=0; n<40000; ++n) {
        float r[4], tr[4], old[4];
        for (int i=0;i<4;++i) { r[i]=old[i]=command(rng); tr[i]=command(rng); }
        const auto a=ROV_AttitudeAllocation::allocate(r,tr,enabled,4);
        assert(a.valid && a.rotation_scale>0 && a.rotation_scale<=1);
        assert(a.translation_scale>=0 && a.translation_scale<=1);
        for (int i=0;i<3;++i) {
            assert(std::fabs(r[i])<=1.000002f);
            assert(near(r[i],old[i]*a.rotation_scale+tr[i]*a.translation_scale));
        }
        assert(r[3]==old[3]);
    }
    float r[4]={0,0,0,0}, tr[4]={0,0,std::numeric_limits<float>::infinity(),0};
    assert(!ROV_AttitudeAllocation::allocate(r,tr,enabled,4).valid);
}

static void depth_latch_finite_reset_and_initialization() {
    ROV_DepthHold::Target t;
    auto first = t.update(-200,0,100,1);
    assert(first.valid && first.hold && first.captured && near(first.altitude_cm,-200));
    auto again = t.update(-210,0,100,1);
    assert(!again.captured && near(again.altitude_cm,-200));
    auto shifted = t.update(-310,0,101,1);
    assert(near(shifted.altitude_cm,-300));
    t.limit_target(-290,-5);
    assert(near(t.output().altitude_cm,-290));
    for (int axis=0; axis<3; ++axis) {
        float measured=-290, rate=0, delta=0;
        const float bad=std::numeric_limits<float>::quiet_NaN();
        if (axis==0) { measured=bad; } else if (axis==1) { rate=bad; } else { delta=bad; }
        assert(!t.update(measured,rate,101,delta).valid);
        assert(t.update(-250,0,101,0).captured);
    }
    t.reset();
    assert(!t.output().valid);
    auto moving=t.update(-200,30,102,0);
    assert(moving.valid && !moving.hold);
    t.limit_target(-100,-5); // A boundary cannot turn pilot movement into hold.
    auto released=t.update(-185,0,103,-1);
    assert(released.hold && released.captured && near(released.altitude_cm,-185));
}

int main() {
    roll_pitch_tolerance_keeps_fixed_reference();
    release_returns_both_axes_to_level();
    entry_switch_and_outside_limit();
    yaw_release_wrap_reset_and_disturbance();
    yaw_release_captures_immediately_despite_inertia();
    stale_and_bad_data();
    random_target_envelope_and_neutral_final();
    allocation_preserves_attitude_and_linear_ratios();
    allocation_random_bounds_and_invalid();
    depth_latch_finite_reset_and_initialization();
    std::cout << "10 ROV control scenario suites passed (80000 randomized cases).\n";
}
