#include "Sub.h"
#include <iostream>
Sub sub;
int main() {
    ModeDepthOnly mode;
    mode.depth_valid=false;
    assert(!mode.init(false));
    assert(sub.neutral==0);
    mode.depth_valid=true;
    assert(mode.init(false));
    assert(sub.neutral==1);
    mode.run();
    assert(mode.depth_calls==0);
    assert(mode.pos.relaxed==1);
    mode.motors.active=true;
    mode.r.input=.1f; mode.p.input=.2f; mode.y.input=-.3f;
    mode.f.input=.4f; mode.l.input=-.5f;
    mode.run();
    assert(mode.depth_calls==1 && mode.pos.limits==1);
    assert(mode.motors.throttle==.65f);
    assert(mode.motors.r==.1f && mode.motors.p==.2f && mode.motors.y==-.3f);
    assert(mode.motors.f==.4f && mode.motors.l==-.5f);
    sub.fresh=false;
    mode.run();
    assert(mode.depth_calls==2); // pressure feedback remains active
    assert(mode.motors.r==0 && mode.motors.p==0 && mode.motors.y==0);
    assert(mode.motors.f==0 && mode.motors.l==0);
    mode.motors.active=false;
    mode.run();
    assert(mode.depth_calls==2);
    std::cout << "Depth-only sensor/armed/fresh-input scenarios passed\n";
}
