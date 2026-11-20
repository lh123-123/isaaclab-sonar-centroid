// Offline checks of the exact header included by the firmware patch.
#include "ROV_DeadzonePWM.h"
#include <cassert>
#include <limits>

using ROV_DeadzonePWM::calculate;

int main()
{
    assert(calculate(0, 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(0.01f, 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(-0.01f, 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(0.01001f, 1000, 2000, 50, 0.01f) == 1550);
    assert(calculate(-0.01001f, 1000, 2000, 50, 0.01f) == 1450);
    assert(calculate(1, 1000, 2000, 50, 0.01f) == 2000);
    assert(calculate(-1, 1000, 2000, 50, 0.01f) == 1000);
    assert(calculate(0.5f, 1000, 2000, 50, 0.01f) < 2000);
    assert(calculate(0.95f, 1000, 2000, 50, 0.01f) < 2000);
    assert(calculate(1, 1100, 1900, 50, 0.01f) == 1900);
    assert(calculate(-1, 1100, 1900, 50, 0.01f) == 1100);
    assert(calculate(2, 1000, 2000, 50, 0.01f) == 2000);
    assert(calculate(-2, 1000, 2000, 50, 0.01f) == 1000);
    assert(calculate(std::numeric_limits<float>::quiet_NaN(), 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(std::numeric_limits<float>::infinity(), 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(0.5f, 1500, 2000, 50, 0.01f) == 1500);
    assert(calculate(0.5f, 1000, 1500, 50, 0.01f) == 1500);
    assert(calculate(0.5f, 1000, 2000, 50, -0.01f) == 1500);
    assert(calculate(0.5f, 1000, 2000, 50, std::numeric_limits<float>::quiet_NaN()) == 1500);
    int16_t previous = 1000;
    for (int i = -10000; i <= 10000; ++i) {
        const float thrust = i / 10000.0f;
        const int16_t output = calculate(thrust, 1000, 2000, 50, 0.01f);
        assert(output >= previous && output >= 1000 && output <= 2000);
        assert(output == 3000 - calculate(-thrust, 1000, 2000, 50, 0.01f));
        assert(calculate(thrust, 1000, 2000, 0, 0.01f) == static_cast<int16_t>(1500 + thrust * 500));
        assert(calculate(thrust, 1000, 2000, -50, 0.01f) == calculate(thrust, 1000, 2000, 0, 0.01f));
        assert(calculate(thrust, 1000, 2000, 500, 0.01f) == calculate(thrust, 1000, 2000, 100, 0.01f));
        previous = output;
    }
    // Requests from motion and attitude loops add/cancel BEFORE this function.
    assert(calculate(0.2f - 0.2f, 1000, 2000, 50, 0.01f) == 1500);
    assert(calculate(0.01f + 0.03f, 1000, 2000, 50, 0.01f) >= 1550);
    for (int repeat = 0; repeat < 1000; ++repeat) {
        assert(calculate(0, 1000, 2000, 50, 0.01f) == 1500);
    }
    return 0;
}
