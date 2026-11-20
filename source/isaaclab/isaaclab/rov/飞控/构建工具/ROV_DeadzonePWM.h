// SPDX-License-Identifier: GPL-3.0-or-later
// Stateless post-mix mapping; used only by normal ArduSub motor outputs.
#pragma once

#include <cmath>
#include <cstdint>

namespace ROV_DeadzonePWM {

inline int16_t calculate(float thrust, int16_t minimum, int16_t maximum,
                         int16_t offset_us, float input_deadband)
{
    constexpr int16_t neutral = 1500;
    if (!std::isfinite(thrust) || minimum >= neutral || maximum <= neutral) {
        return neutral;
    }
    thrust = thrust < -1.0f ? -1.0f : thrust > 1.0f ? 1.0f : thrust;
    const int16_t span = thrust > 0.0f ? maximum - neutral : neutral - minimum;
    // Disabled means the stock conversion, without a new input deadband.
    if (offset_us <= 0) {
        return static_cast<int16_t>(neutral + thrust * span);
    }
    if (!std::isfinite(input_deadband) || input_deadband < 0.0f || input_deadband > 0.1f) {
        return neutral;
    }
    const float magnitude = ::fabsf(thrust);
    if (magnitude <= input_deadband) {
        return neutral;
    }
    // Guard malformed parameter values. Never extend configured endpoints.
    int16_t offset = offset_us > 100 ? 100 : offset_us;
    offset = offset > span ? span : offset;
    const float progress = (magnitude - input_deadband) / (1.0f - input_deadband);
    // Reserve the offset inside the available span: no early saturation.
    int16_t excursion = static_cast<int16_t>(::lroundf(offset + (span - offset) * progress));
    excursion = excursion > span ? span : excursion;
    return thrust > 0.0f ? neutral + excursion : neutral - excursion;
}

} // namespace ROV_DeadzonePWM
