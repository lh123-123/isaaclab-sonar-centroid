// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
#include <cstddef>

namespace ROV_AttitudeLimit {
struct Result { float scale; bool valid; };

// Uniformly scale the combined roll/pitch contribution so its direction is
// retained while no enabled vertical thruster receives more than `limit`.
inline Result cap(float *rotation, const bool *enabled, std::size_t count,
                  float limit) {
    if (!std::isfinite(limit) || limit < 0.0f || limit > 1.0f) {
        return {0.0f, false};
    }
    float maximum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (!enabled[i]) { continue; }
        if (!std::isfinite(rotation[i])) { return {0.0f, false}; }
        maximum = ::fmaxf(maximum, ::fabsf(rotation[i]));
    }
    const float scale = maximum > limit && maximum > 0.0f ? limit / maximum : 1.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (enabled[i]) { rotation[i] *= scale; }
    }
    return {scale, true};
}
} // namespace ROV_AttitudeLimit
