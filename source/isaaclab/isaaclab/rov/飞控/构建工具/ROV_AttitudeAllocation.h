// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
#include <cstddef>

namespace ROV_AttitudeAllocation {
struct Result { float rotation_scale; float translation_scale; bool valid; };
// Allocate each disjoint Vectored-6DOF plane independently. Preserve the
// requested rotational wrench first, then uniformly scale its translation.
inline Result allocate(float *rotation, const float *translation,
                       const bool *enabled, std::size_t count) {
    float maximum = 1.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (!enabled[i]) { continue; }
        if (!std::isfinite(rotation[i]) || !std::isfinite(translation[i])) {
            return {0, 0, false};
        }
        maximum = ::fmaxf(maximum, ::fabsf(rotation[i]));
    }
    const float rotation_scale = 1.0f / maximum;
    float translation_scale = 1.0f;
    for (std::size_t i = 0; i < count; ++i) {
        if (!enabled[i]) { continue; }
        rotation[i] *= rotation_scale;
        const float linear = translation[i];
        if (linear > 0) {
            translation_scale = ::fminf(translation_scale, (1.0f - rotation[i]) / linear);
        } else if (linear < 0) {
            translation_scale = ::fminf(translation_scale, (-1.0f - rotation[i]) / linear);
        }
    }
    translation_scale = ::fmaxf(0.0f, ::fminf(1.0f, translation_scale));
    for (std::size_t i = 0; i < count; ++i) {
        if (enabled[i]) { rotation[i] += translation[i] * translation_scale; }
    }
    return {rotation_scale, translation_scale, true};
}
} // namespace ROV_AttitudeAllocation
