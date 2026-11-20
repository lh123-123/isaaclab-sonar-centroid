// SPDX-License-Identifier: GPL-3.0-or-later
// Final PWM excursion limits, NOT a current model or measured thrust limit.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ROV_OutputEnvelope {
struct Config {
    float per_motor = 0.2f;
    float sum = 1.0f;
    float rise_per_second = 0.5f;
};
struct Result { bool valid; bool limited; float scale; };

template <std::size_t N> class Envelope {
public:
    void reset() { for (float &v : previous) { v = 0; } }

    Result apply(const int16_t *requested, const bool *enabled, int16_t *output,
                 int16_t minimum, int16_t maximum, float dt, bool active, const Config &c) {
        constexpr int16_t neutral = 1500;
        for (std::size_t i=0; i<N; ++i) { output[i] = neutral; }
        if (!active) { reset(); return {true, false, 0}; }
        if (minimum < 900 || minimum >= neutral || maximum <= neutral || maximum > 2100 ||
            !std::isfinite(c.per_motor) || c.per_motor < 0 || c.per_motor > 1 ||
            !std::isfinite(c.sum) || c.sum < 0 || c.sum > N ||
            !std::isfinite(c.rise_per_second) || c.rise_per_second <= 0 || c.rise_per_second > 5 ||
            !std::isfinite(dt) || dt < 0 || dt > 0.1f) {
            reset(); return {false, true, 0};
        }
        float target[N] = {};
        float peak = 0, total = 0;
        for (std::size_t i=0; i<N; ++i) {
            if (!enabled[i]) { continue; }
            if (requested[i] < minimum || requested[i] > maximum) {
                reset(); return {false, true, 0};
            }
            const int16_t delta = requested[i] - neutral;
            const float span = delta >= 0 ? maximum-neutral : neutral-minimum;
            target[i] = delta / span;
            peak = ::fmaxf(peak, ::fabsf(target[i]));
            total += ::fabsf(target[i]);
        }
        float scale = 1;
        if (peak > c.per_motor) { scale = ::fminf(scale, c.per_motor / peak); }
        if (total > c.sum) { scale = ::fminf(scale, c.sum / total); }
        bool limited = scale < 0.999999f;
        const float step = c.rise_per_second * dt;
        // A common rise factor preserves the requested vector proportions.
        // Neutral and reversal are conservative exceptions: reductions are immediate.
        float rise_scale = 1;
        for (std::size_t i=0; i<N; ++i) {
            target[i] *= scale;
            if (previous[i] * target[i] < 0) {
                target[i] = 0;
                limited = true;
            }
            const float magnitude = ::fabsf(target[i]);
            if (magnitude > ::fabsf(previous[i]) + step) {
                rise_scale = ::fminf(rise_scale, (::fabsf(previous[i]) + step) / magnitude);
            }
        }
        limited = limited || rise_scale < 0.999999f;
        for (std::size_t i=0; i<N; ++i) {
            previous[i] = enabled[i] ? target[i] * rise_scale : 0;
            const float span = previous[i] >= 0 ? maximum-neutral : neutral-minimum;
            // Truncate towards neutral: rounding must not exceed either budget.
            if (enabled[i] && scale == 1.0f && rise_scale == 1.0f &&
                (target[i] != 0 || requested[i] == neutral)) {
                // Exact pass-through once feasible and settled: avoid a second
                // float->PWM quantization changing the native small request.
                output[i] = requested[i];
            } else {
                output[i] = neutral + static_cast<int16_t>(previous[i] * span);
            }
            limited = limited || (enabled[i] && std::abs(int(output[i])-int(requested[i])) > 1);
        }
        return {true, limited, scale * rise_scale};
    }
private:
    float previous[N] = {};
};
} // namespace ROV_OutputEnvelope
