#pragma once

#include <cell.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

// spike detector for a lowered cell
template <typename Cell>
class spike_detector {
public:
    using cell_type = Cell;

    spike_detector(
        const cell_type& cell,
        typename Cell::detector_handle h,
        double thresh,
        float t_init
    ):
        handle_(h),
        threshold_(thresh)
    {
        reset(cell, t_init);
    }

    util::optional<float> test(const cell_type& cell, float t) {
        util::optional<float> result = util::nothing;
        auto v = cell.detector_voltage(handle_);

        // these if statements could be simplified, but I keep them like
        // this to clearly reflect the finite state machine
        if (!is_spiking_) {
            if (v>=threshold_) {
                // the threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (threshold_ - previous_v_)/(v - previous_v_);
                result = previous_t_ + pos*(t - previous_t_);

                is_spiking_ = true;
            }
        }
        else {
            if (v<threshold_) {
                is_spiking_ = false;
            }
        }

        previous_v_ = v;
        previous_t_ = t;

        return result;
    }

    bool is_spiking() const { return is_spiking_; }

    float t() const { return previous_t_; }

    float v() const { return previous_v_; }

    void reset(const cell_type& cell, float t_init) {
        previous_t_ = t_init;
        previous_v_ = cell.detector_voltage(handle_);
        is_spiking_ = previous_v_ >= threshold_;
    }

private:
    // parameters/data
    typename cell_type::detector_handle handle_;
    double threshold_;

    // state
    float previous_t_;
    float previous_v_;
    bool is_spiking_;
};


} // namespace mc
} // namespace nest

