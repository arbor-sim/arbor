#pragma once

#include <cell.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

// spike detector for a lowered cell
template <typename Cell>
class spike_detector
{
public:
    using cell_type = Cell;

    spike_detector( const cell_type& cell, segment_location loc, double thresh, float t_init) :
        location_(loc),
        threshold_(thresh),
        previous_t_(t_init)
    {
        previous_v_ = cell.voltage(location_);
        is_spiking_ = previous_v_ >= thresh ? true : false;
    }

    util::optional<float> test(const cell_type& cell, float t) {
        util::optional<float> result = util::nothing;
        auto v = cell.voltage(location_);

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

    segment_location location() const { return location_; }

    float t() const { return previous_t_; }

    float v() const { return previous_v_; }

private:

    // parameters/data
    segment_location location_;
    double threshold_;

    // state
    float previous_t_;
    float previous_v_;
    bool is_spiking_;
};


} // namespace mc
} // namespace nest

