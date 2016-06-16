#pragma once

#include <cell.hpp>
#include <fvm_cell.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

// generic spike source
class spike_source {
    public:

    virtual util::optional<float> test(float t) = 0;
};

// spike detector for a lowered cell
template <typename Cell>
class spike_detector : public spike_source
{
    public:
    using cell_type = Cell;

    spike_detector(
        cell_type const* cell,
        segment_location loc,
        double thresh,
        float t_init
    )
        :   cell_(cell),
            location_(loc),
            threshold_(thresh),
            previous_t_(t_init)
    {
        previous_v_ = cell->voltage(location_);
        is_spiking_ = previous_v_ >= thresh ? true : false;
    }

    util::optional<float> test(float t) override {
        util::optional<float> result = util::nothing;
        auto v = cell_->voltage(location_);
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

    bool is_spiking() const {
        return is_spiking_;
    }

    segment_location location() const {
        return location_;
    }

    private:

    // parameters/data
    cell_type* cell_;
    segment_location location_;
    double threshold_;

    // state
    float previous_t_;
    float previous_v_;
    bool is_spiking_;
};

// spike generator according to a Poisson process
class poisson_generator : public spike_source
{
    public:

    poisson_generator(float r)
    :   dist_(0.0f, 1.0f),
        firing_rate_(r)
    {}

    util::optional<float> test(float t) {
        // generate a uniformly distrubuted random number x \in [0,1]
        // if  (x > r*dt)  we have a spike in the interval
        std::vector<float> spike_times;
        if(dist_(generator_) > firing_rate_*(t-previous_t_)) {
            return t;
        }
        return util::nothing;
    }

    private:

    std::mt19937 generator_; // for now default initialized
    std::uniform_real_distribution<float> dist_;

    // firing rate in spikes/ms
    float firing_rate_;
    float previous_t_;
};


} // namespace mc
} // namespace nest

