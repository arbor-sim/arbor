#pragma once

#include <memory/memory.hpp>

namespace nest {
namespace mc {
namespace multicore {

template <typename T, typename I>
class threshold_watcher {
public:
    using value_type = T;
    using size_type = I;

    using array = memory::host_vector<value_type>;
    using const_view = typename array::const_view_type;
    using iarray = memory::host_vector<size_type>;

    /// stores a single crossing event
    struct threshold_crossing {
        size_type index;    // index of variable
        value_type time;    // time of crossing
        friend bool operator== (
            const threshold_crossing& lhs, const threshold_crossing& rhs)
        {
            return lhs.index==rhs.index && lhs.time==rhs.time;
        }
    };

    threshold_watcher() = default;

    threshold_watcher(
            const_view vals,
            const std::vector<size_type>& indxs,
            const std::vector<value_type>& thresh,
            value_type t=0):
        values_(vals),
        index_(memory::make_const_view(indxs)),
        thresholds_(memory::make_const_view(thresh)),
        v_prev_(vals)
    {
        is_crossed_ = iarray(size());
        reset(t);
    }

    /// Remove all stored crossings that were detected in previous calls
    /// to the test() member function.
    void clear_crossings() {
        crossings_.clear();
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset(value_type t=0) {
        clear_crossings();
        for (auto i=0u; i<size(); ++i) {
            is_crossed_[i] = values_[index_[i]]>=thresholds_[i];
        }
        t_prev_ = t;
    }

    const std::vector<threshold_crossing>& crossings() const {
        return crossings_;
    }

    /// The time at which the last test was performed
    value_type last_test_time() const {
        return t_prev_;
    }

    /// Tests each target for changed threshold state
    /// Crossing events are recorded for each threshold that
    /// is crossed since the last call to test
    void test(value_type t) {
        for (auto i=0u; i<size(); ++i) {
            auto v_prev = v_prev_[i];
            auto v      = values_[index_[i]];
            auto thresh = thresholds_[i];
            if (!is_crossed_[i]) {
                if (v>=thresh) {
                    // the threshold has been passed, so estimate the time using
                    // linear interpolation
                    auto pos = (thresh - v_prev)/(v - v_prev);
                    auto crossing_time = t_prev_ + pos*(t - t_prev_);
                    crossings_.push_back({i, crossing_time});

                    is_crossed_[i] = true;
                }
            }
            else {
                if (v<thresh) {
                    is_crossed_[i] = false;
                }
            }

            v_prev_[i] = v;
        }
        t_prev_ = t;
    }

    bool is_crossed(size_type i) const {
        return is_crossed_[i];
    }

    /// the number of threashold values that are being monitored
    std::size_t size() const {
        return index_.size();
    }

    /// Data type used to store the crossings.
    /// Provided to make type-generic calling code.
    using crossing_list =  std::vector<threshold_crossing>;

private:
    const_view values_;
    iarray index_;

    array thresholds_;
    value_type t_prev_;
    array v_prev_;
    crossing_list crossings_;
    iarray is_crossed_;
};

} // namespace multicore
} // namespace mc
} // namespace nest
