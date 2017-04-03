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
            const_iview vec_ci,
            const_view vec_t_before,
            const_view vec_t_after,
            const_view vals,
            const std::vector<size_type>& indxs,
            const std::vector<value_type>& thresh):
        cv_to_cell_(vec_ci),
        t_before_(vec_t_before),
        t_after_(vec_t_after),
        values_(vals),
        cv_index_(memory::make_const_view(indxs)),
        thresholds_(memory::make_const_view(thresh)),
        v_prev_(vals)
    {
        is_crossed_ = iarray(size());
        reset();
    }

    /// Remove all stored crossings that were detected in previous calls
    /// to the test() member function.
    void clear_crossings() {
        crossings_.clear();
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset() {
        clear_crossings();
        for (auto i=0u; i<size(); ++i) {
            is_crossed_[i] = values_[index_[i]]>=thresholds_[i];
        }
    }

    const std::vector<threshold_crossing>& crossings() const {
        return crossings_;
    }

    /// Tests each target for changed threshold state
    /// Crossing events are recorded for each threshold that
    /// is crossed since the last call to test
    void test() {
        for (auto i=0u; i<size(); ++i) {
            auto cv     = cv_index_[i];
            auto cell   = cv_to_cell_[cv];
            auto v_prev = v_prev_[i];
            auto v      = values_[cv];
            auto thresh = thresholds_[i];
            if (!is_crossed_[i]) {
                if (v>=thresh) {
                    // The threshold has been passed, so estimate the time using
                    // linear interpolation.
                    auto pos = (thresh - v_prev)/(v - v_prev);
                    auto crossing_time = math::lerp(t_before_[cell], t_after_[cell], pos);
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
    const_iview cv_to_cell_;
    const_view t_before_;
    const_view t_after_;
    const_view values_;
    iarray cv_index_;

    array thresholds_;
    array v_prev_;
    crossing_list crossings_;
    iarray is_crossed_;
};

} // namespace multicore
} // namespace mc
} // namespace nest
