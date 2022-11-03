#pragma once

#include <arbor/assert.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>

#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"
#include "multicore_common.hpp"

namespace arb {
namespace multicore {

class threshold_watcher {
public:
    threshold_watcher() = default;

    threshold_watcher(const execution_context& ctx) {}

    threshold_watcher(
        const arb_index_type* cv_to_intdom,
        const arb_index_type* src_to_spike,
        const array* t_before,
        const array* t_after,
        const arb_size_type num_cv,
        const std::vector<arb_index_type>& cv_index,
        const std::vector<arb_value_type>& thresholds,
        const execution_context& context
    ):
        cv_to_intdom_(cv_to_intdom),
        src_to_spike_(src_to_spike),
        t_before_ptr_(t_before),
        t_after_ptr_(t_after),
        n_cv_(cv_index.size()),
        cv_index_(cv_index),
        is_crossed_(n_cv_),
        thresholds_(thresholds),
        v_prev_(num_cv)
    {
        arb_assert(n_cv_==thresholds.size());
        // reset() needs to be called before this is ready for use
    }

    /// Remove all stored crossings that were detected in previous calls
    /// to the test() member function.
    void clear_crossings() {
        crossings_.clear();
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset(const array& values) {
        values_ = values.data();
        std::copy(values.begin(), values.end(), v_prev_.begin());
        clear_crossings();
        for (arb_size_type i = 0; i<n_cv_; ++i) {
            is_crossed_[i] = values_[cv_index_[i]]>=thresholds_[i];
        }
    }

    const std::vector<threshold_crossing>& crossings() const {
        return crossings_;
    }

    /// Tests each target for changed threshold state
    /// Crossing events are recorded for each threshold that
    /// is crossed since the last call to test
    void test(array* time_since_spike) {
        // either number of cvs is 0 or values_ is not null
        arb_assert((n_cv_ == 0) || (bool)values_);

        // Reset all spike times to -1.0 indicating no spike has been recorded on the detector
        const arb_value_type* t_before = t_before_ptr_->data();
        const arb_value_type* t_after  = t_after_ptr_->data();
        for (arb_size_type i = 0; i<n_cv_; ++i) {
            auto cv     = cv_index_[i];
            auto intdom = cv_to_intdom_[cv];
            auto v_prev = v_prev_[cv];
            auto v      = values_[cv];
            auto thresh = thresholds_[i];
            arb_index_type spike_idx = 0;

            if (!time_since_spike->empty()) {
                spike_idx = src_to_spike_[i];
                (*time_since_spike)[spike_idx] = -1.0;
            }

            if (!is_crossed_[i]) {
                if (v>=thresh) {
                    // The threshold has been passed, so estimate the time using
                    // linear interpolation.
                    auto pos = (thresh - v_prev)/(v - v_prev);
                    auto crossing_time = math::lerp(t_before[intdom], t_after[intdom], pos);
                    crossings_.push_back({i, crossing_time});

                    if (!time_since_spike->empty()) {
                        (*time_since_spike)[spike_idx] = t_after[intdom] - crossing_time;
                    }

                    is_crossed_[i] = true;
                }
            }
            else {
                if (v<thresh) {
                    is_crossed_[i] = false;
                }
            }

            v_prev_[cv] = v;
        }
    }

    bool is_crossed(arb_size_type i) const {
        return is_crossed_[i];
    }

    /// The number of threshold values that are monitored.
    std::size_t size() const {
        return n_cv_;
    }

private:
    /// Non-owning pointers to cv-to-intdom map,
    /// the values for to test against thresholds,
    /// and pointers to the time arrays
    const arb_index_type* cv_to_intdom_ = nullptr;
    const arb_value_type* values_ = nullptr;
    const arb_index_type* src_to_spike_ = nullptr;
    const array* t_before_ptr_ = nullptr;
    const array* t_after_ptr_ = nullptr;

    /// Threshold watcher state.
    arb_size_type n_cv_ = 0;
    std::vector<arb_index_type> cv_index_;
    std::vector<arb_size_type> is_crossed_;
    std::vector<arb_value_type> thresholds_;
    std::vector<arb_value_type> v_prev_;
    std::vector<threshold_crossing> crossings_;
};

} // namespace multicore
} // namespace arb
