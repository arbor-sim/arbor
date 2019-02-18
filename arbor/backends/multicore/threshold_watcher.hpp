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
        const fvm_index_type* cv_to_intdom,
        const fvm_value_type* t_before,
        const fvm_value_type* t_after,
        const fvm_value_type* values,
        const std::vector<fvm_index_type>& cv_index,
        const std::vector<fvm_value_type>& thresholds,
        const execution_context& context
    ):
        cv_to_intdom_(cv_to_intdom),
        t_before_(t_before),
        t_after_(t_after),
        values_(values),
        n_cv_(cv_index.size()),
        cv_index_(cv_index),
        is_crossed_(n_cv_),
        thresholds_(thresholds),
        v_prev_(values_, values_+n_cv_)
    {
        arb_assert(n_cv_==thresholds.size());
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
        for (fvm_size_type i = 0; i<n_cv_; ++i) {
            is_crossed_[i] = values_[cv_index_[i]]>=thresholds_[i];
        }
    }

    const std::vector<threshold_crossing>& crossings() const {
        return crossings_;
    }

    /// Tests each target for changed threshold state
    /// Crossing events are recorded for each threshold that
    /// is crossed since the last call to test
    void test() {
        for (fvm_size_type i = 0; i<n_cv_; ++i) {
            auto cv     = cv_index_[i];
            auto cell   = cv_to_intdom_[cv];
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

    bool is_crossed(fvm_size_type i) const {
        return is_crossed_[i];
    }

    /// The number of threshold values that are monitored.
    std::size_t size() const {
        return n_cv_;
    }

private:
    /// Non-owning pointers to cv-to-cell map, per-cell time data,
    /// and the values for to test against thresholds.
    const fvm_index_type* cv_to_intdom_ = nullptr;
    const fvm_value_type* t_before_ = nullptr;
    const fvm_value_type* t_after_ = nullptr;
    const fvm_value_type* values_ = nullptr;

    /// Threshold watcher state.
    fvm_size_type n_cv_ = 0;
    std::vector<fvm_index_type> cv_index_;
    std::vector<fvm_size_type> is_crossed_;
    std::vector<fvm_value_type> thresholds_;
    std::vector<fvm_value_type> v_prev_;
    std::vector<threshold_crossing> crossings_;
};

} // namespace multicore
} // namespace arb
