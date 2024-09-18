#pragma once

#include <arbor/assert.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>

#include "backends/common_types.hpp"
#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"
#include "multicore_common.hpp"

namespace arb {
namespace multicore {

class threshold_watcher {
public:
    threshold_watcher() = default;
    threshold_watcher(const execution_context&) {}

    threshold_watcher(const arb_size_type num_cv,
                      const arb_index_type* src_to_spike,
                      const fvm_detector_info& info):
        threshold_watcher{num_cv, src_to_spike, info.cv, info.threshold, info.ctx}
    {}

    threshold_watcher(const arb_size_type num_cv,
                      const arb_index_type* src_to_spike,
                      const std::vector<arb_index_type>& cv_index,
                      const std::vector<arb_value_type>& thresholds,
                      const execution_context&):
        src_to_spike_(src_to_spike),
        n_detectors_(cv_index.size()),
        cv_index_(cv_index),
        is_crossed_(n_detectors_),
        thresholds_(thresholds),
        v_prev_(num_cv)
    {
        arb_assert(n_detectors_==thresholds.size());
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
        for (arb_size_type i = 0; i<n_detectors_; ++i) {
            is_crossed_[i] = values_[cv_index_[i]]>=thresholds_[i];
        }
    }

    const std::vector<threshold_crossing>& crossings() const {
        return crossings_;
    }

    /// Tests each target for changed threshold state
    /// Crossing events are recorded for each threshold that
    /// is crossed since the last call to test
    void test(array& time_since_spike, const arb_value_type& t_before, const arb_value_type& t_after) {
        if (cv_index_.empty() || n_detectors_ == 0) return;
        arb_assert(values_!=nullptr);

        // Reset all spike times to -1.0 indicating no spike has been recorded on the detector
        for (arb_size_type i = 0; i<n_detectors_; ++i) {
            auto cv     = cv_index_[i];
            auto v_prev = v_prev_[cv];
            auto v      = values_[cv];
            auto thresh = thresholds_[i];
            arb_index_type spike_idx = 0;

            if (!time_since_spike.empty()) {
                spike_idx = src_to_spike_[i];
                time_since_spike[spike_idx] = -1.0;
            }

            if (!is_crossed_[i]) {
                if (v>=thresh) {
                    // The threshold has been passed, so estimate the time using
                    // linear interpolation.
                    auto pos = (thresh - v_prev)/(v - v_prev);
                    auto crossing_time = math::lerp(t_before, t_after, pos);
                    crossings_.push_back({i, crossing_time});

                    if (!time_since_spike.empty()) {
                        time_since_spike[spike_idx] = t_after - crossing_time;
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
        return n_detectors_;
    }

private:
    // Non-owning pointers
    const arb_value_type* values_ = nullptr;
    const arb_index_type* src_to_spike_ = nullptr;

    /// Threshold watcher state.
    arb_size_type n_detectors_ = 0;
    std::vector<arb_index_type> cv_index_;
    std::vector<arb_size_type> is_crossed_;
    std::vector<arb_value_type> thresholds_;
    std::vector<arb_value_type> v_prev_;
    std::vector<threshold_crossing> crossings_;
};

} // namespace multicore
} // namespace arb
