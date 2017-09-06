#pragma once

#include <common_types.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>

#include "managed_ptr.hpp"
#include "stack.hpp"
#include "backends/fvm_types.hpp"
#include "kernels/test_thresholds.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// threshold crossing logic
/// used as part of spike detection back end
class threshold_watcher {
public:
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    using array = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;
    using const_view = typename array::const_view_type;
    using const_iview = typename iarray::const_view_type;

    using stack_type = stack<threshold_crossing>;

    threshold_watcher() = default;

    threshold_watcher(threshold_watcher&& other) = default;
    threshold_watcher& operator=(threshold_watcher&& other) = default;

    threshold_watcher(
            const_iview vec_ci,
            const_view vec_t_before,
            const_view vec_t_after,
            const_view values,
            const std::vector<size_type>& index,
            const std::vector<value_type>& thresh,
            value_type t=0):
        cv_to_cell_(vec_ci),
        t_before_(vec_t_before),
        t_after_(vec_t_after),
        values_(values),
        cv_index_(memory::make_const_view(index)),
        thresholds_(memory::make_const_view(thresh)),
        prev_values_(values),
        is_crossed_(size()),
        stack_(10*size())
    {
        reset();
    }

    /// Remove all stored crossings that were detected in previous calls
    /// to test()
    void clear_crossings() {
        stack_.clear();
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset() {
        clear_crossings();

        // Make host-side copies of the information needed to calculate
        // the initial crossed state
        auto values = memory::on_host(values_);
        auto thresholds = memory::on_host(thresholds_);
        auto cv_index = memory::on_host(cv_index_);

        // calculate the initial crossed state in host memory
        std::vector<size_type> crossed(size());
        for (auto i: util::make_span(0u, size())) {
            crossed[i] = values[cv_index[i]] < thresholds[i] ? 0 : 1;
        }

        // copy the initial crossed state to device memory
        memory::copy(crossed, is_crossed_);
    }

    bool is_crossed(size_type i) const {
        return is_crossed_[i];
    }

    const std::vector<threshold_crossing> crossings() const {
        return std::vector<threshold_crossing>(stack_.begin(), stack_.end());
    }

    /// Tests each target for changed threshold state.
    /// Crossing events are recorded for each threshold that has been
    /// crossed since current time t, and the last time the test was
    /// performed.
    void test() {
        constexpr int block_dim = 128;
        const int grid_dim = (size()+block_dim-1)/block_dim;
        test_thresholds<<<grid_dim, block_dim>>>(
            cv_to_cell_.data(), t_after_.data(), t_before_.data(),
            size(),
            stack_.base(),
            is_crossed_.data(), prev_values_.data(),
            cv_index_.data(), values_.data(), thresholds_.data());

        // Check that the number of spikes has not exceeded
        // the capacity of the stack.
        EXPECTS(stack_.size() <= stack_.capacity());
    }

    /// the number of threashold values that are being monitored
    std::size_t size() const {
        return cv_index_.size();
    }

    /// Data type used to store the crossings.
    /// Provided to make type-generic calling code.
    using crossing_list =  std::vector<threshold_crossing>;

private:
    const_iview cv_to_cell_;    // index to cell mapping: on gpu
    const_view t_before_;       // times per cell corresponding to prev_values_: on gpu
    const_view t_after_;        // times per cell corresponding to values_: on gpu
    const_view values_;         // values to watch: on gpu
    iarray cv_index_;           // compartment indexes of values to watch: on gpu

    array thresholds_;          // threshold for each watch: on gpu
    array prev_values_;         // values at previous sample time: on gpu
    iarray is_crossed_;         // bool flag for state of each watch: on gpu

    stack_type stack_;
};

} // namespace gpu
} // namespace mc
} // namespace nest
