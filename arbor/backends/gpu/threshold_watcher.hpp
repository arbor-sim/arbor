#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>

#include "execution_context.hpp"
#include "memory/memory.hpp"
#include "util/span.hpp"

#include "backends/threshold_crossing.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/stack.hpp"

#include "stack.hpp"

namespace arb {
namespace gpu {

// CUDA implementation entry point:

void test_thresholds_impl(
    int size,
    const fvm_index_type* cv_to_intdom, const fvm_value_type* t_after, const fvm_value_type* t_before,
    stack_storage<threshold_crossing>& stack,
    fvm_index_type* is_crossed, fvm_value_type* prev_values,
    const fvm_index_type* cv_index, const fvm_value_type* values, const fvm_value_type* thresholds);

void reset_crossed_impl(
    int size,
    fvm_index_type* is_crossed,
    const fvm_index_type* cv_index, const fvm_value_type* values, const fvm_value_type* thresholds);


class threshold_watcher {
public:
    using stack_type = stack<threshold_crossing>;

    threshold_watcher() = default;
    threshold_watcher(threshold_watcher&& other) = default;
    threshold_watcher& operator=(threshold_watcher&& other) = default;

    threshold_watcher(const execution_context& ctx): stack_(ctx.gpu) {}

    threshold_watcher(
        const fvm_index_type* cv_to_intdom,
        const fvm_value_type* t_before,
        const fvm_value_type* t_after,
        const fvm_value_type* values,
        const std::vector<fvm_index_type>& cv_index,
        const std::vector<fvm_value_type>& thresholds,
        const execution_context& ctx
    ):
        cv_to_intdom_(cv_to_intdom),
        t_before_(t_before),
        t_after_(t_after),
        values_(values),
        cv_index_(memory::make_const_view(cv_index)),
        is_crossed_(cv_index.size()),
        thresholds_(memory::make_const_view(thresholds)),
        v_prev_(memory::const_host_view<fvm_value_type>(values, cv_index.size())),
        // TODO: allocates enough space for 10 spikes per watch.
        // A more robust approach might be needed to avoid overflows.
        stack_(10*size(), ctx.gpu)
    {
        crossings_.reserve(stack_.capacity());
        reset();
    }

    /// Remove all stored crossings that were detected in previous calls to test()
    void clear_crossings() {
        stack_.update_host();
        stack_.clear();
    }

    /// Reset state machine for each detector.
    /// Assume that the values in values_ have been set correctly before
    /// calling, because the values are used to determine the initial state
    void reset() {
        clear_crossings();
        if (size()>0) {
            reset_crossed_impl((int)size(), is_crossed_.data(), cv_index_.data(), values_, thresholds_.data());
        }
    }

    // Testing-only interface.
    bool is_crossed(int i) const {
        return is_crossed_[i];
    }

    const std::vector<threshold_crossing>& crossings() const {
        stack_.update_host();

        if (stack_.overflow()) {
            throw arbor_internal_error("gpu/threshold_watcher: gpu spike buffer overflow");
        }

        crossings_.clear();
        crossings_.insert(crossings_.end(), stack_.begin(), stack_.end());
        return crossings_;
    }

    /// Tests each target for changed threshold state.
    /// Crossing events are recorded for each threshold that has been
    /// crossed since current time t, and the last time the test was
    /// performed.
    void test() {
        if (size()>0) {
            test_thresholds_impl(
                (int)size(),
                cv_to_intdom_, t_after_, t_before_,
                stack_.storage(),
                is_crossed_.data(), v_prev_.data(),
                cv_index_.data(), values_, thresholds_.data());

            // Check that the number of spikes has not exceeded capacity.
            arb_assert(!stack_.overflow());
        }
    }

    /// the number of threshold values that are being monitored
    std::size_t size() const {
        return cv_index_.size();
    }

private:
    /// Non-owning pointers to gpu-side cv-to-cell map, per-cell time data,
    /// and the values for to test against thresholds.
    const fvm_index_type* cv_to_intdom_ = nullptr;
    const fvm_value_type* t_before_ = nullptr;
    const fvm_value_type* t_after_ = nullptr;
    const fvm_value_type* values_ = nullptr;

    // Threshold watch state, with data on gpu:
    iarray cv_index_;           // Compartment indexes of values to watch.
    iarray is_crossed_;         // Boolean flag for state of each watch.
    array thresholds_;          // Threshold for each watch.
    array v_prev_;              // Values at previous sample time.

    // Hybrid host/gpu data structure for accumulating threshold crossings.
    mutable stack_type stack_;

    // host side storage for the crossings
    mutable std::vector<threshold_crossing> crossings_;
};

} // namespace gpu
} // namespace arb
