#pragma once

#include <map>
#include <string>

#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/managed_ptr.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "gpu_stack.hpp"
#include "gpu_kernels/test_thresholds.hpp"
#include "stimulus_gpu.hpp"
#include "matrix_state_interleaved.hpp"
#include "matrix_state_flat.hpp"

namespace nest {
namespace mc {
namespace gpu {

struct backend {
    /// define the real and index types
    using value_type = double;
    using size_type  = nest::mc::cell_lid_type;

    /// define storage types
    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = typename memory::host_vector<value_type>;
    using host_iarray = typename memory::host_vector<size_type>;

    using host_view   = typename host_array::view_type;
    using host_iview  = typename host_iarray::const_view_type;

    static std::string name() {
        return "gpu";
    }

    // matrix back end implementation
    using matrix_state = matrix_state_interleaved<value_type, size_type>;

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::gpu::stimulus<backend>;

    static mechanism make_mechanism(
        const std::string& name,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->
            second(vec_v, vec_i, memory::make_const_view(weights), memory::make_const_view(node_indices));
    }

    static bool has_mechanism(const std::string& name) { return mech_map_.count(name)>0; }

    /// threshold crossing logic
    /// used as part of spike detection back end
    class threshold_watcher {
    public:
        /// stores a single crossing event
        struct threshold_crossing {
            size_type index;    // index of variable
            value_type time;    // time of crossing
            __host__ __device__
            friend bool operator==
                (const threshold_crossing& lhs, const threshold_crossing& rhs)
            {
                return lhs.index==rhs.index && lhs.time==rhs.time;
            }
        };

        using stack_type = gpu_stack<threshold_crossing>;

        threshold_watcher() = default;

        threshold_watcher(
                const_view values,
                const std::vector<size_type>& index,
                const std::vector<value_type>& thresh,
                value_type t=0):
            values_(values),
            index_(memory::make_const_view(index)),
            thresholds_(memory::make_const_view(thresh)),
            prev_values_(values),
            is_crossed_(size()),
            stack_(memory::make_managed_ptr<stack_type>(10*size()))
        {
            reset(t);
        }

        /// Remove all stored crossings that were detected in previous calls
        /// to test()
        void clear_crossings() {
            stack_->clear();
        }

        /// Reset state machine for each detector.
        /// Assume that the values in values_ have been set correctly before
        /// calling, because the values are used to determine the initial state
        void reset(value_type t=0) {
            clear_crossings();

            // Make host-side copies of the information needed to calculate
            // the initial crossed state
            auto values = memory::on_host(values_);
            auto thresholds = memory::on_host(thresholds_);
            auto index = memory::on_host(index_);

            // calculate the initial crossed state in host memory
            auto crossed = std::vector<size_type>(size());
            for (auto i: util::make_span(0u, size())) {
                crossed[i] = values[index[i]] < thresholds[i] ? 0 : 1;
            }

            // copy the initial crossed state to device memory
            is_crossed_ = memory::on_gpu(crossed);

            // reset time of last test
            t_prev_ = t;
        }

        bool is_crossed(size_type i) const {
            return is_crossed_[i];
        }

        const std::vector<threshold_crossing> crossings() const {
            return std::vector<threshold_crossing>(stack_->begin(), stack_->end());
        }

        /// The time at which the last test was performed
        value_type last_test_time() const {
            return t_prev_;
        }

        /// Tests each target for changed threshold state.
        /// Crossing events are recorded for each threshold that has been
        /// crossed since current time t, and the last time the test was
        /// performed.
        void test(value_type t) {
            EXPECTS(t_prev_<t);

            constexpr int block_dim = 128;
            const int grid_dim = (size()+block_dim-1)/block_dim;
            test_thresholds<<<grid_dim, block_dim>>>(
                t, t_prev_, size(),
                *stack_,
                is_crossed_.data(), prev_values_.data(),
                index_.data(), values_.data(), thresholds_.data());

            // Check that the number of spikes has not exceeded
            // the capacity of the stack.
            EXPECTS(stack_->size() <= stack_->capacity());

            t_prev_ = t;
        }

        /// the number of threashold values that are being monitored
        std::size_t size() const {
            return index_.size();
        }

        /// Data type used to store the crossings.
        /// Provided to make type-generic calling code.
        using crossing_list =  std::vector<threshold_crossing>;

    private:

        const_view values_;         // values to watch: on gpu
        iarray index_;              // indexes of values to watch: on gpu

        array thresholds_;          // threshold for each watch: on gpu
        value_type t_prev_;         // time of previous sample: on host
        array prev_values_;         // values at previous sample time: on host
        iarray is_crossed_;         // bool flag for state of each watch: on gpu

        memory::managed_ptr<stack_type> stack_;
    };

private:

    using maker_type = mechanism (*)(view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
