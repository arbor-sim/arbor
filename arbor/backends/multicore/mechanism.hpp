#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechanism_ppack_base.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

// Base class for all generated mechanisms for multicore back-end.

class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
protected:
    using array  = arb::multicore::array;
    using iarray = arb::multicore::iarray;

public:
    std::size_t size() const override {
        return width_;
    }

    std::size_t memory() const override {
        std::size_t s = object_sizeof();
        s += sizeof(data_[0])    * data_.size();
        s += sizeof(indices_[0]) * indices_.size();
        return s;
    }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;

    void deliver_events() override {
        // Delegate to derived class, passing in event queue state.
        apply_events(event_stream_ptr_->marked_events());
    }
    void update_current() override {
        ppack_ptr()->vec_t_ = vec_t_ptr_->data();
        compute_currents();
    }
    void update_state() override {
        ppack_ptr()->vec_t_ = vec_t_ptr_->data();
        advance_state();
    }
    void update_ions() override {
        ppack_ptr()->vec_t_ = vec_t_ptr_->data();
        write_ions();
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;

    // Peek into mechanism state variable; implements arb::multicore::backend::mechanism_field_data.
    fvm_value_type* field_data(const std::string& state_var) override;

protected:
    fvm_size_type width_ = 0;        // Instance width (number of CVs/sites)
    fvm_size_type width_padded_ = 0; // Width rounded up to multiple of pad/alignment.
    fvm_size_type n_ion_ = 0;

    virtual mechanism_ppack_base* ppack_ptr() = 0;

    const array* vec_t_ptr_;
    deliverable_event_stream* event_stream_ptr_;

    // Per-mechanism index and weight data, excepting ion indices.

    bool mult_in_place_;
    constraint_partition index_constraints_;

    // Bulk storage for state and parameter variables.

    array data_;
    iarray indices_;

    virtual unsigned simd_width() const { return 1; }
};

} // namespace multicore
} // namespace arb
