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

        s += sizeof(data_[0])        * data_.size();
        s += sizeof(ion_indices_[0]) * ion_indices_.size();
        s += sizeof(node_index_[0])  * node_index_.size();
        return s;
    }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;

    void deliver_events() override {
        // Delegate to derived class, passing in event queue state.
        nrn_deliver_events(event_stream_ptr_->marked_events());
    }
    void update_current() override {
        vec_t_ = vec_t_ptr_->data();
        nrn_current();
    }
    void update_state() override {
        vec_t_ = vec_t_ptr_->data();
        nrn_state();
    }
    void update_ions() override {
        vec_t_ = vec_t_ptr_->data();
        write_ions();
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;

    // Peek into mechanism state variable; implements arb::multicore::backend::mechanism_field_data.
    fvm_value_type* field_data(const std::string& state_var);

protected:
    size_type width_ = 0;        // Instance width (number of CVs/sites)
    size_type width_padded_ = 0; // Width rounded up to multiple of pad/alignment.
    size_type n_ion_ = 0;
    size_type n_detectors_ = 0;

    // Non-owning views onto shared cell state, excepting ion state.

    const index_type* vec_ci_;    // CV to cell index
    const index_type* vec_di_;    // CV to indom index
    const value_type* vec_t_;     // Cell index to cell-local time.
    const value_type* vec_t_to_;  // Cell index to cell-local integration step time end.
    const value_type* vec_dt_;    // CV to integration time step.
    const value_type* vec_v_;     // CV to cell membrane voltage.
    value_type* vec_i_;           // CV to cell membrane current density.
    value_type* vec_g_;           // CV to cell membrane conductivity.
    const value_type* temperature_degC_; // CV to temperature.
    const value_type* diam_um_;   // CV to diameter.
    const value_type* time_since_spike_; // Vector containing time since last spike, indexed by cell index and n_detectors_

    const array* vec_t_ptr_;
    const array* vec_t_to_ptr_;
    deliverable_event_stream* event_stream_ptr_;

    // Per-mechanism index and weight data, excepting ion indices.

    iarray node_index_;
    iarray multiplicity_;
    bool mult_in_place_;
    constraint_partition index_constraints_;
    const value_type* weight_;    // Points within data_ after instantiation.

    // Bulk storage for state and parameter variables.

    array data_;
    iarray ion_indices_;

    virtual unsigned simd_width() const { return 1; }
};

} // namespace multicore
} // namespace arb
