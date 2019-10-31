#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>

#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"
#include "backends/multicore/fvm.hpp"


namespace arb {
namespace multicore {

// Base class for all generated mechanisms for multicore back-end.

class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

protected:
    using backend = arb::multicore::backend;
    using deliverable_event_stream = backend::deliverable_event_stream;

    using array  = arb::multicore::array;
    using iarray = arb::multicore::iarray;

    struct ion_state_view {
        value_type* current_density;
        value_type* reversal_potential;
        value_type* internal_concentration;
        value_type* external_concentration;
        value_type* ionic_charge;
    };

public:
    std::size_t size() const override {
        return width_;
    }

    std::size_t memory() const override {
        std::size_t s = object_sizeof();

        s += sizeof(value_type) * data_.size();
        s += sizeof(size_type) * width_padded_ * (n_ion_ + 1); // node and ion indices.
        return s;
    }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;

    void deliver_events() override {
        // Delegate to derived class, passing in event queue state.
        deliver_events(event_stream_ptr_->marked_events());
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;

protected:
    size_type width_ = 0;        // Instance width (number of CVs/sites)
    size_type width_padded_ = 0; // Width rounded up to multiple of pad/alignment.
    size_type n_ion_ = 0;

    // Non-owning views onto shared cell state, excepting ion state.

    const index_type* vec_ci_;    // CV to cell index.
    const value_type* vec_t_;     // Cell index to cell-local time.
    const value_type* vec_t_to_;  // Cell index to cell-local integration step time end.
    const value_type* vec_dt_;    // CV to integration time step.
    const value_type* vec_v_;     // CV to cell membrane voltage.
    value_type* vec_i_;           // CV to cell membrane current density.
    value_type* vec_g_;           // CV to cell membrane conductivity.
    const value_type* temperature_degC_; // CV to temperature.
    const value_type* diam_um_;   // CV to diameter.
    deliverable_event_stream* event_stream_ptr_;

    // Per-mechanism index and weight data, excepting ion indices.

    iarray node_index_;
    iarray multiplicity_;
    bool mult_in_place_;
    constraint_partition index_constraints_;
    const value_type* weight_;    // Points within data_ after instantiation.

    // Bulk storage for state and parameter variables.

    array data_;

    // Generated mechanism field, global and ion table lookup types.
    // First component is name, second is pointer to corresponing member in 
    // the mechanism's parameter pack, or for field_default_table,
    // the scalar value used to initialize the field.

    using global_table_entry = std::pair<const char*, value_type*>;
    using mechanism_global_table = std::vector<global_table_entry>;

    using state_table_entry = std::pair<const char*, value_type**>;
    using mechanism_state_table = std::vector<state_table_entry>;

    using field_table_entry = std::pair<const char*, value_type**>;
    using mechanism_field_table = std::vector<field_table_entry>;

    using field_default_entry = std::pair<const char*, value_type>;
    using mechanism_field_default_table = std::vector<field_default_entry>;

    using ion_state_entry = std::pair<const char*, ion_state_view*>;
    using mechanism_ion_state_table = std::vector<ion_state_entry>;

    using ion_index_entry = std::pair<const char*, iarray*>;
    using mechanism_ion_index_table = std::vector<ion_index_entry>;

    virtual void nrn_init() = 0;

    // Generated mechanisms must implement the following methods, together with
    // fingerprint(), clone(), kind(), nrn_init(), nrn_state(), nrn_current()
    // and deliver_events() (if required) from arb::mechanism.

    // Member tables: introspection into derived mechanism fields, views etc.
    // Default implementations correspond to no corresponding fields/globals/ions.

    virtual mechanism_field_table field_table() { return {}; }
    virtual mechanism_field_default_table field_default_table() { return {}; }
    virtual mechanism_global_table global_table() { return {}; }
    virtual mechanism_state_table state_table() { return {}; }
    virtual mechanism_ion_state_table ion_state_table() { return {}; }
    virtual mechanism_ion_index_table ion_index_table() { return {}; }

    // Report raw size in bytes of mechanism object.

    virtual std::size_t object_sizeof() const = 0;

    // Event delivery, given event queue state:

    virtual void deliver_events(deliverable_event_stream::state) {};
};

} // namespace multicore
} // namespace arb
