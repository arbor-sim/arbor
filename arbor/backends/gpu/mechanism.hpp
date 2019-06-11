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

#include "backends/gpu/fvm.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/mechanism_ppack_base.hpp"

namespace arb {
namespace gpu {

// Base class for all generated mechanisms for gpu back-end.

class mechanism: public arb::concrete_mechanism<arb::gpu::backend> {
public:
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

protected:
    using backend = arb::gpu::backend;
    using deliverable_event_stream = backend::deliverable_event_stream;

    using array  = arb::gpu::array;
    using iarray = arb::gpu::iarray;

public:
    std::size_t size() const override {
        return width_;
    }

    std::size_t memory() const override {
        std::size_t s = object_sizeof();

        s += sizeof(value_type) * data_.size();
        s += sizeof(index_type) * indices_.size();
        return s;
    }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;

    void deliver_events() override {
        // Delegate to derived class, passing in event queue state.
        deliver_events(event_stream_ptr_->marked_events());
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;

    void initialize() override;

protected:
    size_type width_ = 0;        // Instance width (number of CVs/sites)
    size_type num_ions_ = 0;

    // Returns pointer to (derived) parameter-pack object that holds:
    // * pointers to shared cell state `vec_ci_` et al.,
    // * pointer to mechanism weights `weight_`,
    // * pointer to mechanism node indices `node_index_`,
    // * mechanism global scalars and pointers to mechanism range parameters.
    // * mechanism ion_state_view objects and pointers to mechanism ion indices.

    virtual mechanism_ppack_base* ppack_ptr() = 0;

    deliverable_event_stream* event_stream_ptr_;

    // Bulk storage for index vectors and state and parameter variables.

    iarray indices_;
    array data_;
    bool mult_in_place_;

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

    using ion_index_entry = std::pair<const char*, const index_type**>;
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

} // namespace gpu
} // namespace arb
