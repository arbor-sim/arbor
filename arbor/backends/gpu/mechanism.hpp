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

#include "backends/gpu/fvm.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/mechanism_ppack_base.hpp"

namespace arb {
namespace gpu {

// Base class for all generated mechanisms for gpu back-end.

class mechanism: public arb::concrete_mechanism<arb::gpu::backend> {
public:
protected:
    using array = arb::gpu::array;
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
        apply_events(event_stream_ptr_->marked_events());
    }
    void update_current() override {
        mechanism_ppack_base* pp = ppack_ptr();
        pp->vec_t_ = vec_t_ptr_->data();
        compute_currents();
    }
    void update_state() override {
        mechanism_ppack_base* pp = ppack_ptr();
        pp->vec_t_ = vec_t_ptr_->data();
        advance_state();
    }
    void update_ions() override {
        mechanism_ppack_base* pp = ppack_ptr();
        pp->vec_t_ = vec_t_ptr_->data();
        write_ions();
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;

    // Peek into mechanism state variable; implements arb::gpu::backend::mechanism_field_data.
    // Returns pointer to GPU memory corresponding to state variable data.
    fvm_value_type* field_data(const std::string& state_var) override;

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
    const array* vec_t_ptr_;

    // Bulk storage for index vectors and state and parameter variables.

    iarray indices_;
    array data_;
    bool mult_in_place_;
};

} // namespace gpu
} // namespace arb
