#include <algorithm>
#include <cstddef>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>

#include "memory/memory.hpp"
#include "util/index_into.hpp"
#include "util/strprintf.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"
#include "util/span.hpp"

#include "backends/gpu/mechanism.hpp"
#include "backends/gpu/fvm.hpp"

namespace arb {
namespace gpu {

using memory::make_const_view;
using util::make_span;
using util::ptr_by_key;
using util::value_by_key;

template <typename T>
memory::device_view<T> device_view(T* ptr, std::size_t n) {
    return memory::device_view<T>(ptr, n);
}

template <typename T>
memory::const_device_view<T> device_view(const T* ptr, std::size_t n) {
    return memory::const_device_view<T>(ptr, n);
}

// The derived class (typically generated code from modcc) holds pointers to
// data fields. These point point to either:
//   * shared fields read/written by all mechanisms in a cell group
//     (e.g. the per-compartment voltage vec_c);
//   * or mechanism specific parameter or variable fields stored inside the
//     mechanism.
// These pointers need to be set point inside the shared state of the cell
// group, or into the allocated parameter/variable data block.
//
// The mechanism::instantiate() method takes a reference to the cell group
// shared state and discretised cell layout information, and sets the
// pointers. This also involves setting the pointers in the parameter pack,
// which is used to pass pointers to CUDA kernels.

void mechanism::instantiate(unsigned id, backend::shared_state& shared, const mechanism_overrides& overrides, const mechanism_layout& pos_data) {
    // Set internal variables
    mult_in_place_    = !pos_data.multiplicity.empty();
    width_            = pos_data.cv.size();
    num_ions_         = mech_.n_ions;
    vec_t_ptr_        = &shared.time;
    event_stream_ptr_ = &shared.deliverable_events;

    unsigned alignment = std::max(array::alignment(), iarray::alignment());
    width_padded_     = math::round_up(width_, alignment);

    // Assign non-owning views onto shared state:
    ppack_.width            = width_;
    ppack_.mechanism_id     = id;
    ppack_.vec_ci           = shared.cv_to_cell.data();
    ppack_.vec_di           = shared.cv_to_intdom.data();
    ppack_.vec_dt           = shared.dt_cv.data();
    ppack_.vec_v            = shared.voltage.data();
    ppack_.vec_i            = shared.current_density.data();
    ppack_.vec_g            = shared.conductivity.data();
    ppack_.temperature_degC = shared.temperature_degC.data();
    ppack_.diam_um          = shared.diam_um.data();
    ppack_.time_since_spike = shared.time_since_spike.data();
    ppack_.n_detectors      = shared.n_detector;

    // Allocate view pointers
    std::vector<fvm_value_type*> state_var_ptrs(mech_.n_state_vars);
    std::vector<fvm_value_type*> parameter_ptrs(mech_.n_parameters);
    std::vector<arb_ion_state>   ion_ptrs(mech_.n_ions);

    // Set ion views
    for (auto idx: make_span(mech_.n_ions)) {
        auto ion = mech_.ions[idx].name;
        auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
        ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
        if (!oion) throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
        ion_ptrs[idx] = { oion->iX_.data(), oion->eX_.data(), oion->Xi_.data(), oion->Xo_.data(), oion->charge.data() };
    }

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width_==0) return;

    auto append_chunk = [n=width_](const auto& in, auto& out, auto& ptr) {
        memory::copy(make_const_view(in), device_view(ptr, n));
        out = ptr;
        ptr += n;
    };

    auto append_const = [n=width_](auto in, auto& out, auto& ptr) {
        memory::fill(device_view(ptr, n), in);
        out = ptr;
        ptr += n;
    };

    // Allocate and initialize state and parameter vectors with default values.
    {
        // Allocate bulk storage
        auto count = (mech_.n_state_vars + mech_.n_parameters + 1)*width_padded_ + mech_.n_globals;
        data_ = array(count, NAN);
        auto base_ptr = data_.data();
        // First sub-array of data_ is used for weight_
        append_chunk(pos_data.weight, ppack_.weight, base_ptr);
        // Set fields
        for (auto idx: make_span(mech_.n_parameters)) {
            append_const(mech_.parameters[idx].default_value, parameter_ptrs[idx], base_ptr);
        }
        for (auto idx: make_span(mech_.n_state_vars)) {
            append_const(mech_.state_vars[idx].default_value, state_var_ptrs[idx], base_ptr);
        }
        // Assign global scalar parameters
        auto globals = std::vector<arb_value_type>(mech_.n_globals);
        for (auto idx: make_span(mech_.n_globals)) globals[idx] = mech_.globals[idx].default_value;
        for (auto& [k, v]: overrides.globals) {
            auto found = false;
            for (auto idx: make_span(mech_.n_globals)) {
                if (mech_.globals[idx].name == k) {
                    globals[idx] = v;
                    found = true;
                    break;
                }
                if (!found) throw arbor_internal_error(util::pprintf("gpu/mechanism: no such mechanism global '{}'", k));
            }
        }
        memory::copy(make_const_view(globals), device_view(base_ptr, mech_.n_globals));
        base_ptr += mech_.n_globals;
    }

    // For the double indirections, we need to set up the pointers here
    parameter_ptrs_= memory::device_vector<arb_value_type*>(parameters_.size());
    memory::copy(parameters_, parameter_ptrs);
    state_var_ptrs_= memory::device_vector<arb_value_type*>(state_vars_.size());
    memory::copy(state_vars_, state_var_ptrs_);
    ion_ptrs_= memory::device_vector<arb_ion_state>(ion_states_.size());
    memory::copy(ion_states_, ion_ptrs_);

    // Allocate and initialize index vectors, viz. node_index_ and any ion indices.
    {
        // Allocate bulk storage
        auto count    = (mult_in_place_ ? 1 : 0) + mech_.n_ions + 1;
        indices_      = iarray(count*width_padded_);
        auto base_ptr = indices_.data();
        // Setup node indices
        append_chunk(pos_data.cv, ppack_.node_index, base_ptr);
        // Create ion indices
        for (auto idx: make_span(mech_.n_ions)) {
            auto  ion = mech_.ions[idx].name;
            auto& index_ptr = ppack_.ion_states[idx].index;
            // Index into shared_state respecting ion rebindings
            auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
            ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
            if (!oion) throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
            // Obtain index and move data
            auto ni = memory::on_host(oion->node_index_);
            auto indices = util::index_into(pos_data.cv, ni);
            std::vector<index_type> mech_ion_index(indices.begin(), indices.end());
            append_chunk(mech_ion_index, index_ptr, base_ptr);
        }

        if (mult_in_place_) append_chunk(pos_data.multiplicity, ppack_.multiplicity, base_ptr);
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (values.size()!=width_) throw arbor_internal_error("gpu/mechanism: mechanism parameter size mismatch");
    auto field_ptr = field_data(key);
    if (!field_ptr) throw arbor_internal_error("gpu/mechanism: no such mechanism parameter");
    if (!width_) return;
    memory::copy(make_const_view(values), device_view(field_ptr, width_));
}

fvm_value_type* ::arb::gpu::mechanism::field_data(const std::string& var) {
    for (auto idx: make_span(mech_.n_parameters)) {
        if (var == mech_.parameters[idx].name) return ppack_.parameters[idx];
    }
    for (auto idx: make_span(mech_.n_state_vars)) {
        if (var == mech_.state_vars[idx].name) return ppack_.state_vars[idx];
    }
    return nullptr;
}

mechanism_field_table mechanism::field_table() {
    mechanism_field_table result;
    for (auto idx = 0ul; idx < mech_.n_parameters; ++idx) {
        result.emplace_back(mech_.parameters[idx].name, std::make_pair(parameters_[idx], mech_.parameters[idx].default_value));
    }
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(state_vars_[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_global_table mechanism::global_table() {
    mechanism_global_table result;
    for (auto idx = 0ul; idx < mech_.n_globals; ++idx) {
        result.emplace_back(mech_.globals[idx].name, globals_[idx]);
    }
    return result;
}

mechanism_state_table mechanism::state_table() {
    mechanism_state_table result;
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(state_vars_[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_ion_table mechanism::ion_table() {
    mechanism_ion_table result;
    for (auto idx = 0ul; idx < mech_.n_ions; ++idx) {
        ion_state_view v;
        v.current_density        = ion_states_[idx].current_density;
        v.external_concentration = ion_states_[idx].internal_concentration;
        v.internal_concentration = ion_states_[idx].external_concentration;
        v.ionic_charge           = ion_states_[idx].ionic_charge;
        result.emplace_back(mech_.ions[idx].name, std::make_pair(v, ppack_.ion_states[idx].index));
    }
    return result;
}

void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n);

void mechanism::initialize() {
    set_time_ptr();
    iface_.init_mechanism(&ppack_);
    if (!mult_in_place_) return;
    for (auto idx: make_span(mech_.n_state_vars)) {
        multiply_in_place(ppack_.state_vars[idx], ppack_.multiplicity, ppack_.width);
    }
}
} // namespace gpu
} // namespace arb
