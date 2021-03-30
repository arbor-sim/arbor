#include <algorithm>
#include <cstddef>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arbor/fvm_types.hpp>
#include <arbor/common_types.hpp>
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>

#include "util/index_into.hpp"
#include "util/strprintf.hpp"
#include "util/maputil.hpp"
#include "util/padded_alloc.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

#include "backends/multicore/mechanism.hpp"
#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

using util::make_range;
using util::make_span;
using util::ptr_by_key;
using util::value_by_key;

// The derived class (typically generated code from modcc) holds pointers that need
// to be set to point inside the shared state, or into the allocated parameter/variable
// data block.
//
// In ths SIMD case, there may be a 'tail' of values that correspond to a partial
// SIMD value when the width is not a multiple of the SIMD data width. In this
// implementation we do not use SIMD masking to avoid tail values, but instead
// extend the vectors to a multiple of the SIMD width: sites/CVs corresponding to
// these past-the-end values are given a weight of zero, and any corresponding
// indices into shared state point to the last valid slot.
// The tail comprises those elements between width_ and width_padded_:
//
// * For entries in the padded tail of weight_, set weight to zero.
// * For indices in the padded tail of node_index_, set index to last valid CV index.
// * For indices in the padded tail of ion index maps, set index to last valid ion index.

void mechanism::instantiate(unsigned id, backend::shared_state& shared, const mechanism_overrides& overrides, const mechanism_layout& pos_data) {
    util::padded_allocator<> pad(shared.alignment);
    // Set internal variables
    mult_in_place_    = !pos_data.multiplicity.empty();
    width_            = pos_data.cv.size();
    num_ions_         = mech_.n_ions;
    vec_t_ptr_        = &shared.time;
    event_stream_ptr_ = &shared.deliverable_events;
    width_padded_     = math::round_up(width_, shared.alignment);     // Extend width to account for requisite SIMD padding.

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
    state_var_ptrs_.resize(mech_.n_state_vars); ppack_.state_vars = state_var_ptrs_.data();
    parameter_ptrs_.resize(mech_.n_parameters); ppack_.parameters = parameter_ptrs_.data();
    ion_ptrs_.resize(mech_.n_ions); ppack_.ion_states = ion_ptrs_.data();

    // Set ion views
    for (auto idx: make_span(mech_.n_ions)) {
        auto ion = mech_.ions[idx].name;
        auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
        ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
        if (!oion) throw arbor_internal_error(util::pprintf("multicore/mechanism: mechanism holds ion '{}' with no corresponding shared state", ion));
        ppack_.ion_states[idx] = { oion->iX_.data(), oion->eX_.data(), oion->Xi_.data(), oion->Xo_.data(), oion->charge.data() };
    }

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width_==0) return;

    auto append_chunk = [n=width_padded_](const auto& in, auto& out, auto pad, auto& ptr) {
        copy_extend(in, util::range_n(ptr, n), pad);
        out = ptr;
        ptr += n;
    };

    auto append_const = [n=width_padded_](auto in, auto& out, auto& ptr) {
        std::fill(ptr, ptr + n, in);
        out = ptr;
        ptr += n;
    };


    // Initialize state and parameter vectors with default values.
    {
        // Allocate bulk storage
        auto count = (mech_.n_state_vars + mech_.n_parameters + 1)*width_padded_ + mech_.n_globals;
        data_ = array(count, NAN, pad);
        auto base_ptr = data_.data();
        // First sub-array of data_ is used for weight_
        append_chunk(pos_data.weight, ppack_.weight, 0, base_ptr);
        // Set fields
        for (auto idx: make_span(mech_.n_parameters)) {
            append_const(mech_.parameters[idx].default_value, ppack_.parameters[idx], base_ptr);
        }
        for (auto idx: make_span(mech_.n_state_vars)) {
            append_const(mech_.state_vars[idx].default_value, ppack_.state_vars[idx], base_ptr);
        }

        // Assign global scalar parameters
        ppack_.globals = base_ptr;
        for (auto idx: make_span(mech_.n_globals)) {
            ppack_.globals[idx] = mech_.globals[idx].default_value;
        }
        for (auto& [k, v]: overrides.globals) {
            auto found = false;
            for (auto idx: make_span(mech_.n_globals)) {
                if (mech_.globals[idx].name == k) {
                    ppack_.globals[idx] = v;
                    found = true;
                    break;
                }
            }
            if (!found) throw arbor_internal_error(util::pprintf("multicore/mechanism: no such mechanism global '{}'", k));
        }
    }

    // Make index bulk storage
    {
        // Allocate bulk storage
        auto count    = (mult_in_place_ ? 1 : 0) + mech_.n_ions + 1;
        indices_      = iarray(count*width_padded_, 0, pad);
        auto base_ptr = indices_.data();
        // Setup node indices
        append_chunk(pos_data.cv, ppack_.node_index, pos_data.cv.back(), base_ptr);
        auto node_index = util::range_n(ppack_.node_index, width_padded_);
        // Make SIMD index constraints and set the view
        constraints_ = make_constraint_partition(node_index, width_, iface_.partition_width);
        ppack_.index_constraints.contiguous    = constraints_.contiguous.data();
        ppack_.index_constraints.constant      = constraints_.constant.data();
        ppack_.index_constraints.independent   = constraints_.independent.data();
        ppack_.index_constraints.none          = constraints_.none.data();
        ppack_.index_constraints.n_contiguous  = constraints_.contiguous.size();
        ppack_.index_constraints.n_constant    = constraints_.constant.size();
        ppack_.index_constraints.n_independent = constraints_.independent.size();
        ppack_.index_constraints.n_none        = constraints_.none.size();
        // Create ion indices
        for (auto idx: make_span(mech_.n_ions)) {
            auto  ion = mech_.ions[idx].name;
            auto& index_ptr = ppack_.ion_states[idx].index;
            // Index into shared_state respecting ion rebindings
            auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
            ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
            if (!oion) throw arbor_internal_error(util::pprintf("multicore/mechanism: mechanism holds ion '{}' with no corresponding shared state ", ion));
            // Obtain index and move data
            auto indices = util::index_into(node_index, oion->node_index_);
            append_chunk(indices, index_ptr, util::back(indices), base_ptr);
            // Check SIMD constraints
            auto ion_index = util::range_n(index_ptr, width_padded_);
            arb_assert(compatible_index_constraints(node_index, ion_index, iface_.partition_width));
        }
        if (mult_in_place_) append_chunk(pos_data.multiplicity, ppack_.multiplicity, 0, base_ptr);
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (values.size()!=width_) throw arbor_internal_error("multicore/mechanism: mechanism parameter size mismatch");
    auto field_ptr = field_data(key);
    if (!field_ptr) throw arbor_internal_error(util::pprintf("multicore/mechanism: no such mechanism parameter '{}'", key));
    if (!width_) return;
    auto field = util::range_n(field_ptr, width_padded_);
    copy_extend(values, field, values.back());
}

fvm_value_type* mechanism::field_data(const std::string& var) {
    for (auto idx: make_span(mech_.n_parameters)) {
        if (var == mech_.parameters[idx].name) return ppack_.parameters[idx];
    }
    for (auto idx: make_span(mech_.n_state_vars)) {
        if (var == mech_.state_vars[idx].name) return ppack_.state_vars[idx];
    }
    return nullptr;
}

void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n) { std::transform(p, p + n, s, s, std::multiplies<fvm_value_type>{}); }

mechanism_field_table mechanism::field_table() {
    mechanism_field_table result;
    for (auto idx = 0ul; idx < mech_.n_parameters; ++idx) {
        result.emplace_back(mech_.parameters[idx].name, std::make_pair(ppack_.parameters[idx], mech_.parameters[idx].default_value));
    }
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(ppack_.state_vars[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_global_table mechanism::global_table() {
    mechanism_global_table result;
    for (auto idx = 0ul; idx < mech_.n_globals; ++idx) {
        result.emplace_back(mech_.globals[idx].name, ppack_.globals[idx]);
    }
    return result;
}

mechanism_state_table mechanism::state_table() {
    mechanism_state_table result;
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(ppack_.state_vars[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_ion_table mechanism::ion_table() {
    mechanism_ion_table result;
    for (auto idx = 0ul; idx < mech_.n_ions; ++idx) {
        ion_state_view v;
        v.current_density        = ppack_.ion_states[idx].current_density;
        v.external_concentration = ppack_.ion_states[idx].internal_concentration;
        v.internal_concentration = ppack_.ion_states[idx].external_concentration;
        v.ionic_charge           = ppack_.ion_states[idx].ionic_charge;
        result.emplace_back(mech_.ions[idx].name, std::make_pair(v, ppack_.ion_states[idx].index));
    }
    return result;
}

void mechanism::initialize() {
    set_time_ptr();
    iface_.init_mechanism(&ppack_);
    if (!mult_in_place_) return;
    for (auto idx: make_span(mech_.n_state_vars)) {
        multiply_in_place(ppack_.state_vars[idx], ppack_.multiplicity, ppack_.width);
    }
}
} // namespace multicore
} // namespace arb
