#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/optional.hpp>

#include "memory/memory.hpp"
#include "util/index_into.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"
#include "util/span.hpp"

#include "backends/gpu/mechanism.hpp"
#include "backends/gpu/fvm.hpp"

namespace arb {
namespace gpu {

using memory::make_const_view;
using util::value_by_key;
using util::make_span;

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

void mechanism::instantiate(unsigned id,
                            backend::shared_state& shared,
                            const mechanism_overrides& overrides,
                            const mechanism_layout& pos_data)
{
    // Assign global scalar parameters:

    for (auto &kv: overrides.globals) {
        if (auto opt_ptr = value_by_key(global_table(), kv.first)) {
            // Take reference to corresponding derived (generated) mechanism value member.
            value_type& global = *opt_ptr.value();
            global = kv.second;
        }
        else {
            throw arbor_internal_error("multicore/mechanism: no such mechanism global");
        }
    }

    mult_in_place_ = !pos_data.multiplicity.empty();
    mechanism_id_ = id;
    width_ = pos_data.cv.size();

    unsigned alignment = std::max(array::alignment(), iarray::alignment());
    auto width_padded_ = math::round_up(width_, alignment);

    // Assign non-owning views onto shared state:

    mechanism_ppack_base* pp = ppack_ptr(); // From derived class instance.

    pp->width_ = width_;

    pp->vec_ci_   = shared.cv_to_intdom.data();
    pp->vec_t_    = shared.time.data();
    pp->vec_t_to_ = shared.time_to.data();
    pp->vec_dt_   = shared.dt_cv.data();

    pp->vec_v_    = shared.voltage.data();
    pp->vec_i_    = shared.current_density.data();
    pp->vec_g_    = shared.conductivity.data();

    pp->temperature_degC_ = shared.temperature_degC.data();
    pp->diam_um_ = shared.diam_um.data();

    auto ion_state_tbl = ion_state_table();
    num_ions_ = ion_state_tbl.size();

    for (auto i: ion_state_tbl) {
        auto ion_binding = value_by_key(overrides.ion_rebind, i.first).value_or(i.first);

        util::optional<ion_state&> oion = value_by_key(shared.ion_data, ion_binding);
        if (!oion) {
            throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
        }

        ion_state_view& ion_view = *i.second;
        ion_view.current_density = oion->iX_.data();
        ion_view.reversal_potential = oion->eX_.data();
        ion_view.internal_concentration = oion->Xi_.data();
        ion_view.external_concentration = oion->Xo_.data();
        ion_view.ionic_charge = oion->charge.data();
    }

    event_stream_ptr_ = &shared.deliverable_events;

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width_==0) {
        return;
    }

    // Allocate and initialize state and parameter vectors with default values.
    // (First sub-array of data_ is used for width_.)

    auto fields = field_table();
    std::size_t num_fields = fields.size();

    data_ = array((1+num_fields)*width_padded_, NAN);
    memory::copy(make_const_view(pos_data.weight), device_view(data_.data(), width_));
    pp->weight_ = data_.data();

    for (auto i: make_span(0, num_fields)) {
        // Take reference to corresponding derived (generated) mechanism value pointer member.
        fvm_value_type*& field_ptr = *std::get<1>(fields[i]);
        field_ptr = data_.data()+(i+1)*width_padded_;

        if (auto opt_value = value_by_key(field_default_table(), fields[i].first)) {
            memory::fill(device_view(field_ptr, width_), *opt_value);
        }
    }

    // Allocate and initialize index vectors, viz. node_index_ and any ion indices.
    // (First sub-array of indices_ is used for node_index_, last sub-array used for multiplicity_ if it is not empty)

    size_type num_elements = mult_in_place_ ? 2 + num_ions_ : 1 + num_ions_;
    indices_ = iarray((num_elements)*width_padded_);

    memory::copy(make_const_view(pos_data.cv), device_view(indices_.data(), width_));
    pp->node_index_ = indices_.data();

    auto ion_index_tbl = ion_index_table();
    arb_assert(num_ions_==ion_index_tbl.size());

    for (auto i: make_span(0, num_ions_)) {
        auto ion_binding = value_by_key(overrides.ion_rebind, ion_index_tbl[i].first).value_or(ion_index_tbl[i].first);

        util::optional<ion_state&> oion = value_by_key(shared.ion_data, ion_binding);
        if (!oion) {
            throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
        }

        auto ni = memory::on_host(oion->node_index_);
        auto indices = util::index_into(pos_data.cv, ni);
        std::vector<index_type> mech_ion_index(indices.begin(), indices.end());

        // Take reference to derived (generated) mechanism ion index pointer.
        auto& ion_index_ptr = *ion_index_tbl[i].second;
        auto index_start = indices_.data()+(i+1)*width_padded_;
        ion_index_ptr = index_start;
        memory::copy(make_const_view(mech_ion_index), device_view(index_start, width_));
    }

    if (mult_in_place_) {
        memory::copy(make_const_view(pos_data.multiplicity), device_view(indices_.data() + width_padded_, width_));
        pp->multiplicity_ = indices_.data() + (num_ions_ + 1)*width_padded_;
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (auto opt_ptr = value_by_key(field_table(), key)) {
        if (values.size()!=width_) {
            throw arbor_internal_error("gpu/mechanism: mechanism parameter size mismatch");
        }

        if (width_>0) {
            // Retrieve corresponding derived (generated) mechanism value pointer member.
            value_type* field_ptr = *opt_ptr.value();
            memory::copy(make_const_view(values), device_view(field_ptr, width_));
        }
    }
    else {
        throw arbor_internal_error("gpu/mechanism: no such mechanism parameter");
    }
}

void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n);

void mechanism::initialize() {
    nrn_init();
    mechanism_ppack_base* pp = ppack_ptr();
    auto states = state_table();

    if(mult_in_place_) {
        for (auto& state: states) {
            multiply_in_place(*state.second, pp->multiplicity_, pp->width_);
        }
    }
}


} // namespace multicore
} // namespace arb
