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
#include "util/maputil.hpp"
#include "util/padded_alloc.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

#include "backends/multicore/mechanism.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/fvm.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

using util::make_range;
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

void mechanism::instantiate(unsigned id, backend::shared_state& shared, const mechanism_overrides& overrides, const mechanism_layout& pos_data) {
    using util::make_range;

    // Assign global scalar parameters:

    for (auto &kv: overrides.globals) {
        if (auto opt_ptr = value_by_key(global_table(), kv.first)) {
            // Take reference to corresponding derived (generated) mechanism value member.
            fvm_value_type& global = *opt_ptr.value();
            global = kv.second;
        }
        else {
            throw arbor_internal_error("multicore/mechanism: no such mechanism global");
        }
    }

    mult_in_place_ = !pos_data.multiplicity.empty();
    util::padded_allocator<> pad(shared.alignment);
    mechanism_id_ = id;
    width_ = pos_data.cv.size();

    // Assign non-owning views onto shared state:
    auto pp = (arb::multicore::mechanism_ppack*) ppack_ptr();

    pp->width_  = width_;
    pp->vec_ci_ = shared.cv_to_cell.data();
    pp->vec_di_ = shared.cv_to_intdom.data();
    pp->vec_dt_ = shared.dt_cv.data();

    pp->vec_v_  = shared.voltage.data();
    pp->vec_i_  = shared.current_density.data();
    pp->vec_g_  = shared.conductivity.data();

    pp->temperature_degC_ = shared.temperature_degC.data();
    pp->diam_um_  = shared.diam_um.data();
    pp->time_since_spike_ = shared.time_since_spike.data();

    pp->n_detectors_ = shared.n_detector;

    auto ion_state_tbl = ion_state_table();
    num_ions_ = ion_state_tbl.size();
    for (auto i: ion_state_tbl) {
        auto ion_binding = value_by_key(overrides.ion_rebind, i.first).value_or(i.first);

        ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
        if (!oion) {
            throw arbor_internal_error("multicore/mechanism: mechanism holds ion with no corresponding shared state");
        }

        ion_state_view& ion_view = *i.second;
        ion_view.current_density = oion->iX_.data();
        ion_view.reversal_potential = oion->eX_.data();
        ion_view.internal_concentration = oion->Xi_.data();
        ion_view.external_concentration = oion->Xo_.data();
        ion_view.ionic_charge = oion->charge.data();
    }

    vec_t_ptr_        = &shared.time;
    event_stream_ptr_ = &shared.deliverable_events;

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width_==0) {
        return;
    }

    // Extend width to account for requisite SIMD padding.
    width_padded_ = math::round_up(width_, shared.alignment);

    // Allocate and initialize state and parameter vectors with default values.

    auto fields = field_table();
    std::size_t n_field = fields.size();

    // (First sub-array of data_ is used for weight_, below.)
    data_ = array((1+n_field)*width_padded_, NAN, pad);
    for (std::size_t i = 0; i<n_field; ++i) {
        // Take reference to corresponding derived (generated) mechanism value pointer member.
        fvm_value_type*& field_ptr = *(fields[i].second);
        field_ptr = data_.data()+(i+1)*width_padded_;
        if (auto opt_value = value_by_key(field_default_table(), fields[i].first)) {
            std::fill(field_ptr, field_ptr+width_padded_, *opt_value);
        }
    }
    pp->weight_ = data_.data();

    // Allocate and copy local state: weight, node indices, ion indices.
    // The tail comprises those elements between width_ and width_padded_:
    //
    // * For entries in the padded tail of weight_, set weight to zero.
    // * For indices in the padded tail of node_index_, set index to last valid CV index.
    // * For indices in the padded tail of ion index maps, set index to last valid ion index.

    util::copy_extend(pos_data.weight, make_range(data_.data(), data_.data()+width_padded_), 0);

    // Make index bulk storage
    {
        auto table = ion_index_table();
        // Allocate bulk storage
        auto count = table.size() + 1 + (mult_in_place_ ? 1 : 0);
        indices_ = iarray(count*width_padded_, 0, pad);
        auto base_ptr = indices_.data();

        auto append_chunk = [&](const auto& input, auto& output, const auto& pad) {
            copy_extend(input, make_range(base_ptr, base_ptr + width_padded_), pad);
            output = base_ptr;
            base_ptr += width_padded_;
        };

        // Setup node indices
        append_chunk(pos_data.cv, pp->node_index_, pos_data.cv.back());

        auto node_index = make_range(pp->node_index_, pp->node_index_ + width_padded_);
        pp->index_constraints_ = make_constraint_partition(node_index, width_, simd_width());

        // Create ion indices
        for (const auto& [ion_name, ion_index_ptr]: table) {
            // Index into shared_state respecting ion rebindings
            auto ion_binding = value_by_key(overrides.ion_rebind, ion_name).value_or(ion_name);
            ion_state* oion = ptr_by_key(shared.ion_data, ion_binding);
            if (!oion) {
                throw arbor_internal_error("multicore/mechanism: mechanism holds ion with no corresponding shared state");
            }
            // Obtain index and move data
            auto indices = util::index_into(node_index, oion->node_index_);
            append_chunk(indices, *ion_index_ptr, util::back(indices));

            // Check SIMD constraints
            auto ion_index = make_range(*ion_index_ptr, *ion_index_ptr + width_padded_);
            arb_assert(compatible_index_constraints(node_index, ion_index, simd_width()));
        }

        if (mult_in_place_) {
            append_chunk(pos_data.multiplicity, pp->multiplicity_, 0);
        }
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (auto opt_ptr = value_by_key(field_table(), key)) {
        if (values.size()!=width_) {
            throw arbor_internal_error("multicore/mechanism: mechanism parameter size mismatch");
        }

        if (width_>0) {
            // Retrieve corresponding derived (generated) mechanism value pointer member.
            fvm_value_type* field_ptr = *opt_ptr.value();
            util::range<fvm_value_type*> field(field_ptr, field_ptr+width_padded_);

            copy_extend(values, field, values.back());
        }
    }
    else {
        throw arbor_internal_error("multicore/mechanism: no such mechanism parameter");
    }
}

void mechanism::initialize() {
    auto pp_ptr = ppack_ptr();
    pp_ptr->vec_t_ = vec_t_ptr_->data();
    init();

    auto states = state_table();

    if (mult_in_place_) {
        for (auto& state: states) {
            for (std::size_t j = 0; j < width_; ++j) {
                (*state.second)[j] *= pp_ptr->multiplicity_[j];
            }
        }
    }
}

fvm_value_type* mechanism::field_data(const std::string& field_var) {
    if (auto opt_ptr = value_by_key(field_table(), field_var)) {
        return *opt_ptr.value();
    }

    return nullptr;
}


} // namespace multicore
} // namespace arb
