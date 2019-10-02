#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <arbor/fvm_types.hpp>
#include <arbor/common_types.hpp>
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/optional.hpp>

#include "util/index_into.hpp"
#include "util/maputil.hpp"
#include "util/padded_alloc.hpp"
#include "util/range.hpp"

#include "backends/multicore/mechanism.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/fvm.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

using util::make_range;
using util::value_by_key;

constexpr unsigned simd_width = S::simd_abi::native_width<fvm_value_type>::value;


// Copy elements from source sequence into destination sequence,
// and fill the remaining elements of the destination sequence
// with the given fill value.
//
// Assumes that the iterators for these sequences are at least
// forward iterators.

template <typename Source, typename Dest, typename Fill>
void copy_extend(const Source& source, Dest&& dest, const Fill& fill) {
    using std::begin;
    using std::end;

    auto dest_n = util::size(dest);
    auto source_n = util::size(source);

    auto n = source_n<dest_n? source_n: dest_n;
    auto tail = std::copy_n(begin(source), n, begin(dest));
    std::fill(tail, end(dest), fill);
}

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
            value_type& global = *opt_ptr.value();
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

    vec_ci_   = shared.cv_to_intdom.data();
    vec_t_    = shared.time.data();
    vec_t_to_ = shared.time_to.data();
    vec_dt_   = shared.dt_cv.data();

    vec_v_    = shared.voltage.data();
    vec_i_    = shared.current_density.data();
    vec_g_    = shared.conductivity.data();

    temperature_degC_ = shared.temperature_degC.data();
    diam_um_  = shared.diam_um.data();

    auto ion_state_tbl = ion_state_table();
    n_ion_ = ion_state_tbl.size();
    for (auto i: ion_state_tbl) {
        auto ion_binding = value_by_key(overrides.ion_rebind, i.first).value_or(i.first);

        util::optional<ion_state&> oion = value_by_key(shared.ion_data, ion_binding);
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

    // (First sub-array of data_ is used for width_, below.)
    data_ = array((1+n_field)*width_padded_, NAN, pad);
    for (std::size_t i = 0; i<n_field; ++i) {
        // Take reference to corresponding derived (generated) mechanism value pointer member.
        fvm_value_type*& field_ptr = *(fields[i].second);
        field_ptr = data_.data()+(i+1)*width_padded_;

        if (auto opt_value = value_by_key(field_default_table(), fields[i].first)) {
            std::fill(field_ptr, field_ptr+width_padded_, *opt_value);
        }
    }
    weight_ = data_.data();

    // Allocate and copy local state: weight, node indices, ion indices.
    // The tail comprises those elements between width_ and width_padded_:
    //
    // * For entries in the padded tail of weight_, set weight to zero.
    // * For indices in the padded tail of node_index_, set index to last valid CV index.
    // * For indices in the padded tail of ion index maps, set index to last valid ion index.

    node_index_ = iarray(width_padded_, pad);

    copy_extend(pos_data.cv, node_index_, pos_data.cv.back());
    copy_extend(pos_data.weight, make_range(data_.data(), data_.data()+width_padded_), 0);
    index_constraints_ = make_constraint_partition(node_index_, width_, simd_width);

    if (mult_in_place_) {
        multiplicity_ = iarray(width_padded_, pad);
        copy_extend(pos_data.multiplicity, multiplicity_, 1);
    }

    for (auto i: ion_index_table()) {
        auto ion_binding = value_by_key(overrides.ion_rebind, i.first).value_or(i.first);

        util::optional<ion_state&> oion = value_by_key(shared.ion_data, ion_binding);
        if (!oion) {
            throw arbor_internal_error("multicore/mechanism: mechanism holds ion with no corresponding shared state");
        }

        auto indices = util::index_into(node_index_, oion->node_index_);

        // Take reference to derived (generated) mechanism ion index member.
        auto& ion_index = *i.second;
        ion_index = iarray(width_padded_, pad);
        copy_extend(indices, ion_index, util::back(indices));

        arb_assert(compatible_index_constraints(node_index_, ion_index, simd_width));
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (auto opt_ptr = value_by_key(field_table(), key)) {
        if (values.size()!=width_) {
            throw arbor_internal_error("multicore/mechanism: mechanism parameter size mismatch");
        }

        if (width_>0) {
            // Retrieve corresponding derived (generated) mechanism value pointer member.
            value_type* field_ptr = *opt_ptr.value();
            util::range<value_type*> field(field_ptr, field_ptr+width_padded_);

            copy_extend(values, field, values.back());
        }
    }
    else {
        throw arbor_internal_error("multicore/mechanism: no such mechanism parameter");
    }
}

void mechanism::initialize() {
    nrn_init();

    auto states = state_table();

    if (mult_in_place_) {
        for (auto& state: states) {
            for (std::size_t j = 0; j < width_; ++j) {
                (*state.second)[j] *= multiplicity_[j];
            }
        }
    }
}

} // namespace multicore
} // namespace arb
