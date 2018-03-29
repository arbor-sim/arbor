#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <backends/fvm_types.hpp>
#include <common_types.hpp>

#include <math.hpp>
#include <mechanism.hpp>
#include <util/index_into.hpp>
#include <util/optional.hpp>
#include <util/maputil.hpp>
#include <util/padded_alloc.hpp>

#include "mechanism.hpp"
#include "multicore_common.hpp"
#include "fvm.hpp"

namespace arb {
namespace multicore {

// Copy elements from source sequence into destination sequence,
// and fill the remaining elements of the destination sequence
// with the given fill value.
//
// Assumes that the iterators for these sequences are at least
// forward iterators.

template <typename Source, typename Dest, typename Fill>
void copy_extend(const Source& source, Dest& dest, const Fill& fill) {
    using std::begin;
    using std::end;

    auto dest_n = util::size(dest);
    auto source_n = util::size(source);

    auto n = source_n<dest_n? source_n: dest_n;
    auto tail = std::copy_n(begin(source), n, begin(dest));
    std::fill(tail, end(dest), fill);
}

void mechanism::instantiate(fvm_size_type id, backend::shared_state& shared, const layout& w) {
    util::padded_allocator<> pad(shared.alignment);
    mechanism_id_ = id;
    width_ = w.cv.size();

    // Assign non-owning views onto shared state:

    vec_ci_   = shared.cv_to_cell.data();
    vec_t_    = shared.time.data();
    vec_t_to_ = shared.time_to.data();
    vec_dt_   = shared.dt_cv.data();

    vec_v_    = shared.voltage.data();
    vec_i_    = shared.current_density.data();

    auto ion_state_tbl = ion_state_table();
    n_ion_ = ion_state_tbl.size();
    for (auto i: ion_state_tbl) {
        util::optional<ion_state&> oion = util::value_by_key(shared.ion_data, i.first);
        if (!oion) {
            throw std::logic_error("mechanism holds ion with no corresponding shared state");
        }

        ion_state_view& ion_view = *i.second;
        ion_view.current_density = oion->iX_.data();
        ion_view.reversal_potential = oion->eX_.data();
        ion_view.internal_concentration = oion->Xi_.data();
        ion_view.external_concentration = oion->Xo_.data();
    }

    event_stream_ptr_ = &shared.deliverable_events;

    if (width_==0) {
        // If there are no sites (is this ever meaningful?) there is nothing
        // more to do.
        return;
    }

    // Extend width to account for requisite SIMD padding.
    fvm_size_type width_padded = math::round_up(width_, shared.alignment);

    // Allocate and copy local state: weight, node indices, ion indices.
    //
    // For entries in the padded tail of weight_, set weight to zero.
    // For indices in the padded tail of node_index_, set index to last valid CV index.
    // For indices in the padded tail of ion index maps, set index to last valid ion index.

    node_index_ = iarray(width_padded, pad);
    copy_extend(w.cv, node_index_, w.cv.back());

    weight_     = array(width_padded, pad);
    copy_extend(w.weight, weight_, 0);

    for (auto i: ion_index_table()) {
        std::vector<size_type> mech_ion_index;

        util::optional<ion_state&> oion = util::value_by_key(shared.ion_data, i.first);
        if (!oion) {
            throw std::logic_error("mechanism holds ion with no corresponding shared state");
        }

        auto indices = util::index_into(node_index_, oion->node_index_);

        auto& ion_index = *i.second;
        ion_index = iarray(width_padded, pad);
        copy_extend(indices, ion_index, util::back(indices));
    }

    // Allocate and initialize state and parameter vectors.

    auto fields = field_table();
    std::size_t n_field = fields.size();

    data_ = array(n_field*width_padded, NAN, pad);
    for (std::size_t i = 0; i<n_field; ++i) {
        fvm_value_type*& field_ptr = *std::get<1>(fields[i]);

        field_ptr = data_.data()+i*width_padded;
        std::fill(field_ptr, field_ptr+width_, std::get<2>(fields[i]));
    }
}

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (auto opt_ptr = util::value_by_key(field_table(), key)) {
        if (values.size()!=width_) {
            throw std::logic_error("internal error: mechanism parameter size mismatch");
        }

        value_type* field_ptr = *opt_ptr.value();
        std::copy(values.begin(), values.end(), field_ptr);
    }
    else {
        throw std::logic_error("internal error: no such mechanism parameter");
    }
}

void mechanism::set_global(const std::string& key, fvm_value_type value) {
    if (auto opt_ptr = util::value_by_key(global_table(), key)) {
        value_type& global = *opt_ptr.value();
        global = value;
    }
    else {
        throw std::logic_error("internal error: no such mechanism global");
    }
}


} // namespace multicore
} // namespace arb
