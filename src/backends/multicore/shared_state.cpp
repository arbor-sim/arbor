#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <common_types.hpp>
#include <constants.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <simd/simd.hpp>
#include <util/padded_alloc.hpp>
#include <util/rangeutil.hpp>

#include <util/debug.hpp>

#include "multi_event_stream.hpp"
#include "multicore_common.hpp"
#include "shared_state.hpp"

namespace arb {
namespace multicore {

constexpr unsigned simd_width = simd::simd_abi::native_width<fvm_value_type>::value;
using simd_value_type = simd::simd<fvm_value_type, simd_width>;
using simd_index_type = simd::simd<fvm_index_type, simd_width>;

// Pick alignment compatible with native SIMD width for explicitly
// vectorized operations below.
//
// TODO: Is SIMD use here a win? Test and compare; may be better to leave
// these up to the compiler to optimize/auto-vectorize.

inline unsigned min_alignment(unsigned align) {
    unsigned simd_align = sizeof(fvm_value_type)*simd_width;
    return math::next_pow2(std::max(align, simd_align));
}

using pad = util::padded_allocator<>;


// ion_state methods:

ion_state::ion_state(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area,
    unsigned align
):
    alignment(min_alignment(align)),
    node_index_(cv.begin(), cv.end(), pad(alignment)),
    iX_(cv.size(), NAN, pad(alignment)),
    eX_(cv.size(), NAN, pad(alignment)),
    Xi_(cv.size(), NAN, pad(alignment)),
    Xo_(cv.size(), NAN, pad(alignment)),
    weight_Xi_(iconc_norm_area.begin(), iconc_norm_area.end(), pad(alignment)),
    weight_Xo_(econc_norm_area.begin(), econc_norm_area.end(), pad(alignment)),
    charge(info.charge),
    default_int_concentration(info.default_int_concentration),
    default_ext_concentration(info.default_ext_concentration)
{
    EXPECTS(node_index_.size()==weight_Xi_.size());
    EXPECTS(node_index_.size()==weight_Xo_.size());
}

void ion_state::nernst(fvm_value_type temperature_K) {
    // Nernst equation: reversal potenial eX given by:
    //     
    //     eX = RT/zF * ln(Xo/Xi)
    //
    // where:
    //     R: universal gas constant 8.3144598 J.K-1.mol-1
    //     T: temperature in Kelvin
    //     z: valency of species (K, Na: +1) (Ca: +2)
    //     F: Faraday's constant 96485.33289 C.mol-1
    //     Xo/Xi: ratio of out/in concentrations

    // 1e3 factor required to scale from V -> mV.
    constexpr fvm_value_type RF = 1e3*constant::gas_constant/constant::faraday;

    fvm_value_type factor = RF*temperature_K/charge;
    for (std::size_t i=0; i<Xi_.size(); i+=simd_width) {
        simd_value_type xi(Xi_.data()+i);
        simd_value_type xo(Xo_.data()+i);

        auto ex = factor*log(xo/xi);
        ex.copy_to(eX_.data()+i);
    }
}

void ion_state::init_concentration() {
    for (std::size_t i=0u; i<Xi_.size(); i+=simd_width) {
        simd_value_type weight_xi(weight_Xi_.data()+i);
        simd_value_type weight_xo(weight_Xo_.data()+i);

        auto xi = default_int_concentration*weight_xi;
        xi.copy_to(Xi_.data()+i);

        auto xo = default_ext_concentration*weight_xo;
        xo.copy_to(Xo_.data()+i);
    }
}

void ion_state::zero_current() {
    util::fill(iX_, 0);
}


// shared_state methods:

shared_state::shared_state(
    fvm_size_type n_cell,
    const std::vector<fvm_size_type>& cv_to_cell_vec,
    unsigned align
):
    alignment(min_alignment(align)),
    alloc(alignment),
    n_cell(n_cell),
    n_cv(cv_to_cell_vec.size()),
    cv_to_cell(math::round_up(n_cv, alignment), pad(alignment)),
    time(n_cell, pad(alignment)),
    time_to(n_cell, pad(alignment)),
    dt_cell(n_cell, pad(alignment)),
    dt_cv(n_cv, pad(alignment)),
    voltage(n_cv, pad(alignment)),
    current_density(n_cv, pad(alignment)),
    deliverable_events(n_cell)
{
    // For indices in the padded tail of cv_to_cell, set index to last valid cell index.

    if (n_cv>0) {
        std::copy(cv_to_cell_vec.begin(), cv_to_cell_vec.end(), cv_to_cell.begin());
        std::fill(cv_to_cell.begin()+n_cv, cv_to_cell.end(), cv_to_cell_vec.back());
    }
}

void shared_state::add_ion(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area)
{
    ion_data.emplace(std::piecewise_construct,
        std::forward_as_tuple(info.kind),
        std::forward_as_tuple(info, cv, iconc_norm_area, econc_norm_area, alignment));
}

void shared_state::reset(fvm_value_type initial_voltage, fvm_value_type temperature_K) {
    util::fill(voltage, initial_voltage);
    util::fill(current_density, 0);
    util::fill(time, 0);
    util::fill(time_to, 0);

    for (auto& i: ion_data) {
        i.second.reset(temperature_K);
    }
}

void shared_state::zero_currents() {
    util::fill(current_density, 0);
    for (auto& i: ion_data) {
        i.second.zero_current();
    }
}

void shared_state::ions_init_concentration() {
    for (auto& i: ion_data) {
        i.second.init_concentration();
    }
}

void shared_state::ions_nernst_reversal_potential(fvm_value_type temperature_K) {
    for (auto& i: ion_data) {
        i.second.nernst(temperature_K);
    }
}

void shared_state::update_time_to(fvm_value_type dt_step, fvm_value_type tmax) {
    for (fvm_size_type i = 0; i<n_cell; i+=simd_width) {
        simd_value_type t(time.data()+i);
        t = min(t+dt_step, simd_value_type(tmax));
        t.copy_to(time_to.data()+i);
    }
}

void shared_state::set_dt() {
    for (fvm_size_type j = 0; j<n_cell; j+=simd_width) {
        simd_value_type t(time.data()+j);
        simd_value_type t_to(time_to.data()+j);

        auto dt = t_to-t;
        dt.copy_to(dt_cell.data()+j);
    }

    for (fvm_size_type i = 0; i<n_cv; i+=simd_width) {
        simd_index_type cell_idx(cv_to_cell.data()+i);

        simd_value_type dt(simd::indirect(dt_cell.data(), cell_idx));
        dt.copy_to(dt_cv.data()+i);
    }
}

std::pair<fvm_value_type, fvm_value_type> shared_state::time_bounds() const {
    return util::minmax_value(time);
}

std::pair<fvm_value_type, fvm_value_type> shared_state::voltage_bounds() const {
    return util::minmax_value(voltage);
}

void shared_state::take_samples(
    const sample_event_stream::state& s,
    array& sample_time,
    array& sample_value)
{
    for (fvm_size_type i = 0; i<s.n_streams(); ++i) {
        auto begin = s.begin_marked(i);
        auto end = s.end_marked(i);

        // (Note: probably not worth explicitly vectorizing this.)
        for (auto p = begin; p<end; ++p) {
            sample_time[p->offset] = time[i];
            sample_value[p->offset] = *p->handle;
        }
    }
}

// (Debug interface only.)
std::ostream& operator<<(std::ostream& out, const shared_state& s) {
    using util::csv;

    out << "n_cell " << s.n_cell << "\n----\n";
    out << "n_cv " << s.n_cv << "\n----\n";
    out << "cv_to_cell:\n" << csv(s.cv_to_cell) << "\n";
    out << "time:\n" << csv(s.time) << "\n";
    out << "time_to:\n" << csv(s.time_to) << "\n";
    out << "dt:\n" << csv(s.dt_cell) << "\n";
    out << "dt_comp:\n" << csv(s.dt_cv) << "\n";
    out << "voltage:\n" << csv(s.voltage) << "\n";
    out << "current_density:\n" << csv(s.current_density) << "\n";
    for (auto& ki: s.ion_data) {
        auto kn = to_string(ki.first);
        auto& i = const_cast<ion_state&>(ki.second);
        out << kn << ".current_density:\n" << csv(i.iX_) << "\n";
        out << kn << ".reversal_potential:\n" << csv(i.eX_) << "\n";
        out << kn << ".internal_concentration:\n" << csv(i.Xi_) << "\n";
        out << kn << ".external_concentration:\n" << csv(i.Xo_) << "\n";
    }

    return out;
}

} // namespace multicore
} // namespace arb
