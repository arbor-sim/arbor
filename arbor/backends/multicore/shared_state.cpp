#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/ion.hpp>
#include <arbor/math.hpp>
#include <arbor/simd/simd.hpp>

#include "backends/event.hpp"
#include "io/sepval.hpp"
#include "util/padded_alloc.hpp"
#include "util/rangeutil.hpp"

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
    arb_assert(node_index_.size()==weight_Xi_.size());
    arb_assert(node_index_.size()==weight_Xo_.size());
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

    simd_value_type factor = RF*temperature_K/charge;
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
    fvm_size_type n_intdom,
    const std::vector<fvm_index_type>& cv_to_intdom_vec,
    const std::vector<fvm_gap_junction>& gj_vec,
    unsigned align
):
    alignment(min_alignment(align)),
    alloc(alignment),
    n_intdom(n_intdom),
    n_cv(cv_to_intdom_vec.size()),
    n_gj(gj_vec.size()),
    cv_to_intdom(math::round_up(n_cv, alignment), pad(alignment)),
    gap_junctions(math::round_up(n_gj, alignment), pad(alignment)),
    time(n_intdom, pad(alignment)),
    time_to(n_intdom, pad(alignment)),
    dt_intdom(n_intdom, pad(alignment)),
    dt_cv(n_cv, pad(alignment)),
    voltage(n_cv, pad(alignment)),
    current_density(n_cv, pad(alignment)),
    conductivity(n_cv, pad(alignment)),
    temperature_degC(NAN),
    deliverable_events(n_intdom)
{
    // For indices in the padded tail of cv_to_intdom, set index to last valid intdom index.
    if (n_cv>0) {
        std::copy(cv_to_intdom_vec.begin(), cv_to_intdom_vec.end(), cv_to_intdom.begin());
        std::fill(cv_to_intdom.begin() + n_cv, cv_to_intdom.end(), cv_to_intdom_vec.back());
    }
    if (n_gj>0) {
        std::copy(gj_vec.begin(), gj_vec.end(), gap_junctions.begin());
        std::fill(gap_junctions.begin()+n_gj, gap_junctions.end(), gj_vec.back());
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
    util::fill(conductivity, 0);
    util::fill(time, 0);
    util::fill(time_to, 0);
    temperature_degC = temperature_K - 273.15;

    for (auto& i: ion_data) {
        i.second.reset(temperature_K);
    }
}

void shared_state::zero_currents() {
    util::fill(current_density, 0);
    util::fill(conductivity, 0);
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
    for (fvm_size_type i = 0; i<n_intdom; i+=simd_width) {
        simd_value_type t(time.data()+i);
        t = min(t+dt_step, simd_value_type(tmax));
        t.copy_to(time_to.data()+i);
    }
}

void shared_state::set_dt() {
    for (fvm_size_type j = 0; j<n_intdom; j+=simd_width) {
        simd_value_type t(time.data()+j);
        simd_value_type t_to(time_to.data()+j);

        auto dt = t_to-t;
        dt.copy_to(dt_intdom.data()+j);
    }

    for (fvm_size_type i = 0; i<n_cv; i+=simd_width) {
        simd_index_type intdom_idx(cv_to_intdom.data()+i);

        simd_value_type dt(simd::indirect(dt_intdom.data(), intdom_idx));
        dt.copy_to(dt_cv.data()+i);
    }
}

void shared_state::add_gj_current() {
    for (unsigned i = 0; i < n_gj; i++) {
        auto gj = gap_junctions[i];
        auto curr = gj.weight *
                    (voltage[gj.loc.second] - voltage[gj.loc.first]); // nA

        current_density[gj.loc.first] -= curr;
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
    using io::csv;

    out << "n_intdom     " << s.n_intdom << "\n";
    out << "n_cv       " << s.n_cv << "\n";
    out << "cv_to_intdom " << csv(s.cv_to_intdom) << "\n";
    out << "time       " << csv(s.time) << "\n";
    out << "time_to    " << csv(s.time_to) << "\n";
    out << "dt_intdom    " << csv(s.dt_intdom) << "\n";
    out << "dt_cv      " << csv(s.dt_cv) << "\n";
    out << "voltage    " << csv(s.voltage) << "\n";
    out << "current    " << csv(s.current_density) << "\n";
    out << "conductivity " << csv(s.conductivity) << "\n";
    for (auto& ki: s.ion_data) {
        auto kn = to_string(ki.first);
        auto& i = const_cast<ion_state&>(ki.second);
        out << kn << ".current_density        " << csv(i.iX_) << "\n";
        out << kn << ".reversal_potential     " << csv(i.eX_) << "\n";
        out << kn << ".internal_concentration " << csv(i.Xi_) << "\n";
        out << kn << ".external_concentration " << csv(i.Xo_) << "\n";
        out << kn << ".node_index             " << csv(i.node_index_) << "\n";
    }

    return out;
}

} // namespace multicore
} // namespace arb
