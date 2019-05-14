#include <cstddef>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/ion.hpp>

#include "backends/event.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/shared_state.hpp"
#include "backends/multi_event_stream_state.hpp"
#include "memory/wrappers.hpp"
#include "util/rangeutil.hpp"

using arb::memory::make_const_view;

namespace arb {
namespace gpu {

// CUDA implementation entry points:

void init_concentration_impl(
    std::size_t n, fvm_value_type* Xi, fvm_value_type* Xo, const fvm_value_type* weight_Xi,
    const fvm_value_type* weight_Xo, fvm_value_type iconc, fvm_value_type econc);

void nernst_impl(
    std::size_t n, fvm_value_type factor,
    const fvm_value_type* Xi, const fvm_value_type* Xo, fvm_value_type* eX);

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax);

void update_time_to_impl(
    std::size_t n, fvm_value_type* time_to, const fvm_value_type* time,
    fvm_value_type dt, fvm_value_type tmax);

void set_dt_impl(
    fvm_size_type nintdom, fvm_size_type ncomp, fvm_value_type* dt_intdom, fvm_value_type* dt_comp,
    const fvm_value_type* time_to, const fvm_value_type* time, const fvm_index_type* cv_to_intdom);

void add_gj_current_impl(
    fvm_size_type n_gj, const fvm_gap_junction* gj, const fvm_value_type* v, fvm_value_type* i);

void take_samples_impl(
    const multi_event_stream_state<raw_probe_info>& s,
    const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value);

// GPU-side minmax: consider CUDA kernel replacement.
std::pair<fvm_value_type, fvm_value_type> minmax_value_impl(fvm_size_type n, const fvm_value_type* v) {
    auto v_copy = memory::on_host(memory::const_device_view<fvm_value_type>(v, n));
    return util::minmax_value(v_copy);
}

// Ion state methods:

ion_state::ion_state(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area,
    unsigned // alignment/padding ignored.
):
    node_index_(make_const_view(cv)),
    iX_(cv.size(), NAN),
    eX_(cv.size(), NAN),
    Xi_(cv.size(), NAN),
    Xo_(cv.size(), NAN),
    weight_Xi_(make_const_view(iconc_norm_area)),
    weight_Xo_(make_const_view(econc_norm_area)),
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

    fvm_value_type factor = RF*temperature_K/charge;
    nernst_impl(Xi_.size(), factor, Xo_.data(), Xi_.data(), eX_.data());
}

void ion_state::init_concentration() {
    init_concentration_impl(
        Xi_.size(),
        Xi_.data(), Xo_.data(),
        weight_Xi_.data(), weight_Xo_.data(),
        default_int_concentration, default_ext_concentration);
}

void ion_state::zero_current() {
    memory::fill(iX_, 0);
}

// Shared state methods:

shared_state::shared_state(
    fvm_size_type n_intdom,
    const std::vector<fvm_index_type>& cv_to_intdom_vec,
    const std::vector<fvm_gap_junction>& gj_vec,
    unsigned // alignment parameter ignored.
):
    n_intdom(n_intdom),
    n_cv(cv_to_intdom_vec.size()),
    n_gj(gj_vec.size()),
    cv_to_intdom(make_const_view(cv_to_intdom_vec)),
    gap_junctions(make_const_view(gj_vec)),
    time(n_intdom),
    time_to(n_intdom),
    dt_intdom(n_intdom),
    dt_cv(n_cv),
    voltage(n_cv),
    current_density(n_cv),
    conductivity(n_cv),
    temperature_degC(1),
    deliverable_events(n_intdom)
{}

void shared_state::add_ion(
    ion_info info,
    const std::vector<fvm_index_type>& cv,
    const std::vector<fvm_value_type>& iconc_norm_area,
    const std::vector<fvm_value_type>& econc_norm_area)
{
    ion_data.emplace(std::piecewise_construct,
        std::forward_as_tuple(info.kind),
        std::forward_as_tuple(info, cv, iconc_norm_area, econc_norm_area, 1u));
}

void shared_state::reset(fvm_value_type initial_voltage, fvm_value_type temperature_K) {
    memory::fill(voltage, initial_voltage);
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
    memory::fill(time, 0);
    memory::fill(time_to, 0);
    memory::fill(temperature_degC, temperature_K - 273.15);

    for (auto& i: ion_data) {
        i.second.reset(temperature_K);
    }
}

void shared_state::zero_currents() {
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
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
    update_time_to_impl(n_intdom, time_to.data(), time.data(), dt_step, tmax);
}

void shared_state::set_dt() {
    set_dt_impl(n_intdom, n_cv, dt_intdom.data(), dt_cv.data(), time_to.data(), time.data(), cv_to_intdom.data());
}

void shared_state::add_gj_current() {
    add_gj_current_impl(n_gj, gap_junctions.data(), voltage.data(), current_density.data());
}

std::pair<fvm_value_type, fvm_value_type> shared_state::time_bounds() const {
    return minmax_value_impl(n_intdom, time.data());
}

std::pair<fvm_value_type, fvm_value_type> shared_state::voltage_bounds() const {
    return minmax_value_impl(n_cv, voltage.data());
}

void shared_state::take_samples(const sample_event_stream::state& s, array& sample_time, array& sample_value) {
    take_samples_impl(s, time.data(), sample_time.data(), sample_value.data());
}

// Debug interface
std::ostream& operator<<(std::ostream& o, shared_state& s) {
    o << " cv_to_intdom " << s.cv_to_intdom << "\n";
    o << " time       " << s.time << "\n";
    o << " time_to    " << s.time_to << "\n";
    o << " dt_intdom    " << s.dt_intdom << "\n";
    o << " dt_cv      " << s.dt_cv << "\n";
    o << " voltage    " << s.voltage << "\n";
    o << " current    " << s.current_density << "\n";
    o << " conductivity " << s.conductivity << "\n";
    for (auto& ki: s.ion_data) {
        auto kn = to_string(ki.first);
        auto& i = const_cast<ion_state&>(ki.second);
        o << " " << kn << ".current_density        " << i.iX_ << "\n";
        o << " " << kn << ".reversal_potential     " << i.eX_ << "\n";
        o << " " << kn << ".internal_concentration " << i.Xi_ << "\n";
        o << " " << kn << ".external_concentration " << i.Xo_ << "\n";
        o << " " << kn << ".node_index             " << i.node_index_ << "\n";
    }
    return o;
}

} // namespace gpu
} // namespace arb
