#include <cstddef>
#include <limits>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>

#include "backends/event.hpp"
#include "io/sepval.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/shared_state.hpp"
#include "backends/event_stream_state.hpp"
#include "backends/gpu/chunk_writer.hpp"
#include "memory/copy.hpp"
#include "memory/gpu_wrappers.hpp"
#include "memory/wrappers.hpp"
#include "util/index_into.hpp"
#include "util/rangeutil.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"
#include "util/strprintf.hpp"

using arb::memory::make_const_view;

namespace arb {
namespace gpu {

// CUDA implementation entry points:

void take_samples_impl(
    const event_stream_state<raw_probe_info>& s,
    const arb_value_type& time, arb_value_type* sample_time, arb_value_type* sample_value);

void add_scalar(std::size_t n, arb_value_type* data, arb_value_type v);

// GPU-side minmax: consider CUDA kernel replacement.
std::pair<arb_value_type, arb_value_type> minmax_value_impl(arb_size_type n, const arb_value_type* v) {
    auto v_copy = memory::on_host(memory::const_device_view<arb_value_type>(v, n));
    return util::minmax_value(v_copy);
}

// Ion state methods:

ion_state::ion_state(int charge,
                     const fvm_ion_config& ion_data,
                     unsigned, // alignment/padding ignored.
                     solver_ptr ptr):
    write_eX_(ion_data.revpot_written),
    write_Xo_(ion_data.econc_written),
    write_Xi_(ion_data.iconc_written),
    node_index_(make_const_view(ion_data.cv)),
    iX_(ion_data.cv.size(), NAN),
    eX_(ion_data.init_revpot.begin(), ion_data.init_revpot.end()),
    Xi_(ion_data.init_iconc.begin(), ion_data.init_iconc.end()),
    Xd_(ion_data.cv.size(), NAN),
    Xo_(ion_data.init_econc.begin(), ion_data.init_econc.end()),
    gX_(ion_data.cv.size(), NAN),
    init_Xi_(make_const_view(ion_data.init_iconc)),
    init_Xo_(make_const_view(ion_data.init_econc)),
    reset_Xi_(make_const_view(ion_data.reset_iconc)),
    reset_Xo_(make_const_view(ion_data.reset_econc)),
    init_eX_(make_const_view(ion_data.init_revpot)),
    charge(1u, charge),
    solver(std::move(ptr)) {
    arb_assert(node_index_.size()==init_Xi_.size());
    arb_assert(node_index_.size()==init_Xo_.size());
    arb_assert(node_index_.size()==init_eX_.size());
}

void ion_state::init_concentration() {
    // NB. not resetting Xd here, it's controlled via the solver.
    if (write_Xi_) memory::copy(init_Xi_, Xi_);
    if (write_Xo_) memory::copy(init_Xo_, Xo_);
}

void ion_state::zero_current() {
    memory::fill(gX_, 0);
    memory::fill(iX_, 0);
}

void ion_state::reset() {
    zero_current();
    memory::copy(reset_Xi_, Xd_);
    if (write_Xi_) memory::copy(reset_Xi_, Xi_);
    if (write_Xo_) memory::copy(reset_Xo_, Xo_);
    if (write_eX_) memory::copy(init_eX_, eX_);
}

// istim_state methods:

istim_state::istim_state(const fvm_stimulus_config& stim, unsigned) {
    using util::assign;

    // Translate instance-to-CV index from stim to istim_state index vectors.
    std::vector<arb_index_type> accu_index_stage;
    assign(accu_index_stage, util::index_into(stim.cv, stim.cv_unique));

    std::size_t n = accu_index_stage.size();
    std::vector<arb_value_type> envl_a, envl_t;
    std::vector<arb_index_type> edivs;

    frequency_ = make_const_view(stim.frequency);
    phase_ = make_const_view(stim.phase);

    arb_assert(n==frequency_.size());
    arb_assert(n==stim.envelope_time.size());
    arb_assert(n==stim.envelope_amplitude.size());

    edivs.reserve(n+1);
    edivs.push_back(0);

    for (auto i: util::make_span(n)) {
        arb_assert(stim.envelope_time[i].size()==stim.envelope_amplitude[i].size());
        arb_assert(util::is_sorted(stim.envelope_time[i]));

        util::append(envl_a, stim.envelope_amplitude[i]);
        util::append(envl_t, stim.envelope_time[i]);
        edivs.push_back(arb_index_type(envl_t.size()));
    }

    accu_index_ = make_const_view(accu_index_stage);
    accu_to_cv_ = make_const_view(stim.cv_unique);
    accu_stim_ = array(accu_index_.size());
    envl_amplitudes_ = make_const_view(envl_a);
    envl_times_ = make_const_view(envl_t);
    envl_divs_ = make_const_view(edivs);

    // Initial indices into envelope match partition divisions; ignore last (index n) element.
    envl_index_ = envl_divs_;

    // Initialize ppack pointers.
    ppack_.accu_index = accu_index_.data();
    ppack_.accu_to_cv = accu_to_cv_.data();
    ppack_.frequency = frequency_.data();
    ppack_.phase = phase_.data();
    ppack_.envl_amplitudes = envl_amplitudes_.data();
    ppack_.envl_times = envl_times_.data();
    ppack_.envl_divs = envl_divs_.data();
    ppack_.accu_stim = accu_stim_.data();
    ppack_.envl_index = envl_index_.data();

    // The following ppack fields must be set in add_current() before queuing kernel.
    ppack_.time = std::numeric_limits<arb_value_type>::quiet_NaN();
    ppack_.current_density = nullptr;
}

std::size_t istim_state::size() const {
    return frequency_.size();
}

void istim_state::zero_current() {
    memory::fill(accu_stim_, 0.);
}

void istim_state::reset() {
    zero_current();
    memory::copy(envl_divs_, envl_index_);
}

void istim_state::add_current(const arb_value_type time, array& current_density) {
    ppack_.time = time;
    ppack_.current_density = current_density.data();
    istim_add_current_impl((int)size(), ppack_);
}

// Shared state methods:

shared_state::shared_state(arb_size_type n_cell,
                           arb_size_type n_cv_,
                           const std::vector<arb_index_type>& cv_to_cell_vec,
                           const std::vector<arb_value_type>& init_membrane_potential,
                           const std::vector<arb_value_type>& temperature_K,
                           const std::vector<arb_value_type>& diam,
                           const std::vector<arb_index_type>& src_to_spike_,
                           const fvm_detector_info& detector,
                           unsigned, // align parameter ignored
                           arb_seed_type cbprng_seed_):
    n_detector(detector.count),
    n_cv(n_cv_),
    cv_to_cell(make_const_view(cv_to_cell_vec)),
    voltage(n_cv_),
    current_density(n_cv_),
    conductivity(n_cv_),
    init_voltage(make_const_view(init_membrane_potential)),
    temperature_degC(make_const_view(temperature_K)),
    diam_um(make_const_view(diam)),
    time_since_spike(n_cell*n_detector),
    src_to_spike(make_const_view(src_to_spike_)),
    cbprng_seed(cbprng_seed_),
    watcher{n_cv_,
            src_to_spike.data(),
            detector.cv,
            detector.threshold,
            detector.ctx}
{
    memory::fill(time_since_spike, -1.0);
    add_scalar(temperature_degC.size(), temperature_degC.data(), -273.15);
}

void shared_state::update_prng_state(mechanism& m) {
    if (!m.mech_.n_random_variables) return;
    auto const mech_id = m.mechanism_id();
    auto& store = storage[mech_id];
    store.random_numbers_.update(m);
}

void shared_state::instantiate(mechanism& m,
                               unsigned id,
                               const mechanism_overrides& overrides,
                               const mechanism_layout& pos_data,
                               const std::vector<std::pair<std::string, std::vector<arb_value_type>>>& params) {
    assert(m.iface_.backend == arb_backend_kind_gpu);
    using util::make_range;
    using util::make_span;
    using util::ptr_by_key;
    using util::value_by_key;

    bool mult_in_place = !pos_data.multiplicity.empty();
    bool peer_indices = !pos_data.peer_cv.empty();

    auto width        = pos_data.cv.size();
    auto width_padded = math::round_up(pos_data.cv.size(), alignment);

    // Assign non-owning views onto shared state:
    m.ppack_ = {0};
    m.ppack_.width            = width;
    m.ppack_.mechanism_id     = id;
    m.ppack_.vec_ci           = cv_to_cell.data();
    m.ppack_.dt               = dt;
    m.ppack_.vec_v            = voltage.data();
    m.ppack_.vec_i            = current_density.data();
    m.ppack_.vec_g            = conductivity.data();
    m.ppack_.temperature_degC = temperature_degC.data();
    m.ppack_.diam_um          = diam_um.data();
    m.ppack_.time_since_spike = time_since_spike.data();
    m.ppack_.n_detectors      = n_detector;

    if (storage.find(id) != storage.end()) throw arb::arbor_internal_error("Duplicate mech id in shared state");
    auto& store = storage[id];

    // Allocate view pointers
    store.state_vars_ = std::vector<arb_value_type*>(m.mech_.n_state_vars);
    store.parameters_ = std::vector<arb_value_type*>(m.mech_.n_parameters);
    store.ion_states_ = std::vector<arb_ion_state>(m.mech_.n_ions);
    store.globals_    = std::vector<arb_value_type>(m.mech_.n_globals);

    // Set ion views
    for (auto idx: make_span(m.mech_.n_ions)) {
        auto ion = m.mech_.ions[idx].name;
        auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
        ion_state* oion = ptr_by_key(ion_data, ion_binding);
        if (!oion) throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
        auto& ion_state = store.ion_states_[idx];
        ion_state = {0};
        ion_state.current_density         = oion->iX_.data();
        ion_state.reversal_potential      = oion->eX_.data();
        ion_state.internal_concentration  = oion->Xi_.data();
        ion_state.external_concentration  = oion->Xo_.data();
        ion_state.diffusive_concentration = oion->Xd_.data();
        ion_state.ionic_charge            = oion->charge.data();
        ion_state.conductivity            = oion->gX_.data();
    }

    // If there are no sites (is this ever meaningful?) there is nothing more to do.
    if (width==0) return;

    // Allocate and initialize state and parameter vectors with default values.
    {
        // Allocate bulk storage
        std::size_t count = (m.mech_.n_state_vars + m.mech_.n_parameters + 1)*width_padded + m.mech_.n_globals;
        store.data_ = array(count, NAN);
        chunk_writer writer(store.data_.data(), width_padded);

        // First sub-array of data_ is used for weight_
        m.ppack_.weight = writer.append_with_padding(pos_data.weight, 0);
        // Set parameters to either default or explicit setting
        for (auto idx: make_span(m.mech_.n_parameters)) {
            const auto& param = m.mech_.parameters[idx];
            const auto& it = std::find_if(params.begin(), params.end(),
                                          [&](const auto& k) { return k.first == param.name; });
            if (it != params.end()) {
                if (it->second.size() != width) throw arbor_internal_error("mechanism field size mismatch");
                 store.parameters_[idx] = writer.append_with_padding(it->second, param.default_value);
            }
            else {
                store.parameters_[idx] = writer.fill(param.default_value);
            }
        }
        // Make STATE var the default
        for (auto idx: make_span(m.mech_.n_state_vars)) {
            store.state_vars_[idx] = writer.fill(m.mech_.state_vars[idx].default_value);
        }
        // Assign global scalar parameters. NB: Last chunk, since it breaks the width striding.
        for (auto idx: make_span(m.mech_.n_globals)) store.globals_[idx] = m.mech_.globals[idx].default_value;
        for (auto& [k, v]: overrides.globals) {
            auto found = false;
            for (auto idx: make_span(m.mech_.n_globals)) {
                if (m.mech_.globals[idx].name == k) {
                    store.globals_[idx] = v;
                    found = true;
                    break;
                }
            }
            if (!found) throw arbor_internal_error(util::pprintf("gpu/mechanism: no such mechanism global '{}'", k));
        }
        m.ppack_.globals = writer.append_freely(store.globals_);
    }

    // Allocate and initialize index vectors, viz. node_index_ and any ion indices.
    {
        // Allocate bulk storage
        std::size_t count = mult_in_place + peer_indices + m.mech_.n_ions + 1;
        store.indices_ = iarray(count*width_padded);
        chunk_writer writer(store.indices_.data(), width_padded);

        // Setup node indices
        m.ppack_.node_index = writer.append_with_padding(pos_data.cv, 0);
        // Create ion indices
        for (auto idx: make_span(m.mech_.n_ions)) {
            auto  ion = m.mech_.ions[idx].name;
            // Index into shared_state respecting ion rebindings
            auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
            ion_state* oion = ptr_by_key(ion_data, ion_binding);
            if (!oion) throw arbor_internal_error("gpu/mechanism: mechanism holds ion with no corresponding shared state");
            // Obtain index and move data
            auto ni = memory::on_host(oion->node_index_);
            auto indices = util::index_into(pos_data.cv, ni);
            std::vector<arb_index_type> mech_ion_index(indices.begin(), indices.end());
            store.ion_states_[idx].index = writer.append_with_padding(mech_ion_index, 0);
        }

        m.ppack_.multiplicity = mult_in_place? writer.append_with_padding(pos_data.multiplicity, 0): nullptr;
        // `peer_index` holds the peer CV of each CV in node_index.
        // Peer CVs are only filled for gap junction mechanisms. They are used
        // to index the voltage at the other side of a gap-junction connection.
        m.ppack_.peer_index = peer_indices? writer.append_with_padding(pos_data.peer_cv, 0): nullptr;
    }

    // Shift data to GPU, set up pointers
    store.parameters_d_ = memory::on_gpu(store.parameters_);
    m.ppack_.parameters = store.parameters_d_.data();

    store.state_vars_d_ = memory::on_gpu(store.state_vars_);
    m.ppack_.state_vars = store.state_vars_d_.data();

    store.ion_states_d_ = memory::on_gpu(store.ion_states_);
    m.ppack_.ion_states = store.ion_states_d_.data();

    store.random_numbers_.instantiate(m, width_padded, pos_data, cbprng_seed);
}

void shared_state::reset() {
    memory::copy(init_voltage, voltage);
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
    time = 0;
    time_to = 0;
    memory::fill(time_since_spike, -1.0);

    for (auto& i: ion_data) {
        i.second.reset();
    }
    stim_data.reset();
}

void shared_state::zero_currents() {
    memory::fill(current_density, 0);
    memory::fill(conductivity, 0);
    for (auto& i: ion_data) {
        i.second.zero_current();
    }
    stim_data.zero_current();
}

std::pair<arb_value_type, arb_value_type> shared_state::voltage_bounds() const {
    return minmax_value_impl(n_cv, voltage.data());
}

void shared_state::take_samples() {
    sample_events.mark();
    const auto& state = sample_events.marked_events();
    if (!state.empty()) {
        take_samples_impl(state, time, sample_time.data(), sample_value.data());
    }
}

// Debug interface
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, shared_state& s) {
    using io::csv;

    o << " n_cv         " << s.n_cv << "\n";
    o << " time         " << s.time << "\n";
    o << " time_to      " << s.time_to << "\n";
    o << " voltage      " << s.voltage << "\n";
    o << " init_voltage " << s.init_voltage << "\n";
    o << " temperature  " << s.temperature_degC << "\n";
    o << " diameter     " << s.diam_um << "\n";
    o << " current      " << s.current_density << "\n";
    o << " conductivity " << s.conductivity << "\n";
    for (auto& ki: s.ion_data) {
        auto& kn = ki.first;
        auto& i = ki.second;
        o << " " << kn << "/current_density        " << csv(i.iX_) << "\n";
        o << " " << kn << "/reversal_potential     " << csv(i.eX_) << "\n";
        o << " " << kn << "/internal_concentration " << csv(i.Xi_) << "\n";
        o << " " << kn << "/external_concentration " << csv(i.Xo_) << "\n";
        o << " " << kn << "/intconc_initial        " << csv(i.init_Xi_) << "\n";
        o << " " << kn << "/extconc_initial        " << csv(i.init_Xo_) << "\n";
        o << " " << kn << "/revpot_initial         " << csv(i.init_eX_) << "\n";
        o << " " << kn << "/node_index             " << csv(i.node_index_) << "\n";
    }
    return o;
}

} // namespace gpu
} // namespace arb
