#include <algorithm>
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
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/simd/simd.hpp>

#include "io/sepval.hpp"
#include "util/index_into.hpp"
#include "util/padded_alloc.hpp"
#include "util/rangeutil.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"
#include "util/strprintf.hpp"

#include "multicore_common.hpp"
#include "shared_state.hpp"
#include "fvm.hpp"

namespace arb {
namespace multicore {

using util::make_span;
using util::ptr_by_key;
using util::value_by_key;

constexpr unsigned vector_length = (unsigned) simd::simd_abi::native_width<arb_value_type>::value;
using simd_value_type = simd::simd<arb_value_type, vector_length, simd::simd_abi::default_abi>;
using simd_index_type = simd::simd<arb_index_type, vector_length, simd::simd_abi::default_abi>;
const int simd_width  = simd::width<simd_value_type>();

// Pick alignment compatible with native SIMD width for explicitly
// vectorized operations below.
//
// TODO: Is SIMD use here a win? Test and compare; may be better to leave
// these up to the compiler to optimize/auto-vectorize.

inline unsigned min_alignment(unsigned align) {
    unsigned simd_align = sizeof(arb_value_type)*simd_width;
    return math::next_pow2(std::max(align, simd_align));
}

using pad = util::padded_allocator<>;

ion_state::ion_state(const fvm_ion_config& ion_data,
                     unsigned align,
                     solver_ptr ptr):
    alignment(min_alignment(align)),
    write_eX_(ion_data.revpot_written),
    write_Xo_(ion_data.econc_written),
    write_Xi_(ion_data.iconc_written),
    node_index_(ion_data.cv.begin(), ion_data.cv.end(), pad(alignment)),
    iX_(ion_data.cv.size(), NAN, pad(alignment)),
    eX_(ion_data.init_revpot.begin(), ion_data.init_revpot.end(), pad(alignment)),
    Xi_(ion_data.init_iconc.begin(), ion_data.init_iconc.end(), pad(alignment)),
    Xd_(ion_data.reset_iconc.begin(), ion_data.reset_iconc.end(), pad(alignment)),
    Xo_(ion_data.init_econc.begin(), ion_data.init_econc.end(), pad(alignment)),
    gX_(ion_data.cv.size(), NAN, pad(alignment)),
    init_Xi_(ion_data.init_iconc.begin(), ion_data.init_iconc.end(), pad(alignment)),
    init_Xo_(ion_data.init_econc.begin(), ion_data.init_econc.end(), pad(alignment)),
    reset_Xi_(ion_data.reset_iconc.begin(), ion_data.reset_iconc.end(), pad(alignment)),
    reset_Xo_(ion_data.reset_econc.begin(), ion_data.reset_econc.end(), pad(alignment)),
    init_eX_(ion_data.init_revpot.begin(), ion_data.init_revpot.end(), pad(alignment)),
    charge(1u, ion_data.charge, pad(alignment)),
    solver(std::move(ptr)) {
    arb_assert(node_index_.size()==init_Xi_.size());
    arb_assert(node_index_.size()==init_Xo_.size());
    arb_assert(node_index_.size()==eX_.size());
    arb_assert(node_index_.size()==init_eX_.size());
}

void ion_state::init_concentration() {
    // NB. not resetting Xd here, it's controlled via the solver.
    if (write_Xi_) std::copy(init_Xi_.begin(), init_Xi_.end(), Xi_.begin());
    if (write_Xo_) std::copy(init_Xo_.begin(), init_Xo_.end(), Xo_.begin());
}

void ion_state::zero_current() {
    util::zero(gX_);
    util::zero(iX_);
}

void ion_state::reset() {
    zero_current();
    std::copy(reset_Xi_.begin(), reset_Xi_.end(), Xd_.begin());
    if (write_Xi_) std::copy(reset_Xi_.begin(), reset_Xi_.end(), Xi_.begin());
    if (write_Xo_) std::copy(reset_Xo_.begin(), reset_Xo_.end(), Xo_.begin());
    if (write_eX_) std::copy(init_eX_.begin(), init_eX_.end(), eX_.begin());
}

// istim_state methods:

istim_state::istim_state(const fvm_stimulus_config& stim, unsigned align):
    alignment(min_alignment(align)),
    accu_to_cv_(stim.cv_unique.begin(), stim.cv_unique.end(), pad(alignment)),
    frequency_(stim.frequency.begin(), stim.frequency.end(), pad(alignment)),
    phase_(stim.phase.begin(), stim.phase.end(), pad(alignment))
{
    using util::assign;

    // Translate instance-to-CV index from stim to istim_state index vectors.
    assign(accu_index_, util::index_into(stim.cv, accu_to_cv_));
    accu_stim_.resize(accu_to_cv_.size());

    std::size_t n = accu_index_.size();
    std::vector<arb_value_type> envl_a, envl_t;
    std::vector<arb_index_type> edivs;

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

    assign(envl_amplitudes_, envl_a);
    assign(envl_times_, envl_t);
    assign(envl_divs_, edivs);
    envl_index_.assign(edivs.data(), edivs.data()+n);
}

void istim_state::zero_current() {
    util::zero(accu_stim_);
}

void istim_state::reset() {
    zero_current();

    std::size_t n = envl_index_.size();
    std::copy(envl_divs_.data(), envl_divs_.data()+n, envl_index_.begin());
}

void istim_state::add_current(const arb_value_type time, array& current_density) {
    constexpr double two_pi = 2*math::pi<double>;

    // Consider vectorizing...
    for (auto i: util::count_along(accu_index_)) {
        // Advance index into envelope until either
        // - the next envelope time is greater than simulation time, or
        // - it is the last valid index for the envelope.

        arb_index_type ei_left = envl_divs_[i];
        arb_index_type ei_right = envl_divs_[i+1];

        arb_index_type ai = accu_index_[i];
        arb_index_type cv = accu_to_cv_[ai];

        if (ei_left==ei_right || time<envl_times_[ei_left]) continue;

        arb_index_type& ei = envl_index_[i];
        while (ei+1<ei_right && envl_times_[ei+1]<=time) ++ei;

        double J = envl_amplitudes_[ei]; // current density (A/m²)
        if (ei+1<ei_right) {
            // linearly interpolate:
            arb_assert(envl_times_[ei]<=time && envl_times_[ei+1]>time);
            double J1 = envl_amplitudes_[ei+1];
            double u = (time-envl_times_[ei])/(envl_times_[ei+1]-envl_times_[ei]);
            J = math::lerp(J, J1, u);
        }

        if (frequency_[i]) {
            J *= std::sin(two_pi*frequency_[i]*time + phase_[i]);
        }

        accu_stim_[ai] += J;
        current_density[cv] -= J;
    }
}

// shared_state methods:

shared_state::shared_state(task_system_handle,    // ignored in mc backend
                           arb_size_type n_cell,
                           arb_size_type n_cv_,
                           const std::vector<arb_index_type>& cv_to_cell_vec,
                           const std::vector<arb_value_type>& init_membrane_potential,
                           const std::vector<arb_value_type>& temperature_K,
                           const std::vector<arb_value_type>& diam,
                           const std::vector<arb_value_type>& area,
                           const std::vector<arb_index_type>& src_to_spike_,
                           const fvm_detector_info& detector_info,
                           unsigned align,
                           arb_seed_type cbprng_seed_):
    alignment(min_alignment(align)),
    alloc(alignment),
    n_detector(detector_info.count),
    n_cv(n_cv_),
    cv_to_cell(math::round_up(cv_to_cell_vec.size(), alignment), pad(alignment)),
    voltage(n_cv_, pad(alignment)),
    current_density(n_cv_, pad(alignment)),
    conductivity(n_cv_, pad(alignment)),
    init_voltage(init_membrane_potential.begin(), init_membrane_potential.end(), pad(alignment)),
    temperature_degC(n_cv_, pad(alignment)),
    diam_um(diam.begin(), diam.end(), pad(alignment)),
    area_um2(area.begin(), area.end(), pad(alignment)),
    time_since_spike(n_cell*n_detector, pad(alignment)),
    src_to_spike(src_to_spike_.begin(), src_to_spike_.end(), pad(alignment)),
    cbprng_seed(cbprng_seed_),
    watcher{n_cv_, src_to_spike.data(), detector_info}
{
    if (cv_to_cell_vec.size()) {
        std::copy(cv_to_cell_vec.begin(), cv_to_cell_vec.end(), cv_to_cell.begin());
        std::fill(cv_to_cell.begin() + n_cv, cv_to_cell.end(), cv_to_cell_vec.back());
    }

    util::fill(time_since_spike, -1.0);
    std::transform(temperature_K.begin(), temperature_K.end(),
                   temperature_degC.begin(),
                   [](auto T) { return T - 273.15; });
    reset_thresholds();
}

void shared_state::reset() {
    std::copy(init_voltage.begin(), init_voltage.end(), voltage.begin());
    util::zero(current_density);
    util::zero(conductivity);
    time = 0;
    util::fill(time_since_spike, -1.0);

    for (auto& i: ion_data) {
        i.second.reset();
    }

    stim_data.reset();
}

void shared_state::zero_currents() {
    util::zero(current_density);
    util::zero(conductivity);
    for (auto& [i_, d]: ion_data) d.zero_current();
    stim_data.zero_current();
}

std::pair<arb_value_type, arb_value_type> shared_state::voltage_bounds() const {
    return util::minmax_value(voltage);
}

void shared_state::take_samples() {
    sample_events.mark();
    if (!sample_events.empty()) {
        const auto [begin, end] = sample_events.marked_events();
        // Null handles are explicitly permitted, and always give a sample of zero.
        for (auto p = begin; p<end; ++p) {
            sample_time[p->offset] = time;
            sample_value[p->offset] = p->handle? *p->handle: 0;
        }
    }
}

// (Debug interface only.)
ARB_ARBOR_API std::ostream& operator<<(std::ostream& out, const shared_state& s) {
    using io::csv;

    out << "n_cv         " << s.n_cv << "\n"
        << "time         " << s.time << "\n"
        << "time_to      " << s.time_to << "\n"
        << "dt           " << s.dt << "\n"
        << "voltage      " << csv(s.voltage) << "\n"
        << "init_voltage " << csv(s.init_voltage) << "\n"
        << "temperature  " << csv(s.temperature_degC) << "\n"
        << "diameter     " << csv(s.diam_um) << "\n"
        << "area         " << csv(s.area_um2) << "\n"
        << "current      " << csv(s.current_density) << "\n"
        << "conductivity " << csv(s.conductivity) << "\n";
    for (const auto& [kn, i]: s.ion_data) {
        out << kn << "/current_density        " << csv(i.iX_) << "\n"
            << kn << "/reversal_potential     " << csv(i.eX_) << "\n"
            << kn << "/internal_concentration " << csv(i.Xi_) << "\n"
            << kn << "/external_concentration " << csv(i.Xo_) << "\n"
            << kn << "/intconc_initial        " << csv(i.init_Xi_) << "\n"
            << kn << "/extconc_initial        " << csv(i.init_Xo_) << "\n"
            << kn << "/revpot_initial         " << csv(i.init_eX_) << "\n"
            << kn << "/node_index             " << csv(i.node_index_) << "\n";
    }

    return out;
}

namespace {
template <typename T>
struct chunk_writer {
    T* end;
    const std::size_t stride;

    chunk_writer(T* data, std::size_t stride):
        end(data), stride(stride) {}

    template <typename Seq>
    T* append(const Seq& seq, T pad) {
        auto p = end;
        copy_extend(seq, util::make_range(p, end+=stride), pad);
        return p;
    }

    T* fill(T value) {
        auto p = end;
        std::fill(p, end+=stride, value);
        return p;
    }
};

template <typename V>
std::size_t extend_width(const arb::mechanism& mech, std::size_t width) {
    // Width has to accommodate mechanism alignment and SIMD width.
    std::size_t m = std::lcm(mech.data_alignment(), mech.iface_.partition_width*sizeof(V))/sizeof(V);
    return math::round_up(width, m);
}
} // anonymous namespace

void shared_state::update_prng_state(mechanism& m) {
    if (!m.mech_.n_random_variables) return;
    const auto mech_id = m.mechanism_id();
    auto& store = storage[mech_id];
    const auto counter = store.random_number_update_counter_++;
    const auto cache_idx = cbprng::cache_index(counter);

    m.ppack_.random_numbers = store.random_numbers_[cache_idx].data();

    if (cache_idx == 0) {
        // Generate random numbers every cbprng::cache_size() iterations:
        // For each random variable we will generate cbprng::cache_size() values per site
        // and there are width sites.
        // The RNG will be seeded by a global seed, the mechanism id, the variable index, the
        // current site's global cell, the site index within its cell and a counter representing
        // time.
        const auto num_rv = store.random_numbers_[cache_idx].size();
        const auto width_padded = store.value_width_padded;
        const auto width = m.ppack_.width;
        arb_value_type* dst = store.random_numbers_[0][0];
        generate_random_numbers(dst, width, width_padded, num_rv, cbprng_seed, mech_id, counter,
            store.gid_.data(), store.idx_.data());
    }
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
// The tail comprises those elements between width_ and width_padded_:
//
// * For entries in the padded tail of weight_, set weight to zero.
// * For indices in the padded tail of node_index_, set index to last valid CV index.
// * For indices in the padded tail of ion index maps, set index to last valid ion index.

void shared_state::instantiate(arb::mechanism& m,
                               unsigned id,
                               const mechanism_overrides& overrides,
                               const mechanism_layout& pos_data,
                               const std::vector<std::pair<std::string, std::vector<arb_value_type>>>& params) {
    // Mechanism indices and data require:
    // * an alignment that is a multiple of the mechansim data_alignment();
    // * a size which is a multiple of partition_width() for SIMD access.
    //
    // We used the padded_allocator to allocate arrays with the correct alignment, and allocate
    // sizes that are multiples of a width padded to account for SIMD access and per-vector alignment.

    util::padded_allocator<> pad(m.data_alignment());

    if (storage.count(id)) throw arbor_internal_error("Duplicate mechanism id in MC shared state.");
    streams[id] = deliverable_event_stream{};
    auto& store = storage[id];
    auto width = pos_data.cv.size();
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
    m.ppack_.area_um2         = area_um2.data();
    m.ppack_.time_since_spike = time_since_spike.data();
    m.ppack_.n_detectors      = n_detector;
    m.ppack_.events           = {};

    bool mult_in_place = !pos_data.multiplicity.empty();
    bool peer_indices = !pos_data.peer_cv.empty();

    // store indices for random number generation
    store.gid_ = pos_data.gid;
    store.idx_ = pos_data.idx;

    // Allocate view pointers (except globals!)
    store.state_vars_.resize(m.mech_.n_state_vars); m.ppack_.state_vars = store.state_vars_.data();
    store.parameters_.resize(m.mech_.n_parameters); m.ppack_.parameters = store.parameters_.data();
    store.ion_states_.resize(m.mech_.n_ions);       m.ppack_.ion_states = store.ion_states_.data();

    // Set ion views
    for (auto idx: make_span(m.mech_.n_ions)) {
        auto ion = m.mech_.ions[idx].name;
        auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
        auto* oion = ptr_by_key(ion_data, ion_binding);
        if (!oion) throw arbor_internal_error(util::pprintf("multicore/mechanism: mechanism holds ion '{}' with no corresponding shared state", ion));

        auto& ion_state = m.ppack_.ion_states[idx];
        ion_state = {0};
        ion_state.current_density         = oion->iX_.data();
        ion_state.reversal_potential      = oion->eX_.data();
        ion_state.internal_concentration  = oion->Xi_.data();
        ion_state.external_concentration  = oion->Xo_.data();
        ion_state.diffusive_concentration = oion->Xd_.data();
        ion_state.ionic_charge            = oion->charge.data();
        ion_state.conductivity            = oion->gX_.data();
    }

    // Initialize state and parameter vectors with default values.
    {
        // Allocate view pointers for random nubers
        std::size_t num_random_numbers_per_cv = m.mech_.n_random_variables;
        std::size_t random_number_storage = num_random_numbers_per_cv*cbprng::cache_size();
        for (auto& v : store.random_numbers_) v.resize(num_random_numbers_per_cv);
        m.ppack_.random_numbers = store.random_numbers_[0].data();

        // Allocate bulk storage
        std::size_t value_width_padded = extend_width<arb_value_type>(m, pos_data.cv.size());
        store.value_width_padded = value_width_padded;
        std::size_t count = (m.mech_.n_state_vars + m.mech_.n_parameters + 1 +
            random_number_storage)*value_width_padded + m.mech_.n_globals;
        store.data_ = array(count, NAN, pad);
        chunk_writer writer(store.data_.data(), value_width_padded);

        // First sub-array of data_ is used for weight_
        m.ppack_.weight = writer.append(pos_data.weight, 0);
        // Set parameters: either default, or explicit override
        for (auto idx: make_span(m.mech_.n_parameters)) {
            const auto& param = m.mech_.parameters[idx];
            const auto& it = std::find_if(params.begin(), params.end(),
                                          [&](const auto& k) { return k.first == param.name; });
            if (it != params.end()) {
                if (it->second.size() != width) throw arbor_internal_error("mechanism field size mismatch");
                m.ppack_.parameters[idx] = writer.append(it->second, param.default_value);
            }
            else {
                m.ppack_.parameters[idx] = writer.fill(param.default_value);
            }
        }
        // Set initial state values
        for (auto idx: make_span(m.mech_.n_state_vars)) {
            m.ppack_.state_vars[idx] = writer.fill(m.mech_.state_vars[idx].default_value);
        }
        // Set random numbers
        for (auto idx_v: make_span(num_random_numbers_per_cv))
            for (auto idx_c: make_span(cbprng::cache_size()))
                store.random_numbers_[idx_c][idx_v] = writer.fill(0);

        // Assign global scalar parameters
        m.ppack_.globals = writer.end;
        for (auto idx: make_span(m.mech_.n_globals)) {
            m.ppack_.globals[idx] = m.mech_.globals[idx].default_value;
        }
        for (auto& [k, v]: overrides.globals) {
            auto found = false;
            for (auto idx: make_span(m.mech_.n_globals)) {
                if (m.mech_.globals[idx].name == k) {
                    m.ppack_.globals[idx] = v;
                    found = true;
                    break;
                }
            }
            if (!found) throw arbor_internal_error(util::pprintf("multicore/mechanism: no such mechanism global '{}'", k));
        }
        store.globals_ = std::vector<arb_value_type>(m.ppack_.globals, m.ppack_.globals + m.mech_.n_globals);
    }

    // Make index bulk storage
    {
        // Allocate bulk storage
        std::size_t index_width_padded = extend_width<arb_index_type>(m, pos_data.cv.size());
        std::size_t count = mult_in_place + peer_indices + m.mech_.n_ions + 1;
        store.indices_ = iarray(count*index_width_padded, 0, pad);
        chunk_writer writer(store.indices_.data(), index_width_padded);
        // Setup node indices
        //   We usually insert cv.size() == width elements into node index (length: width_padded >= width)
        //   and pad by the last element of cv. If width == 0 we must choose a different pad, that will not
        //   really be used, as width == width_padded == 0. Nevertheless, we need to pass it.
        auto pad_val = pos_data.cv.empty() ? 0 : pos_data.cv.back();
        m.ppack_.node_index = writer.append(pos_data.cv, pad_val);
        auto node_index = util::range_n(m.ppack_.node_index, index_width_padded);
        // Make SIMD index constraints and set the view
        store.constraints_ = make_constraint_partition(node_index, m.ppack_.width, m.iface_.partition_width);
        m.ppack_.index_constraints.contiguous    = store.constraints_.contiguous.data();
        m.ppack_.index_constraints.constant      = store.constraints_.constant.data();
        m.ppack_.index_constraints.independent   = store.constraints_.independent.data();
        m.ppack_.index_constraints.none          = store.constraints_.none.data();
        m.ppack_.index_constraints.n_contiguous  = store.constraints_.contiguous.size();
        m.ppack_.index_constraints.n_constant    = store.constraints_.constant.size();
        m.ppack_.index_constraints.n_independent = store.constraints_.independent.size();
        m.ppack_.index_constraints.n_none        = store.constraints_.none.size();
        // Create ion indices
        for (auto idx: make_span(m.mech_.n_ions)) {
            auto  ion = m.mech_.ions[idx].name;
            // Index into shared_state respecting ion rebindings
            auto ion_binding = value_by_key(overrides.ion_rebind, ion).value_or(ion);
            ion_state* oion = ptr_by_key(ion_data, ion_binding);
            if (!oion) throw arbor_internal_error(util::pprintf("multicore/mechanism: mechanism holds ion '{}' with no corresponding shared state ", ion));
            // Obtain index and move data
            auto indices = util::index_into(node_index, oion->node_index_);
            m.ppack_.ion_states[idx].index = writer.append(indices, util::back(indices));
            // Check SIMD constraints
            arb_assert(compatible_index_constraints(node_index, util::range_n(m.ppack_.ion_states[idx].index, index_width_padded), m.iface_.partition_width));
        }
        if (mult_in_place) m.ppack_.multiplicity = writer.append(pos_data.multiplicity, 0);
        // `peer_index` holds the peer CV of each CV in node_index.
        // Peer CVs are only filled for gap junction mechanisms. They are used
        // to index the voltage at the other side of a gap-junction connection.
        if (peer_indices) m.ppack_.peer_index = writer.append(pos_data.peer_cv, pos_data.peer_cv.back());
    }
}

void shared_state::init_events(const event_lane_subrange& lanes,
                               const std::vector<target_handle>& handles,
                               const std::vector<size_t>& divs,
                               const timestep_range& dts) {
    arb::multicore::event_stream<deliverable_event>::multi_event_stream(lanes, handles, divs, dts, streams);
}


} // namespace multicore
} // namespace arb
