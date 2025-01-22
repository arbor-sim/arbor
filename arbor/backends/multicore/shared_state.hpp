#pragma once

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/serdes.hpp>
#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/simd/simd.hpp>

#include "fvm_layout.hpp"
#include "util/padded_alloc.hpp"
#include "util/rangeutil.hpp"
#include "threading/threading.hpp"
#include "backends/common_types.hpp"
#include "backends/rand_fwd.hpp"
#include "backends/shared_state_base.hpp"
#include "backends/multicore/threshold_watcher.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"
#include "backends/multicore/cable_solver.hpp"
#include "backends/multicore/diffusion_solver.hpp"

namespace arb {
namespace multicore {

/*
 * Ion state fields correspond to NMODL ion variables, where X
 * is replaced with the name of the ion. E.g. for calcium 'ca':
 *
 *     Field   NMODL variable   Meaning
 *     -------------------------------------------------------
 *     iX_     ica              calcium ion current density
 *     eX_     eca              calcium ion channel reversal potential
 *     Xi_     cai              internal calcium concentration
 *     Xo_     cao              external calcium concentration
 */
struct ARB_ARBOR_API ion_state {
    using solver_type = diffusion_solver;
    using solver_ptr  = std::unique_ptr<solver_type>;

    unsigned alignment = 1;   // Alignment and padding multiple.

    ion_data_flags flags_;    // Track what and when to reset / allocate

    iarray node_index_;       // Instance to CV map.
    array iX_;                // (A/m²)  current density
    array eX_;                // (mV)    reversal potential
    array Xi_;                // (mM)    internal concentration
    array Xd_;                // (mM)    diffusive internal concentration
    array Xo_;                // (mM)    external concentration
    array gX_;                // (kS/m²) per-species conductivity

    array init_Xi_;           // (mM) area-weighted initial internal concentration
    array init_Xo_;           // (mM) area-weighted initial external concentration
    array reset_Xi_;          // (mM) area-weighted user-set internal concentration
    array reset_Xo_;          // (mM) area-weighted user-set internal concentration
    array init_eX_;           // (mV) initial reversal potential

    array charge;             // charge of ionic species (global value, length 1)

    solver_ptr solver = nullptr;

    ion_state() = default;

    ion_state(const fvm_ion_config& ion_data, unsigned align, solver_ptr ptr);

    // Set ion concentrations to weighted proportion of default concentrations.
    void init_concentration();

    // Set ionic current density to zero.
    void zero_current();

    // Zero currents, reset concentrations, and reset reversal potential from initial values.
    void reset();
};

struct mech_storage {
    array data_;
    iarray indices_;
    std::size_t value_width_padded;
    constraint_partition constraints_;
    std::vector<arb_value_type>  globals_;
    std::vector<arb_value_type*> parameters_;
    std::vector<arb_value_type*> state_vars_;
    std::vector<arb_ion_state>   ion_states_;

    std::array<std::vector<arb_value_type*>, cbprng::cache_size()> random_numbers_;
    std::vector<arb_size_type> gid_;
    std::vector<arb_size_type> idx_;
    cbprng::counter_type random_number_update_counter_ = 0u;
};

struct ARB_ARBOR_API istim_state {
    unsigned alignment = 1; // Alignment and padding multiple.

    // Immutable data (post initialization):
    iarray accu_index_;     // Instance to accumulator index (accu_stim_ index) map.
    iarray accu_to_cv_;     // Accumulator index to CV map.

    array frequency_;       // (kHz) stimulus frequency per instance.
    array phase_;           // (rad) stimulus waveform phase at t=0.
    array envl_amplitudes_; // (A/m²) stimulus envelope amplitudes, partitioned by instance.
    array envl_times_;      // (A/m²) stimulus envelope timepoints, partitioned by instance.
    iarray envl_divs_;      // Partition divisions for envl_ arrays,

    // Mutable data:
    array accu_stim_;       // (A/m²) accumulated stim current / CV area, one per CV with a stimulus.
    iarray envl_index_;     // Per instance index into envl_ arrays, corresponding to last sample time.

    // Zero stim current.
    void zero_current();

    // Zero stim current, reset indices.
    void reset();

    // Contribute to current density:
    void add_current(const arb_value_type t, array& current_density);

    // Construct state from i_clamp data:
    istim_state(const fvm_stimulus_config& stim_data, unsigned align);

    istim_state() = default;
};

struct ARB_ARBOR_API shared_state:
        public shared_state_base<shared_state, array, ion_state> {

    cable_solver solver;

    unsigned alignment = 1;         // Alignment and padding multiple.
    util::padded_allocator<> alloc; // Allocator with corresponging alignment/padding.

    arb_size_type n_intdom = 0;     // Number of integration domains.
    arb_size_type n_detector = 0;   // Max number of detectors on all cells.
    arb_size_type n_cv = 0;         // Total number of CVs.

    iarray cv_to_cell;              // Maps CV index to GID
    arb_value_type time = 0.0;      // integration start time [ms].
    arb_value_type time_to = 0.0;   // integration end time [ms]
    arb_value_type dt = 0.0;        // dt [ms].
    array voltage;                  // Maps CV index to membrane voltage [mV].
    array current_density;          // Maps CV index to membrane current density contributions [A/m²].
    array conductivity;             // Maps CV index to membrane conductivity [kS/m²].
    array init_voltage;             // Maps CV index to initial membrane voltage [mV].
    array temperature_degC;         // Maps CV to local temperature (read only) [°C].
    array diam_um;                  // Maps CV to local diameter (read only) [µm].
    array area_um2;                 // Maps CV to local lateral surface area (read only) [µm²].

    array time_since_spike;         // Stores time since last spike on any detector, organized by cell.
    iarray src_to_spike;            // Maps spike source index to spike index

    arb_seed_type cbprng_seed;      // random number generator seed

    sample_event_stream sample_events;
    array sample_time;
    array sample_value;
    threshold_watcher watcher;

    // Host-side views/copies and local state.
    util::range<const arb_value_type*> sample_time_host;
    util::range<const arb_value_type*> sample_value_host;

    istim_state stim_data;
    std::unordered_map<std::string, ion_state> ion_data;
    std::unordered_map<unsigned, mech_storage> storage;
    std::unordered_map<unsigned, spike_event_stream> streams;

    shared_state() = default;

    shared_state(task_system_handle tp,
                 arb_size_type n_cell,
                 arb_size_type n_cv,
                 const std::vector<arb_index_type>& cv_to_cell_vec,
                 const std::vector<arb_value_type>& init_membrane_potential,
                 const std::vector<arb_value_type>& temperature_K,
                 const std::vector<arb_value_type>& diam,
                 const std::vector<arb_value_type>& area,
                 const std::vector<arb_index_type>& src_to_spike,
                 const fvm_detector_info& detector_info,
                 unsigned align,
                 arb_seed_type cbprng_seed_=0u);

    shared_state(task_system_handle tp,
                 arb_size_type n_cell,
                 std::vector<arb_index_type> cv_to_cell_vec,
                 const fvm_cv_discretization& D,
                 std::vector<arb_index_type> src_to_spike,
                 const fvm_detector_info& detector,
                 std::unordered_map<std::string, fvm_ion_config> ions,
                 const fvm_stimulus_config& stims,
                 unsigned align,
                 arb_seed_type cbprng_seed_ = 0u)
        : shared_state{std::move(tp),
                       n_cell,
                       D.size(),
                       cv_to_cell_vec,
                       D.init_membrane_potential,
                       D.temperature_K,
                       D.diam_um,
                       D.cv_area,
                       src_to_spike,
                       detector,
                       align,
                       cbprng_seed_}
    {
        configure_stimulus(stims);
        configure_solver(D);
        add_ions(D, ions);
    }

    // Setup a mechanism and tie its backing store to this object
    void instantiate(mechanism&,
                     unsigned,
                     const mechanism_overrides&,
                     const mechanism_layout&,
                     const std::vector<std::pair<std::string, std::vector<arb_value_type>>>&);

    void update_prng_state(mechanism&);

    void zero_currents();

    // Return minimum and maximum voltage value [mV] across cells.
    // (Used for solution bounds checking.)
    std::pair<arb_value_type, arb_value_type> voltage_bounds() const;

    // Take samples according to marked events in a sample_event_stream.
    void take_samples();

    // Reset internal state
    void reset();

    void update_sample_views() {
        sample_time_host = util::range_pointer_view(sample_time);
        sample_value_host = util::range_pointer_view(sample_value);
    }
};

// For debugging only:
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const shared_state& s);
} // namespace multicore

// Xd and gX are the only things that persist
ARB_SERDES_ENABLE_EXT(multicore::ion_state, Xd_);
ARB_SERDES_ENABLE_EXT(multicore::mech_storage,
                      data_,
                      // NOTE(serdes) ion_states_, this is just a bunch of pointers
                      random_numbers_,
                      random_number_update_counter_);
ARB_SERDES_ENABLE_EXT(multicore::shared_state,
                      cbprng_seed,
                      ion_data,
                      storage,
                      streams,
                      voltage,
                      conductivity,
                      time_since_spike,
                      time, time_to,
                      dt);
} // namespace arb
