#pragma once

#include <cstddef>
#include <iosfwd>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/fvm_types.hpp>

#include "fvm_layout.hpp"
#include "timestep_range.hpp"

#include "threading/threading.hpp"

#include "backends/common_types.hpp"
#include "backends/shared_state_base.hpp"
#include "backends/gpu/rand.hpp"
#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/stimulus.hpp"
#include "backends/gpu/diffusion_state.hpp"
#include "backends/gpu/matrix_state_fine.hpp"
#include "backends/gpu/threshold_watcher.hpp"

namespace arb {
namespace gpu {

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
    using solver_type = arb::gpu::diffusion_state<arb_value_type, arb_index_type>;
    using solver_ptr  = std::unique_ptr<solver_type>;

    bool write_eX_;          // is eX written?
    bool write_Xo_;          // is Xo written?
    bool write_Xi_;          // is Xi written?

    iarray node_index_; // Instance to CV map.
    array iX_;          // (A/m²) current density
    array eX_;          // (mV) reversal potential
    array Xi_;          // (mM) internal concentration
    array Xd_;          // (mM) diffusive concentration
    array Xo_;          // (mM) external concentration
    array gX_;             // (kS/m²) per-species conductivity

    array init_Xi_;     // (mM) area-weighted initial internal concentration
    array init_Xo_;     // (mM) area-weighted initial external concentration
    array reset_Xi_;    // (mM) area-weighted user-set internal concentration
    array reset_Xo_;    // (mM) area-weighted user-set internal concentration
    array init_eX_;     // (mM) initial reversal potential

    array charge;       // charge of ionic species (global, length 1)

    solver_ptr solver = nullptr;

    ion_state() = default;

    ion_state(
        int charge,
        const fvm_ion_config& ion_data,
        unsigned align,
        solver_ptr ptr);

    // Set ion concentrations to weighted proportion of default concentrations.
    void init_concentration();

    // Set ionic current density to zero.
    void zero_current();

    // Zero currents, reset concentrations, and reset reversal potential from
    // initial values.
    void reset();
};

struct ARB_ARBOR_API istim_state {
    // Immutable data (post construction/initialization):
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

    // Parameter pack presents pointers to state arrays, relevant shared state to GPU kernels.
    // Initialized at state construction.
    istim_pp ppack_;

    // Zero stim current.
    void zero_current();

    // Zero stim current, reset indices.
    void reset();

    // Contribute to current density:
    void add_current(const arb_value_type time, array& current_density);

    // Number of stimuli:
    std::size_t size() const;

    // Construct state from i_clamp data; references to shared state vectors are used to initialize ppack.
    istim_state(const fvm_stimulus_config& stim_data, unsigned);

    istim_state() = default;
};

struct ARB_ARBOR_API shared_state: shared_state_base<shared_state, array, ion_state> {
    struct mech_storage {
        mech_storage() = default;
        mech_storage(task_system_handle tp) : deliverable_events_(tp) {}

        array data_;
        iarray indices_;
        std::vector<arb_value_type>  globals_;
        std::vector<arb_value_type*> parameters_;
        std::vector<arb_value_type*> state_vars_;
        std::vector<arb_ion_state>   ion_states_;
        memory::device_vector<arb_value_type*> parameters_d_;
        memory::device_vector<arb_value_type*> state_vars_d_;
        memory::device_vector<arb_ion_state>   ion_states_d_;
        random_numbers random_numbers_;
        deliverable_event_stream deliverable_events_;
    };

    task_system_handle thread_pool;

    using cable_solver = arb::gpu::matrix_state_fine<arb_value_type, arb_index_type>;
    cable_solver solver;

    static constexpr std::size_t alignment = std::max(array::alignment(), iarray::alignment());

    arb_size_type n_intdom = 0;   // Number of distinct integration domains.
    arb_size_type n_detector = 0; // Max number of detectors on all cells.
    arb_size_type n_cv = 0;       // Total number of CVs.

    iarray cv_to_cell;       // Maps CV index to cell index.
    arb_value_type time;     // integration start time [ms].
    arb_value_type time_to;  // integration end time [ms]
    arb_value_type dt;       // dt [ms].
    array voltage;           // Maps CV index to membrane voltage [mV].
    array current_density;   // Maps CV index to current density [A/m²].
    array conductivity;      // Maps CV index to membrane conductivity [kS/m²].

    array init_voltage;      // Maps CV index to initial membrane voltage [mV].
    array temperature_degC;  // Maps CV to local temperature (read only) [°C].
    array diam_um;           // Maps CV to local diameter (read only) [µm].

    array time_since_spike;   // Stores time since last spike on any detector, organized by cell.
    iarray src_to_spike;      // Maps spike source index to spike index

    arb_seed_type cbprng_seed; // random number generator seed

    sample_event_stream sample_events;
    array sample_time;
    array sample_value;
    threshold_watcher watcher;

    // Host-side views/copies and local state.
    memory::host_vector<arb_value_type> sample_time_host;
    memory::host_vector<arb_value_type> sample_value_host;

    istim_state stim_data;
    std::unordered_map<std::string, ion_state> ion_data;
    std::unordered_map<unsigned, mech_storage> storage;

    shared_state() = default;

    shared_state(task_system_handle tp,
                 arb_size_type n_cell,
                 arb_size_type n_cv,
                 const std::vector<arb_index_type>& cv_to_cell_vec,
                 const std::vector<arb_value_type>& init_membrane_potential,
                 const std::vector<arb_value_type>& temperature_K,
                 const std::vector<arb_value_type>& diam,
                 const std::vector<arb_index_type>& src_to_spike,
                 const fvm_detector_info& detector,
                 unsigned, // align parameter ignored
                 arb_seed_type cbprng_seed_ = 0u);

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
        sample_time_host  = memory::on_host(sample_time);
        sample_value_host = memory::on_host(sample_value);
    }
};

// For debugging only
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, shared_state& s);

} // namespace gpu
} // namespace arb
