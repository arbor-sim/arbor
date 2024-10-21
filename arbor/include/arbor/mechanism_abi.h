#ifndef ARB_MECH_ABI
#define ARB_MECH_ABI

#include <arbor/arb_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// Marker for non-overlapping arrays/pointers
#ifdef __cplusplus
#if defined ( __GNUC__ ) || defined ( __clang__ )
#define ARB_NO_ALIAS __restrict__
#else
#error "Unknown compiler, please add support."
#endif
#else
#define ARB_NO_ALIAS restrict
#endif

// Version
#define ARB_MECH_ABI_VERSION_MAJOR 0
#define ARB_MECH_ABI_VERSION_MINOR 7
#define ARB_MECH_ABI_VERSION_PATCH 0
#define ARB_MECH_ABI_VERSION ((ARB_MECH_ABI_VERSION_MAJOR * 10000L * 10000L) + (ARB_MECH_ABI_VERSION_MAJOR * 10000L) + ARB_MECH_ABI_VERSION_PATCH)

typedef const char* arb_mechanism_fingerprint;

// Selectors
typedef uint32_t arb_mechanism_kind;
#define arb_mechanism_kind_nil 0
#define arb_mechanism_kind_point 1
#define arb_mechanism_kind_density 2
#define arb_mechanism_kind_reversal_potential 3
#define arb_mechanism_kind_gap_junction 4
#define arb_mechanism_kind_voltage 5

typedef uint32_t arb_backend_kind;
#define arb_backend_kind_nil 0
#define arb_backend_kind_cpu 1
#define arb_backend_kind_gpu 2

inline const char* arb_mechanism_kind_str(const arb_mechanism_kind& mech) {
    switch (mech) {
        case arb_mechanism_kind_density: return "density mechanism kind";
        case arb_mechanism_kind_point:   return "point mechanism kind";
        case arb_mechanism_kind_reversal_potential: return "reversal potential mechanism kind";
        case arb_mechanism_kind_gap_junction: return "gap junction mechanism kind";
        case arb_mechanism_kind_voltage: return "voltage mechanism kind";
        default: return "unknown mechanism kind";
    }
}

// Ion state variables; view into shared_state
typedef struct arb_ion_state {
    arb_value_type* current_density;
    arb_value_type* conductivity;
    arb_value_type* reversal_potential;
    arb_value_type* internal_concentration;
    arb_value_type* external_concentration;
    arb_value_type* diffusive_concentration;
    arb_value_type* ionic_charge;
    arb_index_type* index;
} arb_ion_state;

// Event; consumed by `apply_event`
typedef struct arb_deliverable_event_data {
    arb_size_type mech_index;
    arb_weight_type weight;
} arb_deliverable_event_data;

// A stream of events
typedef struct arb_deliverable_event_stream {
    const arb_deliverable_event_data* begin;
    const arb_deliverable_event_data* end;
}  arb_deliverable_event_stream;


// Constraints for use in SIMD implementations, see there.
typedef struct arb_constraint_partition {
    arb_size_type   n_contiguous;
    arb_size_type   n_constant;
    arb_size_type   n_independent;
    arb_size_type   n_none;
    arb_index_type* contiguous;
    arb_index_type* constant;
    arb_index_type* independent;
    arb_index_type* none;
} arb_constraint_partition;

// Parameter Pack
typedef struct arb_mechanism_ppack {
    arb_size_type   width;                       // Number of CVs.
    arb_index_type  n_detectors;                 // Number of spike detectors.
    arb_index_type* vec_ci;
    arb_value_type  dt;
    arb_value_type* vec_v;
    arb_value_type* vec_i;
    arb_value_type* vec_g;
    arb_value_type* temperature_degC;
    arb_value_type* diam_um;
    arb_value_type* area_um2;
    arb_value_type* time_since_spike;
    arb_index_type* node_index;
    arb_index_type* peer_index;
    arb_index_type* multiplicity;
    arb_value_type* weight;
    arb_size_type   mechanism_id;

    arb_deliverable_event_stream events;            // Events during the last period.
    arb_constraint_partition     index_constraints; // Index restrictions, not initialised for all backend.

    arb_value_type** parameters;                    // Array of setable parameters.    (Array)
    arb_value_type** state_vars;                    // Array of integrable state.      (Array)
    arb_value_type*  globals;                       // Array of global constant state. (Scalar)
    arb_ion_state*   ion_states;                    // Array of views into shared state.

    arb_value_type const * const * random_numbers;  // Array of random numbers
} arb_mechanism_ppack;


/* Mechanism Plugin
 *
 * Everything below has to be filled out by the plugin author/compiler.
 * The interface methods will be called with allocated and initialised `ppack`
 * data. The actual layout is unspecified, but all pointers are allocated and set
 * by the library. Plugins should never allocate memory on their own.
 */
typedef void (*arb_mechanism_method)(arb_mechanism_ppack*); // Convenience for extension methods
typedef void (*arb_mechanism_method_events)(arb_mechanism_ppack*, arb_deliverable_event_stream*);

typedef struct arb_mechanism_interface {
    arb_backend_kind   backend;               // GPU, CPU, ...
    arb_size_type      partition_width;       // Width for partitioning indices, based on SIMD for example
    arb_size_type      alignment;
    // Interface methods; hooks called by the engine during the lifetime of the mechanism.
    /* 1. init_mechanism
     *   - called once during instantiation,
     *   - setup initial state, corresponds to NMODL's INITIAL block,
     *   - will receive an allocated and initialised ppack object
     *     - pointers in ion_state_view are set to their associated values in shared state
     *     - pointers to state, parameters, globals, and constants are allocated and initialised to the given defaults.
     *     - SIMD only: index_constraint is set up
     *     - Internal values (see above) are initialised
     */
    arb_mechanism_method init_mechanism;
    /* 2. compute_currents
     * - compute ionic currents
     *   - pointers in `ion_state` are set to [ion_0, ion_1, ...] from the `ions` table
     *   - currents live in `current_density`
     * - called during each integration time step
     *   - at the start for reversal potential mechanisms, *before* current reset
     *   - after event deliver for anything else
     */
    arb_mechanism_method compute_currents;

    /* 3. apply_events
     * - consume `deliverable_events` and apply effects to internal state
     * - `deliverable_events` is setup correctly externally, is read-only for apply events
     * - called during each integration time step, right after resetting currents
     */
    arb_mechanism_method_events apply_events;
    /* 4. advanced_state
     * - called during each integration time step, after solving Hines matrices
     * - perform integration on state variables, often given as an ODE
     * - state variables live in `ppack::state_vars`
     */
    arb_mechanism_method advance_state;
    /* 5. write_ions
     * - update ionic concentrations
     *   - pointers in `ion_state` are set to [ion_0, ion_1, ...] from the `ions` table
     *   - variables live in `internal_concentration` and `external_concentration`
     * - called during each integration time step, after state integration
     */
    arb_mechanism_method write_ions;
    /* 6. post_event
     * - called during each integration time step, after checking for spikes
     * - corresponds to NET_RECEIVE in NMODL
     */
    arb_mechanism_method post_event;
} arb_mechanism_interface;

typedef struct arb_field_info {
    const char* name;
    const char* unit;
    arb_value_type default_value;
    arb_value_type range_low;
    arb_value_type range_high;
} arb_field_info;

// Ion dependency
typedef struct arb_ion_info {
    const char* name;
    bool write_int_concentration;
    bool write_ext_concentration;
    bool read_int_concentration;
    bool read_ext_concentration;
    bool use_diff_concentration;
    bool write_rev_potential;
    bool read_rev_potential;
    bool read_valence;
    bool verify_valence;
    int  expected_valence;
} arb_ion_info;

typedef struct arb_random_variable_info {
    const char* name;
    arb_size_type index;
} arb_random_variable_info;

// Backend independent data
typedef struct arb_mechanism_type {
    // Metadata
    unsigned long             abi_version;      // plugin ABI version used to build this mechanism
    arb_mechanism_fingerprint fingerprint;      // provide a unique ID
    const char*               name;             // provide unique name
    arb_mechanism_kind        kind;             // Point, Density, ReversalPotential, ...
    bool                      is_linear;        // linear, homogeneous mechanism
    bool                      has_post_events;
    // Tables
    arb_field_info*           globals;          // Global constants
    arb_size_type             n_globals;
    arb_field_info*           state_vars;       // Integrable state
    arb_size_type             n_state_vars;
    arb_field_info*           parameters;       // Mechanism parameters
    arb_size_type             n_parameters;
    arb_ion_info*             ions;             // Ion properties
    arb_size_type             n_ions;
    arb_random_variable_info* random_variables; // Random variable properties
    arb_size_type             n_random_variables;
} arb_mechanism_type;

// Bundle a type and its interfaces
typedef arb_mechanism_type (*arb_get_mechanism_type)();
typedef arb_mechanism_interface* (*arb_get_mechanism_interface)();

typedef struct arb_mechanism {
    arb_get_mechanism_type type;
    arb_get_mechanism_interface i_cpu;
    arb_get_mechanism_interface i_gpu;
} arb_mechanism;

#ifdef __cplusplus
}
#endif
#endif
