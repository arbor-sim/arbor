#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arbor/fvm_types.hpp>
#include <arbor/mechinfo.hpp>

namespace arb {

enum class mechanismKind { point,
    density,
    revpot };

class mechanism;
using mechanism_ptr = std::unique_ptr<mechanism>;

template <typename B>
class concrete_mechanism;
template <typename B>
using concrete_mech_ptr = std::unique_ptr<concrete_mechanism<B>>;

class mechanism {
public:
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type  = fvm_size_type;

    mechanism() = default;
    mechanism(const mechanism&) = delete;

    // Return fingerprint of mechanism dynamics source description for validation/replication.
    virtual const mechanism_fingerprint& fingerprint() const = 0;

    // Name as given in mechanism source.
    virtual std::string internal_name() const { return ""; }

    // Density or point mechanism?
    virtual mechanismKind kind() const = 0;

    // Does the implementation require padding and alignment of shared data structures?
    virtual unsigned data_alignment() const { return 1; }

    // Memory use in bytes.
    virtual std::size_t memory() const = 0;

    // Width of an instance: number of CVs (density mechanism) or sites (point mechanism)
    // that the mechanism covers.
    virtual std::size_t size() const = 0;

    // Cloning makes a new object of the derived concrete mechanism type, but does not
    // copy any state.
    virtual mechanism_ptr clone() const = 0;

    // Non-global parameters can be set post-instantiation:
    virtual void set_parameter(const std::string& key, const std::vector<value_type>& values) = 0;

    // Peek into state variable
    virtual value_type* field_data(const std::string& var) = 0;

    // Simulation interfaces:
    virtual void initialize() {};
    virtual void update_state() {}
    virtual void update_current() {}
    virtual void deliver_events() {}
    virtual void update_ions() {}

    virtual ~mechanism() = default;

    // Per-cell group identifier for an instantiated mechanism.
    unsigned mechanism_id() const { return mechanism_id_; }

protected:
    // Per-cell group identifier for an instantiation of a mechanism; set by
    // concrete_mechanism<B>::instantiate()
    unsigned mechanism_id_ = -1;
};

// Backend-specific implementations provide mechanisms that are derived from `concrete_mechanism<Backend>`,
// likely via an intermediate class that captures common behaviour for that backend.
//
// `concrete_mechanism` provides the `instantiate` method, which takes the backend-specific shared state,
// together with a layout derived from the discretization, and any global parameter overrides.

struct mechanism_layout {
    // Maps in-instance index to CV index.
    std::vector<fvm_index_type> cv;

    // Maps in-instance index to compartment contribution.
    std::vector<fvm_value_type> weight;

    // Number of logical point processes at in-instance index;
    // if empty, point processes are not coalesced and all multipliers are 1.
    std::vector<fvm_index_type> multiplicity;
};

struct mechanism_overrides {
    // Global scalar parameters (any value down-conversion to fvm_value_type is the
    // responsibility of the concrete mechanism).
    std::unordered_map<std::string, double> globals;

    // Ion renaming: keys are ion dependency names as
    // reported by the mechanism info.
    std::unordered_map<std::string, std::string> ion_rebind;
};

template <typename Backend>
class concrete_mechanism: public mechanism {
public:
    using backend = Backend;
    // Instantiation: allocate per-instance state; set views/pointers to shared data.
    virtual void instantiate(unsigned id, typename backend::shared_state&, const mechanism_overrides&, const mechanism_layout&) = 0;

protected:
    using deliverable_event_stream = typename backend::deliverable_event_stream;
    using iarray = typename backend::iarray;

    // Generated mechanism field, global and ion table lookup types.
    // First component is name, second is pointer to corresponing member in
    // the mechanism's parameter pack, or for field_default_table,
    // the scalar value used to initialize the field.
    using global_table_entry = std::pair<const char*, value_type*>;
    using mechanism_global_table = std::vector<global_table_entry>;

    using state_table_entry = std::pair<const char*, value_type**>;
    using mechanism_state_table = std::vector<state_table_entry>;

    using field_table_entry = std::pair<const char*, value_type**>;
    using mechanism_field_table = std::vector<field_table_entry>;

    using field_default_entry = std::pair<const char*, value_type>;
    using mechanism_field_default_table = std::vector<field_default_entry>;

    using ion_state_entry = std::pair<const char*, ion_state_view*>;
    using mechanism_ion_state_table = std::vector<ion_state_entry>;

    using ion_index_entry = std::pair<const char*, index_type**>;
    using mechanism_ion_index_table = std::vector<ion_index_entry>;

    // Generated mechanisms must implement the following methods, together with
    // fingerprint(), clone(), kind(), nrn_init(), nrn_state(), nrn_current()
    // and deliver_events() (if required) from arb::mechanism.

    // Member tables: introspection into derived mechanism fields, views etc.
    // Default implementations correspond to no corresponding fields/globals/ions.
    virtual mechanism_field_table field_table() { return {}; }
    virtual mechanism_field_default_table field_default_table() { return {}; }
    virtual mechanism_global_table global_table() { return {}; }
    virtual mechanism_state_table state_table() { return {}; }
    virtual mechanism_ion_state_table ion_state_table() { return {}; }
    virtual mechanism_ion_index_table ion_index_table() { return {}; }

    virtual void nrn_state() {};
    virtual void nrn_current() {};
    virtual void nrn_deliver_events(typename deliverable_event_stream::state) {};
    virtual void write_ions() {};
    virtual void nrn_init() {};
    // Report raw size in bytes of mechanism object.
    virtual std::size_t object_sizeof() const = 0;
};

} // namespace arb
