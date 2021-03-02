#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arbor/fvm_types.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/mechanism_ppack.hpp>

namespace arb {

enum class mechanismKind { point, density, revpot };

class mechanism;
using mechanism_ptr = std::unique_ptr<mechanism>;

template <typename B> class concrete_mechanism;
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
    virtual void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) = 0;

    // Peek into state variable
    virtual fvm_value_type* field_data(const std::string& var) = 0;

    // Simulation interfaces:
    virtual void initialize() {};
    virtual void update_state() {};
    virtual void update_current() {};
    virtual void deliver_events() {};
    virtual void post_event() {};
    virtual void update_ions() {};

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

struct ion_state_view {
    fvm_value_type* current_density;
    fvm_value_type* reversal_potential;
    fvm_value_type* internal_concentration;
    fvm_value_type* external_concentration;
    fvm_value_type* ionic_charge;
};

template <typename Backend>
class concrete_mechanism: public mechanism {
public:
    using backend = Backend;
    // Instantiation: allocate per-instance state; set views/pointers to shared data.
    virtual void instantiate(unsigned id, typename backend::shared_state&, const mechanism_overrides&, const mechanism_layout&) = 0;

    std::size_t size() const override { return width_; }

    std::size_t memory() const override {
        std::size_t s = object_sizeof();
        s += sizeof(data_[0])    * data_.size();
        s += sizeof(indices_[0]) * indices_.size();
        return s;
    }

    // Delegate to derived class.
    virtual void deliver_events() override { apply_events(event_stream_ptr_->marked_events()); }
    virtual void update_current() override { set_time_ptr(); compute_currents(); }
    virtual void update_state()   override { set_time_ptr(); advance_state(); }
    virtual void update_ions()    override { set_time_ptr(); write_ions(); }

protected:
    using deliverable_event_stream = typename backend::deliverable_event_stream;
    using iarray = typename backend::iarray;
    using array  = typename backend::array;

     void set_time_ptr() { ppack_ptr()->vec_t_ = vec_t_ptr_->data(); }

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

    // Generated mechanisms must implement the following methods

    // Member tables: introspection into derived mechanism fields, views etc.
    // Default implementations correspond to no corresponding fields/globals/ions.
    virtual mechanism_field_table         field_table() { return {}; }
    virtual mechanism_field_default_table field_default_table() { return {}; }
    virtual mechanism_global_table        global_table() { return {}; }
    virtual mechanism_state_table         state_table() { return {}; }
    virtual mechanism_ion_state_table     ion_state_table() { return {}; }
    virtual mechanism_ion_index_table     ion_index_table() { return {}; }

    // Returns pointer to (derived) parameter-pack object that holds:
    // * pointers to shared cell state `vec_ci_` et al.,
    // * pointer to mechanism weights `weight_`,
    // * pointer to mechanism node indices `node_index_`,
    // * mechanism global scalars and pointers to mechanism range parameters.
    // * mechanism ion_state_view objects and pointers to mechanism ion indices.
    virtual mechanism_ppack* ppack_ptr() = 0;

    // to be overridden in mechanism implemetations
    virtual void advance_state() {};
    virtual void compute_currents() {};
    virtual void apply_events(typename deliverable_event_stream::state) {};
    virtual void write_ions() {};
    virtual void init() {};
    // Report raw size in bytes of mechanism object.
    virtual std::size_t object_sizeof() const = 0;

    // events to be processed

    // indirection for accessing time in mechanisms
    const array* vec_t_ptr_;

    deliverable_event_stream* event_stream_ptr_;
    size_type width_ = 0;         // Instance width (number of CVs/sites)
    size_type num_ions_ = 0;      // Ion count
    bool mult_in_place_;          // perform multipliction in place?

    // Bulk storage for index vectors and state and parameter variables.
    iarray indices_;
    array data_;
};

} // namespace arb
