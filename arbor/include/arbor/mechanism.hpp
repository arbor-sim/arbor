#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <arbor/mechinfo.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {

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

    mechanism(const arb_mechanism_type m, const arb_mechanism_interface& i): mech_{m}, iface_{i} {}
    mechanism() = default;
    mechanism(const mechanism&) = delete;

    // Return fingerprint of mechanism dynamics source description for validation/replication.
    virtual const mechanism_fingerprint fingerprint() const { return mech_.fingerprint; };

    // Name as given in mechanism source.
    virtual std::string internal_name() const { return mech_.name; }

    // Density or point mechanism?
    virtual arb_mechanism_kind kind() const { return mech_.kind; };

    // Does the implementation require padding and alignment of shared data structures?
    virtual unsigned data_alignment() const { return 1; }

    // Memory use in bytes.
    virtual std::size_t memory() const = 0;

    // Width of an instance: number of CVs (density mechanism) or sites (point mechanism)
    // that the mechanism covers.
    virtual std::size_t size() const { return ppack_.width; }

    // Cloning makes a new object of the derived concrete mechanism type, but does not
    // copy any state.
    virtual mechanism_ptr clone() const = 0;

    // Non-global parameters can be set post-instantiation:
    virtual void set_parameter(const std::string&, const std::vector<fvm_value_type>&) {}

    // Peek into state variable
    virtual fvm_value_type* field_data(const std::string&) { return nullptr; }

    // Simulation interfaces:
    virtual void initialize() {};
    virtual void update_state() {};
    virtual void update_current() {};
    virtual void deliver_events() {};
    virtual void post_event() {};
    virtual void update_ions() {};

    virtual ~mechanism() = default;

    // Per-cell group identifier for an instantiated mechanism.
    virtual unsigned mechanism_id() const { return ppack_.mechanism_id; }

protected:
    arb_mechanism_type  mech_;
    arb_mechanism_interface iface_;
    arb_mechanism_ppack ppack_;
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

// Generated mechanism field, global and ion table lookup types.
// First component is name, second is pointer to corresponing member in
// the mechanism's parameter pack, or for field_default_table,
// the scalar value used to initialize the field.
using global_table_entry = std::pair<const char*, fvm_value_type>;
using mechanism_global_table = std::vector<global_table_entry>;

using state_table_entry = std::pair<const char*, std::pair<fvm_value_type*, fvm_value_type>>;
using mechanism_state_table = std::vector<state_table_entry>;

using field_table_entry = std::pair<const char*, std::pair<fvm_value_type*, fvm_value_type>>;
using mechanism_field_table = std::vector<field_table_entry>;

using ion_state_entry = std::pair<const char*, std::pair<ion_state_view, fvm_index_type*>>;
using mechanism_ion_table = std::vector<ion_state_entry>;

template <typename Backend>
class concrete_mechanism: public mechanism {
public:
    concrete_mechanism() = default;
    using ::arb::mechanism::mechanism;

    using backend = Backend;
    // Instantiation: allocate per-instance state; set views/pointers to shared data.
    virtual void instantiate(unsigned id, typename backend::shared_state&, const mechanism_overrides&, const mechanism_layout&) {};

    std::size_t size() const override { return width_; }

    std::size_t memory() const override {
        size_t s = 0;
        s += sizeof(data_[0])    * data_.size();
        s += sizeof(indices_[0]) * indices_.size();
        return s;
    }

    // Delegate to derived class.
    void initialize() override {
        iface_.init_mechanism(&ppack_);
    }

    void deliver_events() override {
        auto marked = event_stream_ptr_->marked_events();
        // TODO(TH) fix janky code
        ppack_.events.n_streams = marked.n;
        ppack_.events.begin     = marked.begin_offset;
        ppack_.events.end       = marked.end_offset;
        ppack_.events.events    = (arb_deliverable_event*) marked.ev_data;
        iface_.apply_events(&ppack_);
    }

    void update_current() override {
        set_time_ptr();
        iface_.compute_currents(&ppack_);
    }

    void update_state()   override {
        set_time_ptr();
        iface_.advance_state(&ppack_);
    }

    void update_ions() override {
        set_time_ptr();
        iface_.write_ions(&ppack_);
    }

protected:
    using deliverable_event_stream = typename backend::deliverable_event_stream;
    using iarray = typename backend::iarray;
    using array  = typename backend::array;

     void set_time_ptr() { ppack_.vec_t = vec_t_ptr_->data(); }

    virtual mechanism_field_table field_table() { return {}; }
    virtual mechanism_global_table global_table() { return {}; }
    virtual mechanism_state_table state_table() { return {}; }
    virtual mechanism_ion_table ion_table() { return {}; }

    const array* vec_t_ptr_;                         // indirection for accessing time in mechanisms
    deliverable_event_stream* event_stream_ptr_;     // events to be processed
    size_type width_ = 0;                            // Instance width (number of CVs/sites)
    size_type num_ions_ = 0;                         // Ion count
    bool mult_in_place_;                             // perform multiplication in place?

    // Bulk storage for index vectors and state and parameter variables.
    iarray indices_;
    array data_;
};

} // namespace arb
