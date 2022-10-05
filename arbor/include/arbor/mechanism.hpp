#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism_abi.h>
#include <arbor/mechinfo.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/version.hpp>

namespace arb {

class mechanism;
using mechanism_ptr = std::unique_ptr<mechanism>;

struct ion_state_view {
    arb_value_type* current_density;
    arb_value_type* reversal_potential;
    arb_value_type* internal_concentration;
    arb_value_type* external_concentration;
    arb_value_type* ionic_charge;
};

class mechanism {
public:
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using size_type  = arb_size_type;

    mechanism(const arb_mechanism_type& m,
              const arb_mechanism_interface& i): mech_{m}, iface_{i}, ppack_{} {
        if (mech_.abi_version != ARB_MECH_ABI_VERSION) throw unsupported_abi_error{mech_.abi_version};
        state_prof_id   = profile::profiler_region_id("advance:integrate:state:"+internal_name());
        current_prof_id = profile::profiler_region_id("advance:integrate:current:"+internal_name());
    }
    mechanism() = default;
    mechanism(const mechanism&) = delete;
    ~mechanism() = default;

    // Return fingerprint of mechanism dynamics source description for validation/replication.
    const mechanism_fingerprint fingerprint() const { return mech_.fingerprint; };

    // Name as given in mechanism source.
    std::string internal_name() const { return mech_.name; }

    // Density or point mechanism?
    arb_mechanism_kind kind() const { return mech_.kind; };

    // Minimum expected alignment of allocated vectors and shared state data.
    unsigned data_alignment() const { return iface_.alignment; }

    // Make a new object of the mechanism type, but does not copy any state, so
    // the result must be instantiated.
    mechanism_ptr clone() const { return std::make_unique<mechanism>(mech_, iface_); }

    // Forward to interface methods
    void initialize()     { iface_.init_mechanism(&ppack_); }
    void update_current() { prof_enter(current_prof_id); iface_.compute_currents(&ppack_); prof_exit(); }
    void update_state()   { prof_enter(state_prof_id);   iface_.advance_state(&ppack_);    prof_exit(); }
    void update_ions()    { iface_.write_ions(&ppack_); }
    void post_event()     { iface_.post_event(&ppack_); }
    void deliver_events(arb_deliverable_event_stream& stream) { iface_.apply_events(&ppack_, &stream); }

    // Per-cell group identifier for an instantiated mechanism.
    unsigned mechanism_id() const { return ppack_.mechanism_id; }

    arb_mechanism_type  mech_;
    arb_mechanism_interface iface_;
    arb_mechanism_ppack ppack_;

private:
#ifdef ARB_PROFILE_ENABLED
    void prof_enter(profile::region_id_type id) {
        profile::profiler_enter(id);
    }
    void prof_exit() {
        profile::profiler_leave();
    }
#else
    void prof_enter(profile::region_id_type) {}
    void prof_exit() {}
#endif
    profile::region_id_type state_prof_id;
    profile::region_id_type current_prof_id;
};

struct mechanism_layout {
    // Maps in-instance index to CV index.
    std::vector<arb_index_type> cv;

    // Maps in-instance index to peer CV index (only for gap-junctions).
    std::vector<arb_index_type> peer_cv;

    // Maps in-instance index to compartment contribution.
    std::vector<arb_value_type> weight;

    // Number of logical point processes at in-instance index;
    // if empty, point processes are not coalesced and all multipliers are 1.
    std::vector<arb_index_type> multiplicity;

    std::vector<arb_size_type> gid;
    std::vector<arb_size_type> idx;
};

struct mechanism_overrides {
    // Global scalar parameters (any value down-conversion to arb_value_type is the
    // responsibility of the mechanism).
    std::unordered_map<std::string, double> globals;

    // Ion renaming: keys are ion dependency names as
    // reported by the mechanism info.
    std::unordered_map<std::string, std::string> ion_rebind;
};

} // namespace arb
