#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <arbor/arbexcept.hpp>
#include <arbor/export.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism_abi.h>
#include <arbor/mechinfo.hpp>

namespace arb {

struct mechanism;
using mechanism_ptr = std::unique_ptr<mechanism>;

struct ion_state_view {
    arb_value_type* current_density;
    arb_value_type* reversal_potential;
    arb_value_type* internal_concentration;
    arb_value_type* external_concentration;
    arb_value_type* ionic_charge;
};

struct ARB_ARBOR_API mechanism {
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using size_type  = arb_size_type;

    mechanism(const arb_mechanism_type& m,
              const arb_mechanism_interface& i):
        mech_{m}, iface_{i}, ppack_{} {
        if (mech_.abi_version != ARB_MECH_ABI_VERSION) throw unsupported_abi_error{mech_.abi_version};
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

    void set_dt(arb_value_type dt) { ppack_.dt = dt; }

    // Forward to interface methods
    void initialize();
    void update_current();
    void update_state();
    void update_ions();
    void post_event();
    void deliver_events(arb_deliverable_event_stream& stream);

    // Per-cell group identifier for an instantiated mechanism.
    unsigned mechanism_id() const { return ppack_.mechanism_id; }

    arb_mechanism_type  mech_;
    arb_mechanism_interface iface_;
    arb_mechanism_ppack ppack_;
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
