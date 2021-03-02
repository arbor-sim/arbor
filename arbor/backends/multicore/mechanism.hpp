#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

// Base class for all generated mechanisms for multicore back-end.

class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;

    // Simulation interfaces:
    virtual void initialize()     override { do_initialize(ppack_ptr());     };
    virtual void update_state()   override { do_update_state(ppack_ptr());   };
    virtual void update_current() override { do_update_current(ppack_ptr()); };
    virtual void deliver_events() override { do_deliver_events(ppack_ptr()); };
    virtual void post_event()     override { do_post_event(ppack_ptr());     };
    virtual void update_ions()    override { do_update_ions(ppack_ptr());    };


protected:
    virtual unsigned simd_width() const { return 1; }
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.
    constraint_partition index_constraints_;    // Per-mechanism index and weight data, excepting ion indices.

    // Simulation interfaces:
    virtual void do_initialize(mechanism_ppack* pp) {};
    virtual void do_update_state(mechanism_ppack* pp) {};
    virtual void do_update_current(mechanism_ppack* pp) {};
    virtual void do_deliver_events(mechanism_ppack* pp) {};
    virtual void do_post_event(mechanism_ppack* pp) {};
    virtual void do_update_ions(mechanism_ppack* pp) {};
};

} // namespace multicore
} // namespace arb
