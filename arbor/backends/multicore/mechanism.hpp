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
#include <arbor/mechanism_abi.h>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

// Base class for all generated mechanisms for multicore back-end.
class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    // const mechanism_fingerprint& fingerprint() const override { return ppack_->fingerprint; }
    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;

protected:
    virtual unsigned simd_width() const { return 1; }
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.

    void advance_state()                               override { mech_->advance_state(ppack_.get()); }
    void compute_currents()                            override { mech_->compute_currents(ppack_.get()); }
    void apply_events(deliverable_event_stream::state) override { mech_->apply_events(ppack_.get()); }
    void init()                                        override { mech_->init_mechanism(ppack_.get()); }
    void write_ions()                                  override { mech_->write_ions(ppack_.get()); }

   mechanism_field_table         field_table()          override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }
   mechanism_field_default_table field_default_table()  override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }
   mechanism_global_table        global_table()         override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }
   mechanism_state_table         state_table()          override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }
   mechanism_ion_state_table     ion_state_table()      override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }
   mechanism_ion_index_table     ion_index_table()      override { throw std::runtime_error(__FUNCTION__ + std::string{" not implemented"}); return {}; }

    std::unique_ptr<arb_mechanism_ppack> ppack_ = std::make_unique<arb_mechanism_ppack>();
    std::unique_ptr<arb_mechanism_type>  mech_  = std::make_unique<arb_mechanism_type>();
};

} // namespace multicore
} // namespace arb
