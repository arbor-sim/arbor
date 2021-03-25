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
    using concrete_mechanism<arb::multicore::backend>::concrete_mechanism;
    mechanism() = default;

    mechanism_ptr clone() const override { return std::make_unique<mechanism>(&mech_); }
    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;

protected:
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.
    std::vector<arb_value_type*> parameter_ptrs_;
    std::vector<arb_value_type*> state_var_ptrs_;
    std::vector<arb_ion_state>   ion_ptrs_;
    constraint_partition constraints_;
};

} // namespace multicore
} // namespace arb
