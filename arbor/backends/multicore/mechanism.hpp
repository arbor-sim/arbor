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
#include <arbor/mechanism_ppack.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

// Parameter pack extended for multicore.
struct mechanism_ppack: arb::mechanism_ppack {
    constraint_partition index_constraints_;    // Per-mechanism index and weight data, excepting ion indices.
};

// Base class for all generated mechanisms for multicore back-end.
class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;

protected:
    virtual unsigned simd_width() const { return 1; }
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.
};

} // namespace multicore
} // namespace arb
