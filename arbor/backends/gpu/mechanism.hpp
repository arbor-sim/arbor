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

#include "backends/gpu/fvm.hpp"
#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

// Base class for all generated mechanisms for gpu back-end.

class mechanism: public arb::concrete_mechanism<arb::gpu::backend> {
public:
    mechanism()                 = default;
    mechanism(const mechanism&) = delete;
    using concrete_mechanism<arb::gpu::backend>::concrete_mechanism;
    virtual ~mechanism() = default;

   virtual mechanism_ptr clone() const override { return std::make_unique<mechanism>(&mech_); }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;
};
} // namespace gpu
} // namespace arb
