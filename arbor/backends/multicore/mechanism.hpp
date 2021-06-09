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

#include "fvm.hpp"

namespace arb {
namespace multicore {

struct backend;

// Base class for all generated mechanisms for multicore back-end.
class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    using concrete_mechanism<arb::multicore::backend>::concrete_mechanism;
    mechanism() = default;

    mechanism_ptr clone() const override { return std::make_unique<mechanism>(mech_, iface_); }
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
};

} // namespace multicore
} // namespace arb
