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
    using concrete_mechanism<arb::gpu::backend>::concrete_mechanism;
    // mechanism() = default;

    mechanism_ptr clone() const override { return std::make_unique<mechanism>(mech_, iface_); }
    void instantiate(fvm_size_type id, backend::shared_state& shared, const mechanism_overrides&, const mechanism_layout&) override;
    void initialize() override;
    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override;
    fvm_value_type* field_data(const std::string& state_var) override;

    virtual mechanism_field_table field_table() override;
    virtual mechanism_global_table global_table() override;
    virtual mechanism_state_table state_table() override;
    virtual mechanism_ion_table ion_table() override;

protected:
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.

    memory::device_vector<arb_value_type*> parameters_d_;
    memory::device_vector<arb_value_type*> state_vars_d_;
    memory::device_vector<arb_ion_state>   ion_states_d_;

    // Mirrors for _XYZ_tables
    memory::host_vector<arb_value_type>  globals_h_;
    memory::host_vector<arb_value_type*> parameters_h_;
    memory::host_vector<arb_value_type*> state_vars_h_;
    memory::host_vector<arb_ion_state>   ion_states_h_;
};
} // namespace gpu
} // namespace arb
