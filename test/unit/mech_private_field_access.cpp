#include <arbor/version.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/mechanism.hpp"
#include "util/maputil.hpp"

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#include "backends/gpu/mechanism.hpp"
#include "memory/cuda_wrappers.hpp"
#endif

#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;
using field_table_type = std::vector<std::pair<const char*, fvm_value_type**>>;

// Multicore mechanisms:

ACCESS_BIND(field_table_type (multicore::mechanism::*)(), multicore_field_table_ptr, &multicore::mechanism::field_table)

std::vector<fvm_value_type> mechanism_field(multicore::mechanism* m, const std::string& key) {
    auto opt_ptr = util::value_by_key((m->*multicore_field_table_ptr)(), key);
    if (!opt_ptr) throw std::logic_error("internal error: no such field in mechanism");

    const fvm_value_type* field_data = *opt_ptr.value();
    return std::vector<fvm_value_type>(field_data, field_data+m->size());
}

// GPU mechanisms:

#ifdef ARB_GPU_ENABLED
ACCESS_BIND(field_table_type (gpu::mechanism::*)(), gpu_field_table_ptr, &gpu::mechanism::field_table)

std::vector<fvm_value_type> mechanism_field(gpu::mechanism* m, const std::string& key) {
    auto opt_ptr = util::value_by_key((m->*gpu_field_table_ptr)(), key);
    if (!opt_ptr) throw std::logic_error("internal error: no such field in mechanism");

    const fvm_value_type* field_data = *opt_ptr.value();
    std::vector<fvm_value_type> values(m->size());

    memory::cuda_memcpy_d2h(values.data(), field_data, sizeof(fvm_value_type)*m->size());
    return values;
}
#endif

// Generic access:

std::vector<fvm_value_type> mechanism_field(mechanism* m, const std::string& key) {
    if (auto p = dynamic_cast<multicore::mechanism*>(m)) {
        return mechanism_field(p, key);
    }

#ifdef ARB_GPU_ENABLED
    if (auto p = dynamic_cast<gpu::mechanism*>(m)) {
        return mechanism_field(p, key);
    }
#endif

    throw std::logic_error("internal error: mechanism instantiated on unknown backend");
}

