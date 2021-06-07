#include <arbor/version.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/mechanism.hpp"
#include "util/maputil.hpp"

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#include "backends/gpu/mechanism.hpp"
#include "memory/gpu_wrappers.hpp"
#endif

#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;

// Multicore mechanisms:

ACCESS_BIND(mechanism_field_table (concrete_mechanism<multicore::backend>::*)(), multicore_field_table_ptr, &concrete_mechanism<multicore::backend>::field_table);
ACCESS_BIND(arb_mechanism_type  mechanism::*, mech_type_ptr,  &mechanism::mech_);
ACCESS_BIND(arb_mechanism_ppack mechanism::*, mech_ppack_ptr, &mechanism::ppack_);

std::vector<fvm_value_type> mechanism_field(multicore::mechanism* m, const std::string& key) {
    auto opt_ptr = util::value_by_key((m->*multicore_field_table_ptr)(), key);
    if (!opt_ptr) throw std::logic_error("internal error: no such field in mechanism");

    const fvm_value_type* field_data = opt_ptr.value().first;
    auto ppack = m->*mech_ppack_ptr;
    return std::vector<fvm_value_type>(field_data, field_data+ppack.width);
}

// GPU mechanisms:

#ifdef ARB_GPU_ENABLED
ACCESS_BIND(mechanism_field_table (concrete_mechanism<gpu::backend>::*)(), gpu_field_table_ptr, &concrete_mechanism<gpu::backend>::field_table)

std::vector<fvm_value_type> mechanism_field(gpu::mechanism* m, const std::string& key) {
    auto opt_ptr = util::value_by_key((m->*gpu_field_table_ptr)(), key);
    if (!opt_ptr) throw std::logic_error("internal error: no such field in mechanism");

    const fvm_value_type* field_data = opt_ptr.value().first;
    std::vector<fvm_value_type> values(m->size());

    memory::gpu_memcpy_d2h(values.data(), field_data, sizeof(fvm_value_type)*m->size());
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

