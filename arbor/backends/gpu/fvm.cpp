#include <string>

#include <arbor/mechanism.hpp>
#include "fvm.hpp"
#include "mechanism.hpp"

// Provides implementation of backend::mechanism_field_data.

namespace arb {
namespace gpu {

fvm_value_type* backend::mechanism_field_data(arb::mechanism* mptr, const std::string& field) {
    arb::gpu::mechanism* m = dynamic_cast<arb::gpu::mechanism*>(mptr);
    return m? m->field_data(field): nullptr;
}

void kernel::multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n) {

void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n) {
    kernel::multiply_in_place(s, p, n);
}


} // namespace gpu
} // namespace arb
