#include <string>

#include <arbor/mechanism.hpp>
#include "fvm.hpp"
#include "mechanism.hpp"

// Provides implementation of backend::mechanism_field_data.

namespace arb {
namespace multicore {

fvm_value_type* backend::mechanism_field_data(arb::mechanism* mptr, const std::string& field) {
    arb::multicore::mechanism* m = dynamic_cast<arb::multicore::mechanism*>(mptr);
    return m? m->field_data(field): nullptr;
}

} // namespace multicore
} // namespace arb
