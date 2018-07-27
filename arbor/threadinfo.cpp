#include <string>

#include <arbor/threadinfo.hpp>

#include "threading/threading.hpp"

namespace arb {

std::string thread_implementation() {
    return threading::description();
}

} // namespace arb
