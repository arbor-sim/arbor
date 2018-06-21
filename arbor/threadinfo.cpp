#include <string>

#include <arbor/threadinfo.hpp>

#include "threading/threading.hpp"

namespace arb {

int thread_count() {
    return threading::num_threads();
}

std::string thread_implementation() {
    return threading::description();
}

} // namespace arb
