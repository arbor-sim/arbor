#include <arbor/domain_decomposition.hpp>
#include <arbor/threadinfo.hpp>

#include "hardware/node_info.hpp"

namespace arb {

domain_info local_domain_info() {
    domain_info info;
    info.num_threads = arb::num_threads();
    info.num_gpus = arb::hw::node_gpus();

    return info;
}

} // namespace arb
