#include <arbor/context.hpp>

#include "hardware/node_info.hpp"
#include "threading/thread_info.hpp"
#include "threading/threading.hpp"

namespace arb {

local_resources get_local_resources() {
    auto avail_threads = threading::num_threads_init();
    auto avail_gpus = arb::hw::node_gpus();

    return local_resources(avail_threads, avail_gpus);
}

} // namespace arb
