#pragma once

#if defined(NMC_HAVE_MPI)
    #include "communication/mpi_global_policy.hpp"
#elif defined(NMC_HAVE_DRYRUN)
    #include "communication/dryrun_global_policy.hpp"
#else
    #include "communication/serial_global_policy.hpp"
#endif

namespace nest {
namespace mc {
namespace communication {

template <typename Policy>
struct policy_guard {
    using policy_type = Policy;

    policy_guard(int argc, char**& argv) {
        policy_type::setup(argc, argv);
    }

    policy_guard() = delete;
    policy_guard(policy_guard&&) = delete;
    policy_guard(const policy_guard&) = delete;
    policy_guard& operator=(policy_guard&&) = delete;
    policy_guard& operator=(const policy_guard&) = delete;

    ~policy_guard() {
        Policy::teardown();
    }
};

using global_policy_guard = policy_guard<global_policy>;

} // namespace communication
} // namespace mc
} // namespace nest
