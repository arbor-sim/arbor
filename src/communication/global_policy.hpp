#pragma once

#include <string>

namespace arb {  namespace communication {
    enum class global_policy_kind {serial, mpi, dryrun};
}}

namespace std {
    inline
    std::string to_string(arb::communication::global_policy_kind k) {
        using namespace arb::communication;
        if (k == global_policy_kind::mpi) {
            return "MPI";
        }
        if (k == global_policy_kind::dryrun) {
            return "dryrun";
        }
        return "serial";
    }
}

#if defined(NMC_HAVE_MPI)
    #include "mpi_global_policy.hpp"
#elif defined(NMC_HAVE_DRYRUN)
    #include "dryrun_global_policy.hpp"
#else
    #include "serial_global_policy.hpp"
#endif

namespace arb {
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
} // namespace arb
