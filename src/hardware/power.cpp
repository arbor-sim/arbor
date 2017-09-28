#include <fstream>

#include "power.hpp"

namespace arb {
namespace hw {

#ifdef NMC_HAVE_CRAY

energy_size_type energy() {
    energy_size_type result = energy_size_type(-1);

    std::ifstream fid("/sys/cray/pm_counters/energy");
    if (fid) {
        fid >> result;
    }

    return result;
}

#else

energy_size_type energy() {
    return -1;
}

#endif

} // namespace hw
} // namespace arb
