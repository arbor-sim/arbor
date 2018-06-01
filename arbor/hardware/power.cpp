#include <fstream>

#include "power.hpp"

namespace arb {
namespace hw {

// Currently only supporting Cray PM counters.

energy_size_type energy() {
    energy_size_type result = energy_size_type(-1);

    std::ifstream fid("/sys/cray/pm_counters/energy");
    if (fid) {
        fid >> result;
    }

    return result;
}

} // namespace hw
} // namespace arb
