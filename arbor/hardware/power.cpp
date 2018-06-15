#include <fstream>

#include "power.hpp"

// Currently only supporting Cray PM counters.

#define CRAY_PM_COUNTER_ENERGY "/sys/cray/pm_counters/energy"

namespace arb {
namespace hw {

bool has_energy_measurement() {
    return static_cast<bool>(std::ifstream(CRAY_PM_COUNTER_ENERGY));
}

energy_size_type energy() {
    energy_size_type result = energy_size_type(-1);

    std::ifstream fid(CRAY_PM_COUNTER_ENERGY);
    if (fid) {
        fid >> result;
    }

    return result;
}

} // namespace hw
} // namespace arb

