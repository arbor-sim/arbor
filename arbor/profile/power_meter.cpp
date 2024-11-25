#include <string>
#include <vector>

#include <arbor/profile/meter.hpp>

#include "hardware/power.hpp"

namespace arb {
namespace profile {

class power_meter: public meter {
    std::vector<hw::energy_size_type> readings_;

public:
    std::string name() override {
        return "energy";
    }

    std::string units() override {
        return "J";
    }

    std::vector<double> measurements() override {
        std::vector<double> diffs;

        for (auto i=1ul; i<readings_.size(); ++i) {
            diffs.push_back(readings_[i]-readings_[i-1]);
        }

        return diffs;
    }

    void take_reading() override {
        readings_.push_back(hw::energy());
    }
};

meter_ptr make_power_meter() {
    if (!arb::hw::has_energy_measurement()) {
        return nullptr;
    }
    return meter_ptr(new power_meter());
}

} // namespace profile
} // namespace arb
