#include <string>
#include <vector>

#include <arbor/profile/meter.hpp>

#include "hardware/memory.hpp"
#include "memory_meter.hpp"

namespace arb {
namespace profile {

class memory_meter: public meter {
protected:
    std::vector<hw::memory_size_type> readings_;

public:
    std::string name() override {
        return "memory";
    }

    std::string units() override {
        return "B";
    }

    void take_reading() override {
        readings_.push_back(hw::allocated_memory());
    }

    std::vector<double> measurements() override {
        std::vector<double> diffs;

        for (auto i=1ul; i<readings_.size(); ++i) {
            diffs.push_back(readings_[i]-readings_[i-1]);
        }

        return diffs;
    }
};

meter_ptr make_memory_meter() {
    if (hw::allocated_memory()==-1) {
        return nullptr;
    }
    return meter_ptr(new memory_meter());
}


// The gpu memory meter specializes the reading and name methods of the basic
// memory_meter.

class gpu_memory_meter: public memory_meter {
public:
    std::string name() override {
        return "memory-gpu";
    }

    void take_reading() override {
        readings_.push_back(hw::gpu_allocated_memory());
    }
};

meter_ptr make_gpu_memory_meter() {
    if (hw::gpu_allocated_memory()==-1) {
        return nullptr;
    }
    return meter_ptr(new gpu_memory_meter());
}

} // namespace profile
} // namespace arb
