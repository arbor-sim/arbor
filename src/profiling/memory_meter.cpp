#include <string>
#include <vector>

#include <util/config.hpp>

#include "memory_meter.hpp"

namespace nest {
namespace mc {
namespace util {

//
//  memory_meter
//

class memory_meter: public meter {
protected:
    std::vector<memory_size_type> readings_;

public:
    std::string name() override {
        return "memory";
    }

    std::string units() override {
        return "B";
    }

    void take_reading() override {
        readings_.push_back(allocated_memory());
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
    if (not config::has_memory_measurement) {
        return nullptr;
    }
    return meter_ptr(new memory_meter());
}

//
//  gpu_memory_meter
//

// The gpu memory meter specializes the reading and name methods of the basic
// memory_meter.
class gpu_memory_meter: public memory_meter {
public:
    std::string name() override {
        return "memory-gpu";
    }

    void take_reading() override {
        readings_.push_back(gpu_allocated_memory());
    }
};

meter_ptr make_gpu_memory_meter() {
    if (not config::has_cuda) {
        return nullptr;
    }
    return meter_ptr(new gpu_memory_meter());
}

} // namespace util
} // namespace mc
} // namespace nest
