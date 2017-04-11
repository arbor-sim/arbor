#pragma once

#include <string>
#include <vector>

#include <util/memory.hpp>

#include "meter.hpp"

namespace nest {
namespace mc {
namespace util {

class memory_meter: public meter {
protected:
    std::vector<memory_size_type> readings_;

public:
    std::string name() override;
    std::string units() override;
    void take_reading() override;
    std::vector<double> measurements() override;
};

meter_ptr make_memory_meter();

// The gpu memory meter specializes the reading and name methods of the basic
// memory_meter.
class gpu_memory_meter: public memory_meter {
public:
    std::string name() override;
    void take_reading() override;
};

meter_ptr make_gpu_memory_meter();

} // namespace util
} // namespace mc
} // namespace nest
