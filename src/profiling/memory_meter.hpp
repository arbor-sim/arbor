#pragma once

#include <string>
#include <vector>

#include <util/memory.hpp>

#include "meter.hpp"

namespace nest {
namespace mc {
namespace util {

class memory_meter : public meter {
    std::vector<memory_size_type> readings_;

public:
    std::string name() override;
    void take_reading() override;
    virtual std::vector<measurement> measurements() override;
};

} // namespace util
} // namespace mc
} // namespace nest
