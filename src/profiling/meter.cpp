#include "meter.hpp"

namespace nest {
namespace mc {
namespace util {

nlohmann::json to_json(const measurement& mnt) {
    nlohmann::json measurements;
    for (const auto& m: mnt.measurements) {
        measurements.push_back(m);
    }

    return {
        {"name", mnt.name},
        {"units", mnt.units},
        {"measurements", measurements}
    };
}

} // namespace util
} // namespace mc
} // namespace nest

