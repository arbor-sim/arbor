#include <string>
#include <vector>
#include <json/json.hpp>

#include "time_meter.hpp"
#include <communication/global_policy.hpp>

namespace nest {
namespace mc {
namespace util {

std::string time_meter::name() {
    return "time";
}

void time_meter::take_reading() {
    readings_.push_back(timer_type::tic());
    communication::global_policy::barrier();
}

// This call may perform expensive operations to process and analyse the readings
nlohmann::json time_meter::as_json() {
    using nlohmann::json;
    using gcom = communication::global_policy;
    const bool is_root = gcom::id()==0;

    std::vector<double> times;
    times.push_back(0);

    for (auto i=1u; i<readings_.size(); ++i) {
        double t = timer_type::difference(readings_[i-1], readings_[i]);
        times.push_back(t);
    }

    auto num_readings = times.size();

    //auto num_domains = gcom::size();
    if (gcom::min(num_readings)!=gcom::max(num_readings)) {
        throw std::out_of_range(
            "the number of checkpoints in the \"time\" meter do not match across domains");
    }

    json results;
    //std::vector<std::vector<double>> results;
    for (auto t: times) {
        results.push_back(gcom::gather(t, 0));
    }

    if (is_root) {
        return {
            {"name", "walltime"},
            {"units", "s"},
            {"measurements", results}
        };
    }

    return {};
}

} // namespace util
} // namespace mc
} // namespace nest
