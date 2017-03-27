#pragma once

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
}

// This call may perform expensive operations to process and analyse the readings
nlohmann::json time_meter::as_json() {
    using gcom = communication::global_policy;
    const bool is_root = gcom::id()==0;

    std::vector<double> times;
    times.push_back(0);

    for (auto i=1u; i<readings_.size(); ++i) {
        double t = timer_type::difference(readings_[i-1], readings_[i]);
        times.push_back(t);
    }

    auto num_readings = times.size();

    auto num_domains = gcom::size();
    if (gcom::min(times.size())!=gcom::max(times.size())) {
        throw std::out_of_range(
            "the number of checkpoints in the \"time\" meter do not match across domains");
    }

    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> mean;
    for (auto t: times) {
        auto values = gcom::gather(t, 0);
        if (is_root) {
            auto minmax = std::minmax_element(values.begin(), values.end());
            min.push_back(*(minmax.first));
            max.push_back(*(minmax.second));
            mean.push_back( algorithms::sum(values)/values.size() );
        }
    }

    if (is_root) {
        return {
            {"name", "time"},
            {"values", {"min", min}, {"max", max}, {"mean", mean}}
        };
    }

    return {};
}

} // namespace util
} // namespace mc
} // namespace nest
