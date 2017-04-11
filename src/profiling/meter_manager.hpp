#pragma once

#include <memory>
#include <vector>

#include <json/json.hpp>

#include "meter.hpp"
#include "profiler.hpp"

namespace nest {
namespace mc {
namespace util {

// A measurement has the following:
//  * name
//    * e.g. walltime or allocated-memory
//  * units
//    * use SI units
//    * e.g. s or MiB
//  * measurements
//    * a vector with one entry for each checkpoint
//    * each entry is a std::vector<double> of measurements gathered across
//      domains at one checkpoint.
struct measurement {
    std::string name;
    std::string units;
    std::vector<std::vector<double>> measurements;
    measurement(std::string, std::string, const std::vector<double>&);
};

// Converts a measurement to a json type for serialization to file.
// See src/profiling/meters.md for more information about the json formating.
nlohmann::json to_json(const measurement& m);

class meter_manager {
private:
    bool started_ = false;

    timer_type::time_point start_time_;
    std::vector<double> times_;

    std::vector<std::unique_ptr<meter>> meters_;
    std::vector<std::string> checkpoint_names_;

public:
    meter_manager();
    void start();
    void checkpoint(std::string name);

    const std::vector<std::unique_ptr<meter>>& meters() const;
    const std::vector<std::string>& checkpoint_names() const;
    const std::vector<double>& times() const;
};

nlohmann::json to_json(const meter_manager&);
void save_to_file(const meter_manager& manager, const std::string& name);

} // namespace util
} // namespace mc
} // namespace nest
