#pragma once

#include <string>
#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

// A measurement from a meter has the following:
//  * name
//    * e.g. walltime or allocated-memory
//  * units
//    * use SI units
//    * e.g. s or MiB
//  * measurements
//    * a vector with one entry for each checkpoint
//    * each entry is a std::vector<double> of measurements gathered across
//      domains at one checkpoint.
//
struct measurement {
    std::string name;
    std::string units;
    std::vector<std::vector<double>> measurements;
};

// Converts a measurement to a json type for serialization to file.
// See src/profiling/meters.md for more information about the json formating.
nlohmann::json to_json(const measurement& m);

// A meter can be used to take a measurement of resource consumption, for
// example wall time, memory or energy consumption.
// Each specialization must:
//  1) Record the resource consumption on calling meter::take_reading.
//      * How and which information is recorded is implementation dependent.
//  2) Return a std::vector containing the measurements that are derived
//     from the information recorded on calls to meter::take_reading.
//      * The return value is a vector of measurements, because a meter
//        may derive multiple measurements from the recorded checkpoint
//        information.
class meter {
public:
    meter() = default;

    // Provide a human readable name for the meter
    virtual std::string name() = 0;

    // Take a reading/measurement of the resource
    virtual void take_reading() = 0;

    // Return a summary of the recordings.
    // May perform expensive operations to process and analyse the readings.
    // Full output is expected only on the root domain, i.e. when
    // global_policy::id()==0
    virtual std::vector<measurement> measurements() = 0;

    virtual ~meter() = default;
};

} // namespace util
} // namespace mc
} // namespace nest
