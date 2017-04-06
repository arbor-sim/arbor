#pragma once

#include <string>
#include <json/json.hpp>
#include <communication/global_policy.hpp>

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

namespace impl {
    // Helper function for collating measurements across the global domain.
    // The difference functor takes two successive readings and returns a
    // double precision difference between the readings in the correct units.
    template <typename T, typename F>
    measurement collate(
        const std::vector<T>& readings,
        std::string name,
        std::string units,
        F&& difference)
    {
        using gcom = communication::global_policy;

        // Calculate the local change in the given quantity.
        std::vector<double> diffs;
        diffs.push_back(0);
        for (auto i=1u; i<readings.size(); ++i) {
            diffs.push_back(difference(readings[i-1], readings[i]));
        }

        // Assert that the same number of readings were taken on every domain.
        const auto num_readings = diffs.size();
        if (gcom::min(num_readings)!=gcom::max(num_readings)) {
            throw std::out_of_range(
                "the number of checkpoints in the \""+name+"\" meter do not match across domains");
        }

        // Gather across all of the domains onto the root domain.
        // Note: results are only valid on the root domain on completion.
        measurement results;
        results.name = std::move(name);
        results.units = std::move(units);
        for (auto m: diffs) {
            results.measurements.push_back(gcom::gather(m, 0));
        }

        return results;
    }
} // namespace impl

} // namespace util
} // namespace mc
} // namespace nest
