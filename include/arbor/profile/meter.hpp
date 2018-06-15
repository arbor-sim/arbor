#pragma once

#include <memory>
#include <string>
#include <vector>

namespace arb {
namespace profile {

// A meter can be used to take a measurement of resource consumption, for
// example wall time, memory or energy consumption.
// Each specialization must:
//  1) Record the resource consumption on calling meter::take_reading.
//      * How and which information is recorded is implementation dependent.
//  2) Provide the name of the resource being measured via name()
//      e.g. : energy
//  3) Provide the units of the resource being measured via units()
//      e.g. : J
//  4) Return the resources consumed between each pair of readings as a
//     std::vector<double> from measurements(). So, for n readings, there will
//     be n-1 differences.
class meter {
public:
    meter() = default;

    // Provide a human readable name for the meter
    virtual std::string name() = 0;

    // Take a reading/measurement of the resource
    virtual void take_reading() = 0;

    // The units of the values returned in from the measurements method.
    virtual std::string units() = 0;

    virtual std::vector<double> measurements() = 0;

    virtual ~meter() = default;
};

using meter_ptr = std::unique_ptr<meter>;

} // namespace profile
} // namespace arb
