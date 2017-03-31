#pragma once

#include <string>
#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

// A meter can be used to take a measurement of resource consumption, for
// example wall time, memory or energy consumption.
// Each specialization must:
//  1) Record the resource consumption on calling meter::take_reading.
//      * how and which information is recorded is implementation dependent
//  2) Return a json record that lists the measured information from a
//     call to meter::as_json.
//      * the format of the output is a json Array of json Objects
//      * each Object corresponds to a derived measurement:
//        * "name": a string describing the measurement
//        * "units": a string with SI units for measurements
//        * "measurements": a json Array of measurements, with one
//          entry per checkpoint (corresponding to a call to
//          meter::take_reading)
//          * each measurement is itself a numeric array, with one
//            recording for each domain in the global communicator
//
//  For example, the output of a meter for measuring wall time where 5 readings
//  were taken on 4 MPI ranks could be represented as follows:
//
//   [{
//     "name": "walltime",
//     "units": "s",
//     "measurements": [
//       [ 0, 0, 0, 0, ],
//       [ 0.001265837, 0.001344004, 0.001299362, 0.001195762, ],
//       [ 0.014114013, 0.015045662, 0.015071675, 0.014209514, ],
//       [ 1.491986631, 1.491121134, 1.490957219, 1.492064233, ],
//       [ 0.00565307, 0.004375347, 0.002228206, 0.002483978, ]
//     ]
//   }]
class meter {
public:
    meter() = default;

    // Provide a human readable name for the meter
    virtual std::string name() = 0;

    // Take a reading/measurement of the resource
    virtual void take_reading() = 0;

    // Return a summary of the recordings in json format (see documentation above).
    // May perform expensive operations to process and analyse the readings.
    // Full output is expected only on the root domain, i.e. when
    // global_policy::id()==0
    virtual nlohmann::json as_json() = 0;

    virtual ~meter() = default;
};

} // namespace util
} // namespace mc
} // namespace nest
