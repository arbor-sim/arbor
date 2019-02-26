#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arbor/context.hpp>
#include <arbor/profile/meter.hpp>
#include <arbor/profile/timer.hpp>

namespace arb {
namespace profile {

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
    measurement(std::string, std::string, const std::vector<double>&, const context&);
};

class meter_manager {
private:
    bool started_ = false;

    tick_type start_time_;
    std::vector<double> times_;

    std::vector<std::unique_ptr<meter>> meters_;
    std::vector<std::string> checkpoint_names_;

public:
    meter_manager();
    void start(const context& ctx);
    void checkpoint(std::string name, const context& ctx);

    const std::vector<std::unique_ptr<meter>>& meters() const;
    const std::vector<std::string>& checkpoint_names() const;
    const std::vector<double>& times() const;

};

// Simple type for gathering distributed meter information
struct meter_report {
    std::vector<std::string> checkpoints;
    unsigned num_domains;
    unsigned num_hosts;
    std::vector<measurement> meters;
    std::vector<std::string> hosts;
};

meter_report make_meter_report(const meter_manager& manager, const context& ctx);
std::ostream& operator<<(std::ostream& o, const meter_report& report);

} // namespace profile
} // namespace arb
