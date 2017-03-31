#include <string>
#include <vector>

#ifdef NMC_HAVE_GPU
    #include <cuda_runtime.h>
#endif

#include "time_meter.hpp"
#include <communication/global_policy.hpp>
#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

std::string time_meter::name() {
    return "time";
}

void time_meter::take_reading() {
    // Wait for execution on this global domain to finish before recording the
    // time stamp. For now this means waiting for all work to finish executing
    // on the GPU (if GPU support is enabled)
#ifdef NMC_HAVE_GPU
    cudaDeviceSynchronize();
#endif

    // Record the time stamp
    readings_.push_back(timer_type::tic());

    // Enforce a global barrier after taking the time stamp
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
