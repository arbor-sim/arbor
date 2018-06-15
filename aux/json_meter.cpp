#include <arbor/profiling/meter_manager.hpp>
#include <nlohmann/json.hpp>

namespace aux {

static nlohmann::json to_json(const arb::profile::measurement& mnt) {
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

nlohmann::json to_json(const arb::profile::meter_report&) {
    return {
        {"checkpoints", report.checkpoints},
        {"num_domains", report.num_domains},
        {"meters", util::transform_view(report.meters, [](measurement const& m){return to_json(m);})},
        {"hosts", report.hosts},
    };
}

}
