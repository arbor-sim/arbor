#include <fstream>
#include <string>

#include <json/src/json.hpp>

#include "trace_sampler.hpp"

namespace nest {
namespace mc {

sample_to_trace::sample_to_trace(cell_member_type probe_id,
                const std::string &name,
                const std::string &units,
                float dt,
                float t_start):
    trace_{name, units, probe_id.gid, probe_id.index},
    sample_dt_(dt),
    t_next_sample_(t_start)
{}

sample_to_trace::sample_to_trace(cell_member_type probe_id,
                probeKind kind,
                segment_location loc,
                float dt,
                float t_start):
    sample_to_trace(probe_id, "", "", dt, t_start)
{
    std::string name = "";
    std::string units = "";

    switch (kind) {
    case probeKind::membrane_voltage:
        name = "v";
        units = "mV";
        break;
    case probeKind::membrane_current:
        name = "i";
        units = "mA/cm^2";
        break;
    default: ;
    }

    trace_.name = name + (loc.segment? "dend": "soma");
    trace_.units = units;
}

void sample_to_trace::write_trace(const std::string& prefix) const {
    auto path = prefix + std::to_string(trace_.cell_gid) +
                "." + std::to_string(trace_.probe_index) + ".json";

    nlohmann::json jrep;
    jrep["name"] = trace_.name;
    jrep["units"] = trace_.units;
    jrep["cell"] = trace_.cell_gid;
    jrep["probe"] = trace_.probe_index;

    auto& jt = jrep["data"]["time"];
    auto& jy = jrep["data"][trace_.name];

    for (const auto& sample: trace_.samples) {
        jt.push_back(sample.time);
        jy.push_back(sample.value);
    }
    std::ofstream file(path);
    file << std::setw(1) << jrep << std::endl;
}

} // namespace mc
} // namespace nest
