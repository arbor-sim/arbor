#include <fstream>
#include <string>

#include <json/json.hpp>

#include <common_types.hpp>
#include <util/strprintf.hpp>

#include "trace.hpp"

using namespace arb;

static std::string to_string(cell_member_type m) {
    return std::to_string(m.gid)+"."+std::to_string(m.index);
}

void write_trace_csv(const sample_trace& trace, const std::string& prefix) {
    auto path = prefix + to_string(trace.probe_id) + "_" + trace.name + ".csv";

    std::ofstream file(path);
    file << "# cell: " << trace.probe_id.gid << "\n";
    file << "# probe: " << trace.probe_id.index << "\n";
    file << "time_ms, " << trace.name << "_" << trace.units << "\n";

    for (const auto& sample: trace.samples) {
        file << util::strprintf("% 20.15f, % 20.15f\n", sample.t, sample.v);
    }
}

void write_trace_json(const sample_trace& trace, const std::string& prefix) {
    auto path = prefix + to_string(trace.probe_id) + "_" + trace.name + ".json";

    nlohmann::json jrep;
    jrep["name"] = trace.name;
    jrep["units"] = trace.units;
    jrep["cell"] = trace.probe_id.gid;
    jrep["probe"] = trace.probe_id.index;

    auto& jt = jrep["data"]["time"];
    auto& jy = jrep["data"][trace.name];

    for (const auto& sample: trace.samples) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(path);
    file << std::setw(1) << jrep << "\n";
}

