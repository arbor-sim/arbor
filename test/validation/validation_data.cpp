#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <arbor/simple_sampler.hpp>
#include <sup/path.hpp>

#include "trace_analysis.hpp"
#include "validation_data.hpp"

namespace arb {

trace_io g_trace_io;

#ifndef ARB_DATADIR
#define ARB_DATADIR ""
#endif

sup::path trace_io::find_datadir() {
    // If environment variable is set, use that in preference.

    if (const char* env_path = std::getenv("ARB_DATADIR")) {
        return env_path;
    }

    // Otherwise try compile-time path ARB_DATADIR and hard-coded
    // relative paths below in turn, returning the first that
    // corresponds to an existing directory.

    const char* paths[] = {
        ARB_DATADIR,
        "./validation/data",
        "../validation/data"
    };

    std::error_code ec;
    for (auto p: paths) {
        if (sup::is_directory(p, ec)) {
            return p;
        }
    }

    // Otherwise set to empty path, and rely on command-line option.
    return "";
}

void trace_io::save_trace(const std::string& label, const trace_data<double>& data, const nlohmann::json& meta) {
    save_trace("time", label, data, meta);
}

void trace_io::save_trace(const std::string& abscissa, const std::string& label, const trace_data<double>& data, const nlohmann::json& meta) {
    using nlohmann::json;

    json j = meta;
    json& times = j["data"][abscissa];
    json& values = j["data"][label];

    for (const auto& e: data) {
        times.push_back(e.t);
        values.push_back(e.v);
    }

    jtraces_ += std::move(j);
}

template <typename Seq1, typename Seq2>
static trace_data<double> zip_trace_data(const Seq1& ts, const Seq2& vs) {
    trace_data<double> trace;

    auto ti = std::begin(ts);
    auto te = std::end(ts);
    auto vi = std::begin(vs);
    auto ve = std::end(vs);

    while (ti!=te && vi!=ve) {
        trace.push_back({*ti++, *vi++});
    }
    return trace;
}

static void parse_trace_json(const nlohmann::json& j, std::map<std::string, trace_data<double>>& traces) {
    if (j.is_array()) {
        for (auto& i: j) parse_trace_json(i, traces);
    }
    else if (j.is_object() && j.count("data")>0 && j["data"].count("time")>0) {
        auto data = j["data"];
        auto time = data["time"].get<std::vector<float>>();
        for (const auto& p: nlohmann::json::iterator_wrapper(data)) {
            if (p.key()=="time") continue;

            traces[p.key()] = zip_trace_data(time, p.value().get<std::vector<double>>());
        }
    }
}

std::map<std::string, trace_data<double>> trace_io::load_traces(const sup::path& name) {
    sup::path file  = datadir_/name;
    std::ifstream fid(file);
    if (!fid) {
        throw std::runtime_error("unable to load validation data: "+file.native());
    }

    nlohmann::json data;
    fid >> data;

    std::map<std::string, trace_data<double>> traces;
    parse_trace_json(data, traces);
    return traces;
}

} // namespace arb

