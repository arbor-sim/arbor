#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include <json/json.hpp>

#include <simple_sampler.hpp>
#include <util/path.hpp>

#include "validation_data.hpp"

namespace arb {

trace_io g_trace_io;

#ifndef NMC_DATADIR
#define NMC_DATADIR ""
#endif

util::path trace_io::find_datadir() {
    // If environment variable is set, use that in preference.

    if (const char* env_path = std::getenv("NMC_DATADIR")) {
        return util::path(env_path);
    }

    // Otherwise try compile-time path NMC_DATADIR and hard-coded
    // relative paths below in turn, returning the first that
    // corresponds to an existing directory.

    const char* paths[] = {
        NMC_DATADIR,
        "./validation/data",
        "../validation/data"
    };

    std::error_code ec;
    for (auto p: paths) {
        if (util::is_directory(p, ec)) {
            return util::path(p);
        }
    }

    // Otherwise set to empty path, and rely on command-line option.
    return util::path();
}

void trace_io::save_trace(const std::string& label, const trace_data<double>& data, const nlohmann::json& meta) {
    save_trace("time", label, data, meta);
}

void trace_io::save_trace(const std::string& abscissa, const std::string& label, const trace_data<double>& data, const nlohmann::json& meta) {
    using namespace arb;

    nlohmann::json j = meta;
    j["data"] = {
        {abscissa, times(data)},
        {label, values(data)}
    };

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

std::map<std::string, trace_data<double>> trace_io::load_traces(const util::path& name) {
    util::path file  = datadir_/name;
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

