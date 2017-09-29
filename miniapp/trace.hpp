#pragma once

/*
 * Store trace data from samplers with metadata.
 */

#include <string>
#include <vector>

#include <common_types.hpp>
#include <simple_sampler.hpp>

struct sample_trace {
    arb::cell_member_type probe_id;
    std::string name;
    std::string units;
    arb::trace_data<double> samples;
};

void write_trace_csv(const sample_trace& trace, const std::string& prefix);
void write_trace_json(const sample_trace& trace, const std::string& prefix);
