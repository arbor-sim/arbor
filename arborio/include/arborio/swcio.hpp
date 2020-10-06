#pragma once

#include <arbor/swcio.hpp>

namespace arb {

// Missing soma.
struct swc_no_soma: swc_error {
    explicit swc_no_soma(int record_id);
};

// Non-consecutive soma samples.
struct swc_non_consecutive_soma: swc_error {
    explicit swc_non_consecutive_soma(int record_id);
};

// Non-serial soma samples.
struct swc_non_serial_soma: swc_error {
    explicit swc_non_serial_soma(int record_id);
};

// Sample connecting to the middle of a soma causing an unsupported branch.
struct swc_branchy_soma: swc_error {
    explicit swc_branchy_soma(int record_id);
};

// Sample connecting to the middle of a soma causing an unsupported branch.
struct swc_collocated_soma: swc_error {
    explicit swc_collocated_soma(int record_id);
};

struct swc_single_sample_segment: swc_error {
    explicit swc_single_sample_segment(int record_id);
};

segment_tree load_swc_neuron(const std::vector<swc_record>& records);

} // namespace arb
