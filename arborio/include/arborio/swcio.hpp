#pragma once

#include <arbor/swcio.hpp>

using arb::swc_error;

namespace arborio {
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

// Sample is not part of a segment
struct swc_single_sample_segment: swc_error {
    explicit swc_single_sample_segment(int record_id);
};

// Segment cannot have samples with different tags
struct swc_mismatched_tags: swc_error {
    explicit swc_mismatched_tags(int record_id);
};

// Only tags 1, 2, 3, 4 supported
struct swc_unsupported_tag: swc_error {
    explicit swc_unsupported_tag(int record_id);
};

// No gaps allowed
struct swc_unsupported_gaps: swc_error {
    explicit swc_unsupported_gaps(int record_id);
};

arb::segment_tree load_swc_neuron(const std::vector<arb::swc_record>& records);
arb::segment_tree load_swc_allen(std::vector<arb::swc_record>& records, bool no_gaps=false);

} // namespace arborio
