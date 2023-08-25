#pragma once

#include <cstddef>
#include <functional>

#include <arbor/common_types.hpp>
#include <arbor/util/any_ptr.hpp>

namespace arb {

using cell_member_predicate = std::function<bool (const cell_address_type&)>;

static cell_member_predicate all_probes = [](const cell_address_type&) { return true; };

struct one_probe {
    one_probe(cell_address_type p): pid{std::move(p)} {}
    cell_address_type pid;
    bool operator()(const cell_address_type& x) { return x == pid; }
};

struct one_gid {
    one_gid(cell_gid_type p): gid{std::move(p)} {}
    cell_gid_type gid;
    bool operator()(const cell_address_type& x) { return x.gid == gid; }
};
struct one_tag {
    one_tag(cell_tag_type p): tag{std::move(p)} {}
    cell_tag_type tag;
    bool operator()(const cell_address_type& x) { return x.tag == tag; }
};


// Probe-specific metadata is provided by cell group implementations.
//
// User code is responsible for correctly determining the metadata type,
// but the value of that metadata must be sufficient to determine the
// correct interpretation of sample data provided to sampler callbacks.

struct probe_metadata {
    cell_address_type id; // probe id
    unsigned index;       // index of probe source within those supplied by probe id
    util::any_ptr meta;   // probe-specific metadata
};

struct sample_record {
    time_type time;
    util::any_ptr data;
};

using sampler_function = std::function<
    void (probe_metadata,
          std::size_t,          // number of sample records
          const sample_record*  // pointer to first sample record
         )>;

using sampler_association_handle = std::size_t;

} // namespace arb
