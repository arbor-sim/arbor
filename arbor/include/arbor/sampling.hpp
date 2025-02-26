#pragma once

#include <cstddef>
#include <format>
#include <functional>
#include <iostream>

#include <arbor/assert.hpp>
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
    cell_address_type id;  // probe id
    unsigned index;        // index of probe source within those supplied by probe id
    util::any_ptr meta;    // probe-specific metadata
};

struct sample_records {
    std::size_t n_sample = 0;         // count of sample _rows_
    std::size_t width = 0;            // count of sample _columns_
    const time_type* time = nullptr;  // pointer to time data
    std::any values;                  // resolves to pointer of probe-specific payload data D of layout D[n_sample][width]
};

// Helper class, to be specialized in each cell header, mapping from Metadata to Value types
template <typename M>
struct probe_value_type_of {
    using meta_type = M;
    using type = void;
};

template<typename M,
         typename V// = probe_value_type_of<M>::type
         >
struct sample_reader {
    using value_type = V;
    using meta_type = M;

    std::size_t width = 0;
    std::size_t n_sample = 0;
    const time_type* time = nullptr;
    value_type* values = nullptr;
    meta_type* metadata = nullptr;

    // Retrieve sample value corresponding to
    // - time=get_time(i)
    // - location=get_metadata(j)
    value_type get_value(std::size_t i, std::size_t j = 0) const {
        arb_assert(i < n_sample);
        arb_assert(j < width);
        return values[i*width + j];
    }

    time_type get_time(std::size_t i) const {
        arb_assert(i < n_sample);
        return time[i];
    }

    meta_type get_metadata(std::size_t j) const {
        arb_assert(j < width);
        return metadata[j];
    }
};

// TODO M is enough to know V!
template<typename M, typename V>
auto make_sample_reader(util::any_ptr apm, const sample_records& sr) {
    using util::any_cast;
    auto pm = any_cast<M*>(apm);
    if (!pm) {
        throw std::runtime_error{std::format("Sample reader: could not cast to metadata type; expected {}, got {}.",
                                             typeid((M*)nullptr).name(), apm.type().name())};
    }
    V* val = nullptr;
    try {
        val = any_cast<V*>(sr.values);
    }
    catch(const std::bad_any_cast& e) {
        throw std::runtime_error{std::format("Sample reader: could not cast to value type; expected {}, got {}.",
                                             typeid((V*)nullptr).name(), sr.values.type().name())};
    }
    return sample_reader<M, V> { .width=sr.width,
                                 .n_sample=sr.n_sample,
                                 .time=sr.time,
                                 .values=val,
                                 .metadata=pm, };
}

using sampler_function = std::function<void(const probe_metadata&, const sample_records&)>;

using sampler_association_handle = std::size_t;

} // namespace arb
