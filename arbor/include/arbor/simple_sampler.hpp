#pragma once

/*
 * Simple(st?) implementation of a recorder of scalar
 * trace data from a cell probe, with some metadata.
 */

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>
#include <arbor/util/optional.hpp>

namespace arb {

template <typename V>
struct trace_entry {
    time_type t;
    V v;
};

// Trace data wraps a std::vector of trace_entry, optionally with
// a copy of the metadata associated with a probe.

template <typename V, typename Meta = void>
struct trace_data: public std::vector<trace_entry<V>> {
    util::optional<Meta> metadata;
};

template <typename V>
struct trace_data<V, void>: public std::vector<trace_entry<V>> {};

// Add a bit of smarts for collecting variable-length samples which are
// passed back as a pair of pointers describing a range; these can
// be used to populate a trace of vectors.

template <typename V>
struct trace_push_back {
    template <typename Meta>
    static bool push_back(trace_data<V, Meta>& trace, const sample_record& rec) {
        if (auto p = util::any_cast<const V*>(rec.data)) {
            trace.push_back({rec.time, *p});
            return true;
        }
        return false;
    }
};

template <typename V>
struct trace_push_back<std::vector<V>> {
    template <typename Meta>
    static bool push_back(trace_data<std::vector<V>, Meta>& trace, const sample_record& rec) {
        if (auto p = util::any_cast<const std::vector<V>*>(rec.data)) {
            trace.push_back({rec.time, *p});
            return true;
        }
        else if (auto p = util::any_cast<const std::pair<const V*, const V*>*>(rec.data)) {
            trace.push_back({rec.time, std::vector<V>(p->first, p->second)});
            return true;
        }
        return false;
    }
};

template <typename V, typename Meta = void>
class simple_sampler {
public:
    explicit simple_sampler(trace_data<V, Meta>& trace): trace_(trace) {}

    void operator()(cell_member_type probe_id, probe_tag tag, util::any_ptr meta, std::size_t n, const sample_record* recs) {
        // TODO: C++17 use if constexpr to test for Meta = void case.
        if (meta && !trace_.metadata) {
            if (auto m = util::any_cast<const Meta*>(meta)) {
                trace_.metadata = *m;
            }
            else {
                throw std::runtime_error("unexpected metadata type in simple_sampler");
            }
        }

        for (std::size_t i = 0; i<n; ++i) {
            if (!trace_push_back<V>::push_back(trace_, recs[i])) {
                throw std::runtime_error("unexpected sample type in simple_sampler");
            }
        }
    }

private:
    trace_data<V, Meta>& trace_;
};

template <typename V>
class simple_sampler<V, void> {
public:
    explicit simple_sampler(trace_data<V, void>& trace): trace_(trace) {}

    void operator()(cell_member_type probe_id, probe_tag tag, util::any_ptr meta, std::size_t n, const sample_record* recs) {
        for (std::size_t i = 0; i<n; ++i) {
            if (!trace_push_back<V>::push_back(trace_, recs[i])) {
                throw std::runtime_error("unexpected sample type in simple_sampler");
            }
        }
    }

private:
    trace_data<V, void>& trace_;
};

template <typename V, typename Meta>
inline simple_sampler<V, Meta> make_simple_sampler(trace_data<V, Meta>& trace) {
    return simple_sampler<V, Meta>(trace);
}

} // namespace arb
