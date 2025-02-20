#pragma once

/*
 * Simple(st?) implementation of a recorder of scalar
 * trace data from a cell probe, with some metadata.
 */

#include <stdexcept>
#include <type_traits>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

namespace arb {

template<typename M, typename V>
struct simple_sampler_result {
    std::size_t n_sample = 0;
    std::size_t width = 0;
    std::vector<time_type> time;
    std::vector<std::vector<std::remove_const_t<V>>> values;
    std::vector<std::remove_const_t<M>> metadata;

    void from_reader(const sample_reader<M, V>& reader) {
        n_sample = reader.n_sample;
        width = reader.width;
        values.resize(width);
        for (std::size_t ix = 0ul; ix < reader.n_sample; ++ix) {
            time.push_back(reader.get_time(ix));
            metadata.push_back(reader.get_metadata(ix));
            for (std::size_t iy = 0ul; iy < reader.n_sample; ++iy) {
                auto v = reader.get_value(ix, iy);
                values[iy].push_back(v);
            }
        }
    }
};

template <typename M, typename V>
auto make_simple_sampler(simple_sampler_result<M, V>& trace) {
    return [&trace](const probe_metadata& pm, const sample_records& recs) {
        auto reader = make_sample_reader<M, V>(pm.meta, recs);
        trace.from_reader(reader);
    };
}

} // namespace arb
