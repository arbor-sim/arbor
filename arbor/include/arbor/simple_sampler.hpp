#pragma once

#include <type_traits>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

namespace arb {

// Simple(st?) implementation of a recorder of scalar trace data from a cell
// probe, with some metadata.
template<typename M>
struct simple_sampler_result {
    using value_type = probe_value_type_of_t<M>;
    std::size_t n_sample = 0;
    std::size_t width = 0;
    std::vector<time_type> time;
    std::vector<std::vector<std::remove_const_t<value_type>>> values;
    std::vector<std::remove_const_t<M>> metadata;

    void clear() {
        n_sample = 0;
        width = 0;
        values.clear();
        time.clear();
        metadata.clear();
    }

    void from_reader(const sample_reader<M>& reader) {
        clear();
        width = reader.n_column();
        values.resize(width);
        for (std::size_t iy = 0ul; iy < reader.n_column(); ++iy) {
            metadata.push_back(reader.metadata(iy));
        }
        append_reader_unchecked(reader);
    }

    void append_reader_unchecked(const sample_reader<M>& reader) {
        n_sample += reader.n_row();
        for (std::size_t ix = 0ul; ix < reader.n_row(); ++ix) {
            time.push_back(reader.time(ix));
            for (std::size_t iy = 0ul; iy < reader.n_column(); ++iy) {
                auto v = reader.value(ix, iy);
                values[iy].push_back(v);
            }
        }
    }

    void append_reader(const sample_reader<M>& reader) {
        if (width != reader.n_column()) {
            throw std::invalid_argument("Expected reader of width " + std::to_string(width) + " but received " + std::to_string(reader.n_column()));
        }
        for (std::size_t iy = 0ul; iy < reader.n_column(); ++iy) {
            if (metadata[iy] != reader.metadata(iy)) {
                throw std::invalid_argument("Attemptyed to append reader with incompatible metadata");
            }
        }
        append_reader_unchecked(reader);
    }
};

template <typename M>
auto make_simple_sampler(simple_sampler_result<M>& trace) {
    return [&trace](const probe_metadata& pm, const sample_records& recs) {
        auto reader = sample_reader<M>(pm.meta, recs);
        trace.from_reader(reader);
    };
}

} // namespace arb
