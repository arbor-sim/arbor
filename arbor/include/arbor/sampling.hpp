#pragma once

#include <cstddef>
#include <functional>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/any_ptr.hpp>
#include <arbor/util/extra_traits.hpp>
#include <arbor/arbexcept.hpp>

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
    std::size_t width;     // width ie count of meta items
    util::any_ptr meta;    // probe-specific metadata
};

struct sample_records {
    std::size_t n_sample = 0;         // count of sample _rows_
    std::size_t width = 0;            // count of sample _columns_
    const time_type* time = nullptr;  // pointer to time data
    std::any values;                  // resolves to pointer of probe-specific payload data D of layout D[n_sample][width]
};

template<typename M>
struct sample_reader {
    using value_type = probe_value_type_of_t<M>;
    using meta_type = M;

    std::size_t n_row() const { return n_sample_; }
    std::size_t n_column() const { return width_; }

    // Retrieve sample value corresponding to
    // - time=get_time(i)
    // - location=get_metadata(j)
    value_type value(std::size_t i, std::size_t j = 0) const {
        arb_assert(i < n_sample_);
        arb_assert(j < width_);
        return values_[i*width_ + j];
    }

    time_type time(std::size_t i) const {
        arb_assert(i < n_sample_);
        return time_[i];
    }

    meta_type metadata(std::size_t j) const {
        arb_assert(j < width_);
        return metadata_[j];
    }

    sample_reader(util::any_ptr apm,
                  const sample_records& sr):
        width_(sr.width),
        n_sample_(sr.n_sample),
        time_(sr.time)
    {
        using util::any_cast;
        if (n_sample_ == 0) return;
        metadata_ = any_cast<M*>(apm);
        if(!metadata_) throw sample_reader_metadata_error<M>{apm};
        using V = sample_reader<M>::value_type;
        try {
            values_ = any_cast<V*>(sr.values);
        }
        catch(const std::bad_any_cast& e) {
            throw sample_reader_value_error<V>{sr.values};
        }
    }

private:
    std::size_t width_ = 0;
    std::size_t n_sample_ = 0;
    const time_type* time_ = nullptr;
    value_type* values_ = nullptr;
    meta_type* metadata_ = nullptr;        
};

using sampler_function = std::function<void(const probe_metadata&, const sample_records&)>;

using sampler_association_handle = std::size_t;

} // namespace arb
