#pragma once

/*
 * Helper classes for managing sampler/schedule associations in
 * cell group classes (see sampling_api doc).
 */

#include <functional>
#include <mutex>
#include <unordered_map>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/schedule.hpp>

#include "util/transform.hpp"

namespace arb {

// An association between a samplers, schedule, and set of probe ids, as provided
// to e.g. `model::add_sampler()`.

struct sampler_association {
    schedule sched;
    sampler_function sampler;
    std::vector<cell_member_type> probe_ids;
};

// Maintain a set of associations paired with handles used for deletion.

class sampler_association_map {
public:
    void add(sampler_association_handle h, sampler_association assoc) {
        std::lock_guard<std::mutex> lock(m_);
        map_.insert({h, std::move(assoc)});
    }

    void remove(sampler_association_handle h) {
        std::lock_guard<std::mutex> lock(m_);
        map_.erase(h);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_);
        map_.clear();
    }

private:
    using assoc_map = std::unordered_map<sampler_association_handle, sampler_association>;
    assoc_map map_;
    std::mutex m_;

    static sampler_association& second(assoc_map::value_type& p) { return p.second; }
    auto assoc_view() { return util::transform_view(map_, &sampler_association_map::second); }

public:
    // Range-like view presents just the associations, omitting the handles.

    auto begin() { return assoc_view().begin(); }
    auto end()   { return assoc_view().end(); }
};

// Manage associations between probe ids, probe tags, and (lowered cell) probe handles.

template <typename Handle>
struct probe_association {
    using probe_handle_type = Handle;
    probe_handle_type handle;
    probe_tag tag;
};

template <typename Handle>
using probe_association_map = std::unordered_map<cell_member_type, probe_association<Handle>>;

} // namespace arb
