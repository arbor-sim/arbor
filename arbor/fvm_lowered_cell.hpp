#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/recipe.hpp>
#include <arbor/util/any_ptr.hpp>

#include "backends/event.hpp"
#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"
#include "sampler_map.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"
#include "util/transform.hpp"

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const arb_value_type*> sample_time;
    util::range<const arb_value_type*> sample_value;
};

// A sample for a probe may be derived from multiple 'raw' sampled
// values from the backend.

// An `fvm_probe_data` object represents the mapping between
// a sample value (possibly non-scalar) presented to a sampler
// function, and one or more probe handles that reference data
// in the FVM back-end.

struct fvm_probe_scalar {
    probe_handle raw_handles[1] = {nullptr};
    std::variant<mlocation, cable_probe_point_info> metadata;

    util::any_ptr get_metadata_ptr() const {
        return std::visit([](const auto& x) -> util::any_ptr { return &x; }, metadata);
    }
};

struct fvm_probe_interpolated {
    probe_handle raw_handles[2] = {nullptr, nullptr};
    double coef[2] = {};
    mlocation metadata;

    util::any_ptr get_metadata_ptr() const { return &metadata; }
};

struct fvm_probe_multi {
    std::vector<probe_handle> raw_handles;
    std::variant<mcable_list, std::vector<cable_probe_point_info>> metadata;

    void shrink_to_fit() {
        raw_handles.shrink_to_fit();
        std::visit([](auto& v) { v.shrink_to_fit(); }, metadata);
    }

    util::any_ptr get_metadata_ptr() const {
        return std::visit([](const auto& x) -> util::any_ptr { return &x; }, metadata);
    }
};

struct fvm_probe_weighted_multi {
    std::vector<probe_handle> raw_handles;
    std::vector<double> weight;
    mcable_list metadata;

    void shrink_to_fit() {
        raw_handles.shrink_to_fit();
        weight.shrink_to_fit();
        metadata.shrink_to_fit();
    }

    util::any_ptr get_metadata_ptr() const { return &metadata; }
};

struct fvm_probe_interpolated_multi {
    std::vector<probe_handle> raw_handles; // First half take coef[0], second half coef[1].
    std::vector<double> coef[2];
    mcable_list metadata;

    void shrink_to_fit() {
        raw_handles.shrink_to_fit();
        coef[0].shrink_to_fit();
        coef[1].shrink_to_fit();
        metadata.shrink_to_fit();
    }

    util::any_ptr get_metadata_ptr() const { return &metadata; }
};

// Trans-membrane currents require special handling!
struct fvm_probe_membrane_currents {
    std::vector<probe_handle> raw_handles; // Voltage per CV, followed by stim current densities.
    std::vector<mcable> metadata;          // Cables from each CV, in CV order.

    std::vector<unsigned> cv_parent;       // Parent CV index for each CV.
    std::vector<double> cv_parent_cond;    // Face conductance between CV and parent.
    std::vector<double> weight;            // Area of cable : area of CV.
    std::vector<unsigned> cv_cables_divs;  // Partitions metadata by CV index.

    std::vector<double> stim_scale;        // CV area for scaling raw stim current densities.
    std::vector<unsigned> stim_cv;         // CV index corresponding to each stim raw handle.

    void shrink_to_fit() {
        raw_handles.shrink_to_fit();
        metadata.shrink_to_fit();
        cv_parent.shrink_to_fit();
        cv_parent_cond.shrink_to_fit();
        weight.shrink_to_fit();
        cv_cables_divs.shrink_to_fit();
        stim_scale.shrink_to_fit();
        stim_cv.shrink_to_fit();
    }

    util::any_ptr get_metadata_ptr() const { return &metadata; }
};

struct missing_probe_info {
    // dummy data...
    std::array<probe_handle, 0> raw_handles;
    void* metadata = nullptr;

    util::any_ptr get_metadata_ptr() const { return util::any_ptr{}; }
};

struct fvm_probe_data {
    fvm_probe_data() = default;
    fvm_probe_data(fvm_probe_scalar p): info(std::move(p)) {}
    fvm_probe_data(fvm_probe_interpolated p): info(std::move(p)) {}
    fvm_probe_data(fvm_probe_multi p): info(std::move(p)) {}
    fvm_probe_data(fvm_probe_weighted_multi p): info(std::move(p)) {}
    fvm_probe_data(fvm_probe_interpolated_multi p): info(std::move(p)) {}
    fvm_probe_data(fvm_probe_membrane_currents p): info(std::move(p)) {}

    std::variant<
        missing_probe_info,
        fvm_probe_scalar,
        fvm_probe_interpolated,
        fvm_probe_multi,
        fvm_probe_weighted_multi,
        fvm_probe_interpolated_multi,
        fvm_probe_membrane_currents
    > info = missing_probe_info{};

    auto raw_handle_range() const {
        return util::make_range(
            std::visit(
                [](auto& i) -> std::pair<const probe_handle*, const probe_handle*> {
                    using std::data;
                    using std::size;
                    return {data(i.raw_handles), data(i.raw_handles)+size(i.raw_handles)};
                },
                info));
    }

    util::any_ptr get_metadata_ptr() const {
        return std::visit([](const auto& i) -> util::any_ptr { return i.get_metadata_ptr(); }, info);
    }

    sample_size_type n_raw() const { return raw_handle_range().size(); }

    explicit operator bool() const { return !std::get_if<missing_probe_info>(&info); }
};

// Samplers are tied to probe ids, but one probe id may
// map to multiple probe representations within the mc_cell_group.

struct probe_association_map {
    // Keys are probe id.

    std::unordered_map<cell_member_type, probe_tag> tag;
    std::unordered_multimap<cell_member_type, fvm_probe_data> data;

    std::size_t size() const {
        arb_assert(tag.size()==data.size());
        return data.size();
    }

    // Return range of fvm_probe_data values associated with probeset_id.
    auto data_on(cell_member_type probeset_id) const {
        return util::transform_view(util::make_range(data.equal_range(probeset_id)), util::second);
    }
};

struct fvm_initialization_data {
    // Map from gid to integration domain id
    std::vector<arb_index_type> cell_to_intdom;

    // Handles for accessing lowered cell.
    std::vector<target_handle> target_handles;

    // Maps probe ids to probe handles and tags.
    probe_association_map probe_map;

    // Structs required for {gid, label} to lid resolution
    cell_label_range source_data;
    cell_label_range target_data;
    cell_label_range gap_junction_data;

    // Maps storing number of sources/targets per cell.
    std::unordered_map<cell_gid_type, arb_size_type> num_sources;
    std::unordered_map<cell_gid_type, arb_size_type> num_targets;
};

// Common base class for FVM implementation on host or gpu back-end.

struct fvm_lowered_cell {
    virtual void reset() = 0;

    virtual fvm_initialization_data initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec) = 0;

    virtual fvm_integration_result integrate(
        arb_value_type tfinal,
        arb_value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples) = 0;

    virtual arb_value_type time() const = 0;

    virtual ~fvm_lowered_cell() {}
};

using fvm_lowered_cell_ptr = std::unique_ptr<fvm_lowered_cell>;

ARB_ARBOR_API fvm_lowered_cell_ptr make_fvm_lowered_cell(backend_kind p, const execution_context& ctx,
        std::uint64_t seed = 0);

} // namespace arb
