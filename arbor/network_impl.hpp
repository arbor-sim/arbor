#pragma once

#include <vector>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/network.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>

#include "connection.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"

namespace arb {

struct network_selection_impl {
    virtual std::optional<double> max_distance() const { return std::nullopt; }

    virtual bool select_connection(const network_site_info& src,
        const network_site_info& dest) const = 0;

    virtual bool select_source(cell_kind kind,
        cell_gid_type gid,
        const cell_tag_type& tag) const = 0;

    virtual bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const cell_tag_type& tag) const = 0;

    virtual ~network_selection_impl() = default;
};

} // namespace arb
