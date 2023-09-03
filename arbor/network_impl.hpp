#pragma once

#include <memory>
#include <ostream>
#include <string_view>
#include <vector>

#include <arbor/arbexcept.hpp>
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

struct ARB_SYMBOL_VISIBLE network_full_site_info {
    network_full_site_info() = default;

    network_full_site_info(cell_gid_type gid,
        cell_lid_type lid,
        cell_kind kind,
        std::string_view label,
        mlocation location,
        mpoint global_location);

    cell_gid_type gid;
    cell_lid_type lid;
    cell_kind kind;
    std::string_view label;
    mlocation location;
    mpoint global_location;
    network_hash_type hash;
};

struct network_selection_impl {
    virtual std::optional<double> max_distance() const { return std::nullopt; }

    virtual bool select_connection(const network_full_site_info& source,
        const network_full_site_info& target) const = 0;

    virtual bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& tag) const = 0;

    virtual bool select_target(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& tag) const = 0;

    virtual void initialize(const network_label_dict& dict){};

    virtual void print(std::ostream& os) const = 0;

    virtual ~network_selection_impl() = default;
};

inline std::shared_ptr<network_selection_impl> thingify(network_selection s,
    const network_label_dict& dict) {
    s.impl_->initialize(dict);
    return s.impl_;
}

struct network_value_impl {
    virtual double get(const network_full_site_info& source,
        const network_full_site_info& target) const = 0;

    virtual void initialize(const network_label_dict& dict){};

    virtual void print(std::ostream& os) const = 0;

    virtual ~network_value_impl() = default;
};

inline std::shared_ptr<network_value_impl> thingify(network_value v,
    const network_label_dict& dict) {
    v.impl_->initialize(dict);
    return v.impl_;
}

std::vector<connection> generate_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec);

}  // namespace arb
