#pragma once

#include <memory>
#include <vector>
#include <string_view>
#include <ostream>

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

struct network_selection_impl {
    virtual std::optional<double> max_distance() const { return std::nullopt; }

    virtual bool select_connection(const network_site_info& src,
        const network_site_info& dest) const = 0;

    virtual bool select_source(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& tag) const = 0;

    virtual bool select_destination(cell_kind kind,
        cell_gid_type gid,
        const std::string_view& tag) const = 0;

    virtual void initialize(const network_label_dict& dict) {};

    virtual void print(std::ostream& os) const = 0;

    virtual ~network_selection_impl() = default;
};

inline std::shared_ptr<network_selection_impl> thingify(network_selection s,
    const network_label_dict& dict) {
    s.impl_->initialize(dict);
    return s.impl_;
}


struct network_value_impl {
    virtual double get(const network_site_info& src, const network_site_info& dest) const = 0;

    virtual void initialize(const network_label_dict& dict) {};

    virtual void print(std::ostream& os) const = 0;

    virtual ~network_value_impl() = default;
};

inline std::shared_ptr<network_value_impl> thingify(network_value v,
    const network_label_dict& dict) {
    v.impl_->initialize(dict);
    return v.impl_;
}

}  // namespace arb
