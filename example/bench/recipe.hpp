#pragma once

#include <arbor/common_types.hpp>

#include <recipe.hpp>

#include "parameters.hpp"

using arb::cell_kind;
using arb::cell_gid_type;
using arb::cell_size_type;

class bench_recipe: public arb::recipe {
    bench_params params_;
public:
    bench_recipe(bench_params p): params_(std::move(p)) {}
    cell_size_type num_cells() const override;
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override;
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override;
    cell_size_type num_targets(cell_gid_type gid) const override;
    cell_size_type num_sources(cell_gid_type gid) const override;
    std::vector<arb::cell_connection> connections_on(cell_gid_type) const override;
};

