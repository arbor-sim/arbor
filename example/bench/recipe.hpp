#pragma once

#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>
#include <arbor/util/unique_any.hpp>

#include "parameters.hpp"

class bench_recipe: public arb::recipe {
private:
    bench_params params_;

public:
    bench_recipe(bench_params p): params_(std::move(p)) {}
    arb::cell_size_type num_cells() const override;
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override;
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override;
    arb::cell_size_type num_targets(arb::cell_gid_type gid) const override;
    arb::cell_size_type num_sources(arb::cell_gid_type gid) const override;
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type) const override;
};

