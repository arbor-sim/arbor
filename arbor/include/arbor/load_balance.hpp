#pragma once

#include <cstddef>
#include <unordered_map>

#include <arbor/export.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

struct partition_hint {
    constexpr static std::size_t max_size = -1;

    std::size_t cpu_group_size = 1;
    std::size_t gpu_group_size = max_size;
    bool prefer_gpu = true;
};

using partition_hint_map = std::unordered_map<cell_kind, partition_hint>;

ARB_ARBOR_API domain_decomposition partition_load_balance(
    const recipe& rec,
    context ctx,
    const partition_hint_map& hint_map = {});
} // namespace arb
