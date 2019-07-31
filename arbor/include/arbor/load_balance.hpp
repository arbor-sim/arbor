#pragma once

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

struct partition_hint {

    partition_hint() :  cpu_group_size{cpu_group_size_default},
                        gpu_group_size{max_size},
                        prefer_gpu{true}
    {};

    partition_hint(std::size_t cpu, std::size_t gpu, bool prefer) {
        set_cpu_group_size(cpu);
        set_gpu_group_size(gpu);
        set_prefer_gpu(prefer);
    };

    void set_cpu_group_size(std::size_t cpu){
        cpu_group_size = (cpu > 0) ? cpu : cpu_group_size_default;
    };

    void set_gpu_group_size(std::size_t gpu){
        gpu_group_size = (gpu > 0) ? gpu : max_size;
    };

    void set_prefer_gpu(std::size_t prefer) {
        prefer_gpu = prefer;
    };

    std::size_t get_cpu_group_size() const {
        return cpu_group_size;
    };

    std::size_t get_gpu_group_size() const {
        return gpu_group_size;
    };

    bool get_prefer_gpu() const {
        return prefer_gpu;
    };

    std::size_t get_max_size() const {
        return max_size;
    };

    private:
        constexpr static std::size_t max_size = -1;
        constexpr static std::size_t cpu_group_size_default = 1;

        std::size_t cpu_group_size;
        std::size_t gpu_group_size;
        bool prefer_gpu;

};

using partition_hint_map = std::unordered_map<cell_kind, partition_hint>;

domain_decomposition partition_load_balance(
    const recipe& rec,
    const context& ctx,
    partition_hint_map hint_map = {});

} // namespace arb
