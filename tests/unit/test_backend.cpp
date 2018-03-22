#include <type_traits>

#include <mechcat.hpp>
#include <backends/fvm.hpp>
#include <memory/memory.hpp>
#include <util/config.hpp>

#include "../gtest.h"

TEST(backends, gpu_is_null) {
    using backend = arb::gpu::backend;

    static_assert(std::is_same<backend, arb::null_backend>::value || arb::config::has_cuda,
        "gpu back should be defined as null when compiling without gpu support.");

    if (!arb::config::has_cuda) {
        EXPECT_FALSE(backend::is_supported());

        auto& cat = arb::global_default_catalogue();
        EXPECT_TRUE(cat.has("hh"));
        EXPECT_ANY_THROW(cat.instance<backend>("hh"));
    }
}
