#include <type_traits>

#include <backends/fvm.hpp>
#include <memory/memory.hpp>
#include <util/config.hpp>

#include "../gtest.h"

TEST(backends, gpu_is_null) {
    using backend = nest::mc::gpu::backend;

    static_assert(std::is_same<backend, nest::mc::null_backend>::value || nest::mc::config::has_cuda,
        "gpu back should be defined as null when compiling without gpu support.");

    if (not nest::mc::config::has_cuda) {
        EXPECT_FALSE(backend::is_supported());

        EXPECT_FALSE(backend::has_mechanism("hh"));
        EXPECT_THROW(
            backend::make_mechanism("hh", backend::view(), backend::view(), {}, {}),
            std::runtime_error);
    }
}
