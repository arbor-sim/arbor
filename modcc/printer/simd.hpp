#pragma once

#include <string>

constexpr unsigned no_size = unsigned(-1);

struct simd_spec {
    enum simd_abi { none, neon, avx, avx2, avx512, sve, vls_sve, native, default_abi } abi = none;
    std::string width;
    std::string size;

    simd_spec() = default;
    simd_spec(enum simd_abi a, unsigned w = 2):
        abi(a)
    {
        switch(abi) {
            // default_abi allows for any width (will choose generic backend if it doesn't match the native abi)
            case(default_abi):
                width = std::to_string(w);
                break;
            // all other backends use the native width
            default:
                width = "S::simd_abi::native_width<double>::value";
        }
        switch(abi) {
            // sve is "sizeless"
            case(sve):
                size = "0";
                break;
            // all other backends have size
            default:
                size = width;
        }
    }
};
