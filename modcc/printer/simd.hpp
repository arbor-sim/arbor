#pragma once

struct simd_spec {
    enum simd_abi { none, neon, avx, avx2, avx512, native, default_abi } abi = none;
    unsigned width = 0; // zero => use `simd::native_width` to determine.

    simd_spec() = default;
    simd_spec(enum simd_abi a, unsigned w = 0):
        abi(a), width(w)
    {
        if (width==0) {
            // Pick a width based on abi, if applicable.
            switch (abi) {
            case avx:
            case avx2:
                width = 4;
                break;
            case avx512:
                width = 8;
                break;
            case neon:
                width = 2;
                break;
            default: ;
            }
        }
    }
};
