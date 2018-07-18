#pragma once

// Simple helper utilities for validation tests.

#include <sstream>
#include <string>

#include <arbor/common_types.hpp>

template <typename T, std::size_t N>
constexpr std::size_t size(T (&)[N]) noexcept {
    return N;
}

template <typename X>
constexpr std::size_t size(const X& x) { return x.size(); }

inline std::string to_string(arb::backend_kind kind) {
    std::stringstream out;
    out << kind;
    return out.str();
}
