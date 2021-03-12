#pragma once

namespace arb {
namespace util {

// TODO: C++20 replace with std::remove_cvref, std::remove_cvref_t

template <typename T>
struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace util
} // namespace arb
