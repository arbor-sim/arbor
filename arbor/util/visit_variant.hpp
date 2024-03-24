#pragma once


#include <type_traits>
#include <variant>


namespace arb {
namespace util {

namespace impl {
template <typename VARIANT, typename F>
inline void visit_variant_impl(VARIANT &&v, F &&f) {
    constexpr auto index = std::variant_size_v<std::remove_reference_t<VARIANT>> - 1;
    if (v.index() == index) f(std::get<index>(v));
}

template <typename VARIANT, typename F, typename... FUNCS>
inline void visit_variant_impl(VARIANT &&v, F &&f, FUNCS &&...functions) {
    constexpr auto index =
        std::variant_size_v<std::remove_reference_t<VARIANT>> - sizeof...(FUNCS) - 1;
    if (v.index() == index) f(std::get<index>(v));
    visit_variant_impl(std::forward<VARIANT &&>(v), std::forward<FUNCS &&>(functions)...);
}
}  // namespace impl

/*
 * Similar to std::visit, call contained type with matching function. Expects a function for each
 * type in variant and in the same order. More performant than std::visit through the use of
 * indexing instead of function tables.
 */
template <typename VARIANT, typename... FUNCS>
inline void visit_variant(VARIANT &&v, FUNCS &&...functions) {
    static_assert(std::variant_size_v<std::remove_reference_t<VARIANT>> ==
                      sizeof...(FUNCS),
                  "The first argument must be of type std::variant and the "
                  "number of functions must match the variant size.");
    impl::visit_variant_impl(std::forward<VARIANT &&>(v), std::forward<FUNCS &&>(functions)...);
}
} // namespace util
} // namespace arb
