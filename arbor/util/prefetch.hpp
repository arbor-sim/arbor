#pragma once

#include <vector>

namespace arb {
namespace prefetch {

template<typename E, typename D>
struct element {
    const E e;
    D d;

    element(const E& e_, const D& d_)
        : e{e_},
          d{d_}
    {prefetch();}

    element(const E& e_, D&& d_)
        : e{e_},
          d{std::move(d_)}
    {prefetch();}

    element(E&& e_, const D& d_)
        : e{std::move(e_)},
          d{d_}
    {prefetch();}

    element(E&& e_, D&& d_)
        : e{std::move(e_)},
          d{std::move(d_)}
    {prefetch();}

    element() = default;

    void prefetch() {__builtin_prefetch(&*e);}
};

typename<typename T, typename D, std::size_t N> 
struct elements: std::vector<element<T, D>> {
    constexpr std::size_t n = N;
    elements() {resize(n);}
};

}
}

