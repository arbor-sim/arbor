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

template<typename E, typename D, std::size_t N, typename Element = element<E, D>> 
struct elements: std::vector<Element> {
    using element_type = Element;
    static constexpr std::size_t n = N;

    using std::vector<Element>::reserve;
    using std::vector<Element>::clear;
    
    elements() {reserve(n);}

    template<typename F>
    void process(F f) {
        for (Element& element: *this) {
            f(std::move(element));
        };
        clear();
    }
};

}
}
