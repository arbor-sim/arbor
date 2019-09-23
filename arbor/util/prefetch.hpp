#pragma once

#include <vector>

namespace arb {
namespace prefetch {

// default conversion from pointer-like P to pointer P
template<typename P>
auto get_pointer(P& prefetched) {
    return &*prefetched;
}

template<typename P, typename D>
struct element {
    P prefetched; // should be a pointer-like type, where &*prefetched is an address
    D payload; // and this is anything we want to be associated with it.

    element(const P& p, const D& d): prefetched{p}, payload{d} {prefetch();}
    element(const P& p, D&& d): prefetched{p}, payload{std::move(d)} {prefetch();}
    element(P&& p, const D& d): prefetched{std::move(p)}, payload{d} {prefetch();}
    element(P&& p, D&& d): prefetched{std::move(p)}, payload{std::move(d)} {prefetch();}
    element() = default;

    void prefetch() {__builtin_prefetch(get_pointer(prefetched), 1);}
};

template<typename P, typename D, std::size_t N, typename Element = element<P, D>> 
struct elements: std::vector<Element> {
    using element_type = Element;
    static constexpr std::size_t n = N;

    elements() {reserve(n);}

    // process: applies some function f to every element of the vector
    // and then clears the vector
    // hopefully, everything is in cache by the time this is called
    template<typename F>
    void process(F f) {
        for (Element& element: *this) {
            f(std::move(element));
        };
        clear();
    }

    using std::vector<Element>::reserve;
    using std::vector<Element>::clear;
};

}
}
