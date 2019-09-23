#pragma once

#include <vector>

namespace arb {
namespace prefetch {

// default conversion from pointer-like P to pointer P
template<typename P>
auto get_pointer(P& prefetched) {
    return &*prefetched;
}

template<typename P, typename ... Types>
struct element {
    P prefetched; // should be a pointer-like type, where &*prefetched is an address
    std::tuple<Types...> payload; // and this is anything we want to be associated with it.

    element(const P& p, const Types&... args): prefetched{p}, payload{args...} {prefetch();}
    element(const P& p, Types&&... args): prefetched{p}, payload{std::move(args)...} {prefetch();}
    element(P&& p, const Types&... args): prefetched{std::move(p)}, payload{args...} {prefetch();}
    element(P&& p, Types&&... args): prefetched{std::move(p)}, payload{std::move(args)...} {prefetch();}
    element() = default;

    void prefetch() {__builtin_prefetch(get_pointer(prefetched), 1);}
};

template<typename P, std::size_t N, typename ... Types> 
struct elements: std::vector<element<P, Types...>> {
    using element_type = element<P, Types...>;
    using parent = std::vector<element_type>;
    
    static constexpr std::size_t n = N;
    elements() {reserve(n);}

    // process: applies some function f to every element of the vector
    // and then clears the vector
    // hopefully, everything is in cache by the time this is called
    template<typename F>
    void process(F f) {
        for (element_type& element: *this) {
            f(std::move(element));
        };
        clear();
    }

    using parent::reserve;
    using parent::clear;
};

}
}
