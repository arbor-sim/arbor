#pragma once

#include <vector>

namespace arb {
namespace prefetch {

// default conversion from pointer-like P to pointer P
template<typename P>
auto get_pointer(P& prefetched) {
    return &*prefetched;
}

// pass only pointer-like things in here!
// Makes templates much simpler -- we're just storing a cut
// through arrays
template<typename P, typename ... Types>
class element: public std::tuple<P, Types...> {
public:
    using parent = std::tuple<P, Types...>;
    static constexpr std::size_t size = std::tuple_size<parent>();
    
    element(P p, Types... args): parent{p, args...} {prefetch();}
    element() = default;

    template<typename F>
    void apply(F f) {
        apply(std::forward<F>(f), std::make_index_sequence<size>{});
    }

private:
    void prefetch() {__builtin_prefetch(get_pointer(std::get<0>(*this)), 1);}

    template<typename F, size_t... I>
    void apply(F f, std::index_sequence<I...>) {
        std::forward<F>(f)(std::get<I>(*this)...);
    }
};

template<std::size_t N, typename P, typename ... Types> 
struct elements: public std::vector<element<P, Types...>> {
    using element_type = element<P, Types...>;
    using parent = std::vector<element_type>;
    
    static constexpr std::size_t n = N;
    elements() {reserve(n);}

    // append an element to prefetch pointer-like P associated with pointer-like args
    void add(P p, Types... args) {push_back(element_type{p, args...});}

    // process: applies some function f to every element of the vector
    // and then clears the vector
    // hopefully, everything is in cache by the time this is called
    template<typename F>
    void process(F f) {
        for (auto&& element: *this) {
            element.apply(std::forward<F>(f));
        };
        clear();
    }

    using parent::reserve;
    using parent::clear;
    using parent::push_back;
};

}
}
