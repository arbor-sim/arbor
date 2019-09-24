#pragma once

#include <vector>

namespace arb {
namespace prefetch {

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
template<typename P>
auto get_pointer(P& prefetched) {
    return &*prefetched;
}

/////////////////////////////
// Prefetch as read or write
// constants for __builtin_prefetch
enum prefetch_mode {
    read = 0,
    write = 1
};

///////////////////////////////////////////////////////////////////
// element<prefetch_mode, prefetch-pointer, payload-pointers...> //
///////////////////////////////////////////////////////////////////
//
// pass only pointer-like things in here!
// Makes templates much simpler -- we're just storing a cut
// through arrays
//
// Internal element of elements
//
template<prefetch_mode M, typename P, typename ... Types>
class element: public std::tuple<P, Types...> {
public:
    using parent = std::tuple<P, Types...>;
    static constexpr std::size_t size = std::tuple_size<parent>();
    static constexpr prefetch_mode m = M;
    
    element(P p, Types... args): parent{p, args...} {prefetch();}
    element() = default;

    template<typename F>
    auto apply(F f) {
        return apply(std::forward<F>(f), std::make_index_sequence<size>{});
    }

private:
    void prefetch() {__builtin_prefetch(get_pointer(std::get<0>(*this)), m);}

    template<typename F, size_t... I>
    auto apply(F f, std::index_sequence<I...>) {
        return std::forward<F>(f)(std::get<I>(*this)...);
    }
};

//////////////////////////////////////////////////////////////////////////////////////
//////// elements<prefetch_mode, prefetch-pointer, payload-pointers...> //////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// a list of addresses to prefetch, and associated address
// the concept is that you continously `add` prefetch address from
// an array and their cuts through other arrays until fill
// (see `elements(n)`). Then you `process` a function that takes all those addresses,
// does something with them, and repeat until the entire list is
// handled. After that, the vector is `clear`ed for the next iteration
// So:
//   elements<A*, B*, C*> e(4);
//   for (A* a = Ar; a < Ar+end; a++) {
//      e.add(a, Br+(a-Ar), Cr+(a-Ar));
//      if (e.full()) {
//         e.process([] (auto&& a_, auto&& b_, auto&& c_) {
//              a->do_domething(b_, c_);
//         });
//      };
//   }
//   e.process([] (auto&& a_, auto&& b_, auto&& c_) { /* handle left over */
//       a->do_domething(b_, c_);
//   });
//
// Prefetching is actually called when the `element` is constructed from the add arguments, where the
// first argument has prefetch::get_pointer applied giving an address to call
// on __builtin_prefetch
//
// The "hard bit" is determining the range of good sizes for the capacity.
// How many elements should we add before calling e.process?
// Not too many or we will be pushing things out of the cache
// and not too few or we'll hit the function application before
// the data has arrived

//
// prefetch_mode M = read or write, do we prefetch with a read or write expection?
// P = type of prefetch (pointer-like object)
// Types... =  payload types for calling functions on (pointer-like objects)
// process is applied on f: f(P&&, Types&&...)
//
template<prefetch_mode M, typename P, typename ... Types> 
struct elements: public std::vector<element<M, P, Types...>> {
    using element_type = element<M, P, Types...>;
    using parent = std::vector<element_type>;

    const std::size_t n;
    elements(): n{} {}
    elements(std::size_t n_): n{n_} {reserve(n);};
    

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

    bool not_full() const {return size() < n;}
    bool full() const {return ! not_full();}

    using parent::reserve;
    using parent::clear;
    using parent::push_back;
    using parent::size;
};

}
}
