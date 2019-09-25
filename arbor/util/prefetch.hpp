#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

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

template<int v>
class mode_type {};

static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

template<std::size_t sz>
struct size_type {};

template<size_t sz>
static constexpr struct size_type<sz> size;

///////////////////////////////////////////////////////////////////
// element<prefetch_mode, prefetch-pointer, payload-pointers...> //
///////////////////////////////////////////////////////////////////
//
// pass only pointer-like things in here!
// Makes templates much simpler -- we're just storing a cut
// through arrays
//
// Internal element of prefetch
//
template<int m, typename P, typename ... Types>
class element {
public:
    using values = std::tuple<P, Types...>;
    static constexpr std::size_t size = std::tuple_size<values>();
    static constexpr int mode = m;
    
    element(P p, Types... args): v{p, args...} {prefetch();}
    element() = default;

    template<typename F>
    auto apply(F f) {
        return apply(std::forward<F>(f), std::make_index_sequence<size>{});
    }

private:
    void prefetch() {
        __builtin_prefetch(get_pointer(std::get<0>(v)), mode);
    }

    template<typename F, size_t... I>
    auto apply(F f, std::index_sequence<I...>) {
        return std::forward<F>(f)(std::move(std::get<I>(v))...);
    }

    values v;
};

//////////////////////////////////////////////////////////////////////////////////////
//////// prefetch<prefetch_mode, prefetch-pointer, payload-pointers...> //////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// a list of addresses to prefetch, and associated address
// the concept is that you continously `add` prefetch address from
// an array and their cuts through other arrays until fill
// (see `prefetch(n)`). Then you `process` a function that takes all those addresses,
// does something with them, and repeat until the entire list is
// handled. After that, the vector is `clear`ed for the next iteration
// So:
//   prefetch<A*, B*, C*> e(4);
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
// How many prefetch should we add before calling e.process?
// Not too many or we will be pushing things out of the cache
// and not too few or we'll hit the function application before
// the data has arrived

//
// prefetch_mode M = read or write, do we prefetch with a read or write expection?
// P = type of prefetch (pointer-like object)
// Types... =  payload types for calling functions on (pointer-like objects)
// process is applied on f: f(P&&, Types&&...)
//
template<std::size_t s, int m, typename F, typename P, typename ... Types>
class prefetch {
public:
    using element_type = element<m, P, Types...>;
    using array = std::array<element_type, s>;
    using iterator = typename array::iterator;
    using function_type = F;
    const function_type function;

    prefetch(F&& f): function{std::move(f)} {}
    prefetch(const F& f): function{f} {}
    ~prefetch() {process();}
    
    prefetch(prefetch&&) = default; //needed < C++17
    prefetch(const prefetch&) = delete; 
    prefetch& operator=(const prefetch&) = delete;

    // append an element to prefetch pointer-like P associated with pointer-like args
    void store(P p, Types... args) {
        if (curr == arr.end()) {
            process();
        }
        *(curr++) = element_type{p, args...};
    }

private:
    // process: applies some function f to every element of the vector
    // and then clears the vector
    // hopefully, everything is in cache by the time this is called
    void process() {
        for (auto&& element: util::make_range(arr.begin(), curr)) {
            element.apply(function);
        };
        curr = arr.begin();
    }

    array arr;
    iterator curr = arr.begin();
};

template<std::size_t s, int m, typename F, typename P, typename... Types>
constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f, P, Types...) {
    return prefetch<s, m, F, P, Types...>{std::forward<F>(f)}; // should be elided
}

// template<typename P, typename... Types, std::size_t s, int m, typename F>
// constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f) {
//     return prefetch<s, m, F, P, Types...>{std::forward<F>(f)}; // should be elided
// }

template<typename T, std::size_t s, int m, typename F = T>
struct get_prefetch_functor_args: public get_prefetch_functor_args<decltype(&T::operator()), s, m, F>
{};

template<typename T, std::size_t s, int m, typename F, typename P, typename... Types>
struct get_prefetch_functor_args<void(T::*)(P, Types...) const, s, m, F>
{
    template<typename U>
    using Remove = std::remove_cv_t<std::remove_reference_t<U>>;
    
    static constexpr auto make_prefetch(F f) {
        return prefetch<s, m, F, Remove<P>, Remove<Types>...>{std::forward<F>(f)};
    }
};

template<std::size_t s, int m, typename F>
constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f) {
    return get_prefetch_functor_args<F, s, m>::make_prefetch(std::forward<F>(f));
}

}
}
