#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

namespace arb {
namespace prefetch {

// Internal utility
template<typename T>
using remove_qualifier_t = std::remove_cv_t<std::remove_reference_t<T>>;

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
template<typename P>
auto get_pointer(P prefetched) {
    return &*prefetched;
}

/////////////////////////////
// Prefetch as read or write
// constants for __builtin_prefetch

/* 
   element<prefetch_mode, prefetch-pointer, payload-pointers...> //

   pass only pointer-like things in here!
   Makes templates much simpler -- we're just storing a cut
   through arrays

   Internal element of prefetch<>

   m = 0 for read, or 1 for write
*/
template<int m, typename P, typename ... Types>
class element {
public:
    static constexpr int mode = m;
    using values = std::tuple<remove_qualifier_t<P>, remove_qualifier_t<Types>...>;
    static constexpr std::size_t size = std::tuple_size<values>();
    
    element(remove_qualifier_t<P> p, remove_qualifier_t<Types>... args): v{p, args...} {prefetch();}
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
        return std::forward<F>(f)(get_raw_value<I>()...);
    }

    template<std::size_t n>
    using raw_type = std::tuple_element_t<n, std::tuple<P, Types...>>;
    
    template<std::size_t n>
    raw_type<n> get_raw_value() {
        return static_cast<raw_type<n>>(std::get<n>(v));
    }

    values v;
};

/*
  prefetch<prefetch-size, prefetch-mode, func-to-apply prefetch-pointer, payload-pointers...>
  construct with make_prefetch utilities below

  stores a list of addresses to prefetch, and associated argument addresses
  then calls a function on them when full (or at destruction)

  the concept is that you continously `store` prefetch address from
  an array and their cuts through other arrays until fill
  (see `store()`). Prefetch is called when they are stored.

  When full, a functor is called on all the arguments in the hope that
  the prefetch has already pulled the data associated with the prefetch-pointer.

  void do_it(vec) {
    auto p = make_prefetch(
               size_type<n>, 
               `read` or `write`,
               [] (prefetch-pointer&& p, args-pointer* args...) {
                 p->do_something(args...);
             });

    for (obj: vec) {
      p.store(obj.p, obj.args...);
    }
  }

  Prefetching is actually called when the `element` is constructed from the store arguments, where the
  first argument has prefetch::get_pointer applied giving an address to call
  on __builtin_prefetch. For iterators or pointers, prefetch::get_pointer applies &* to the object.

  The "hard bit" is determining the range of good sizes for the capacity.
  How many prefetch should we add before calling e.process?
  Not too many or we will be pushing things out of the cache
  and not too few or we'll hit the function application before
  the data has arrived

  prefetch size = number of read-aheads
  prefetch_mode m = read or write, do we prefetch with a read or write expection?
  P = type of prefetch (pointer-like object)
  Types... =  payload types for calling functions on (pointer-like objects)
  process is applied on f: f(P&&, Types&&...)
*/
template<std::size_t s, int m, typename F, typename P, typename ... Types>
class prefetch {
public:
    using element_type = element<m, P, Types...>;
    using array = std::array<element_type, s+1>; // 1 sentinel element
    using iterator = typename array::iterator;
    using function_type = F;
    const function_type function;

    prefetch(F&& f): function{std::move(f)} {}
    prefetch(const F& f): function{f} {}
    ~prefetch() {while (begin != end) {pop();}}
    
    prefetch(prefetch&&) = default; //needed < C++17
    prefetch(const prefetch&) = delete; 
    prefetch& operator=(const prefetch&) = delete;

    // append an element to prefetch pointer-like P associated
    // with pointer-like args. If enough look-aheads pending
    // process one (call function on it).
    void store(remove_qualifier_t<P> p, remove_qualifier_t<Types>... args) {
        if (begin == next) {pop();}
        push(p, args...);
    }

private:
    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() {
        begin->apply(function);
        if (++begin == arr.end()) {begin = arr.begin();}
    }

    // add an element to end of ring
    // precondition: begin != next
    void push(remove_qualifier_t<P> p, remove_qualifier_t<Types>... args) {
        *end = element_type{p, args...};
        end = next;
        if (++next == arr.end()) {next = arr.begin();}
    }
    
    array arr;
    iterator begin = arr.begin();
    iterator end = arr.begin();
    iterator next = end+1;
};


/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// First types for deduction in make_prefetch
// mode types that encapsulate 0 or 1
template<int v> class mode_type {};
//   constants: read or write
static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

// size types to encapsulate lookahead
template<std::size_t sz> struct size_type {};
//   constants: size<lookahead-n>
template<size_t sz>
static constexpr size_type<sz> size;

// and now the utility functions `make_prefetch`

// make_prefetch(
//    size_type<n-lookaheads>,
//    read|write,
//    [] (auto&& prefetch, auto&& params...) {},
//    ignored-variable-of-prefetch-type,
//   ignore-variables-of-params-types...
// )
template<std::size_t s, int m, typename F, typename P, typename... Types>
constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f, P, Types...) {
    return prefetch<s, m, F, P, Types...>{std::forward<F>(f)};
}

// make_prefetch<prefetch-type, param-types...>(
//    size_type<n-lookaheads>,
//    read|write,
//    [] (auto&& prefetch, auto&& params...) {}
// )
template<typename P, typename... Types, std::size_t s, int m, typename F>
constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f) {
    return prefetch<s, m, F, P, Types...>{std::forward<F>(f)};
}

// make_prefetch(
//    size_type<n-lookaheads>,
//    read|write,
//    [] (prefetch-type&&, param-types&&...) {}
// )
// first, we need to build traits to get the parameter types
namespace get_prefetch_functor_args {
// for functors
  template<typename, std::size_t, int, typename>
  struct functor_traits;

// pull off the functor argument types to construct prefetch
  template<typename F, std::size_t s, int m, typename T, typename P, typename... Types>
  struct functor_traits<F, s, m, void(T::*)(P, Types...) const>
  {
      using C = prefetch<s, m, F, P, Types...>;
      
      static constexpr auto make_prefetch(F f) {
          return C{std::forward<F>(f)};
      }
  };

// base type, assumes F is lambda or other functor, apply functor_traits
  template< typename F, std::size_t s, int m>
  using FTF = functor_traits<F, s, m, decltype(&F::operator())>;

  template<typename F, std::size_t s, int m>
  struct traits: public FTF<F, s, m>
  {};

// for function pointers: create a matching functor type for functor_traits
  template<typename F, std::size_t s, int m>
  using FTP = functor_traits<F, s, m, decltype(&std::function<F>::operator())>;

  template<std::size_t s, int m, typename P, typename... Types>
  struct traits<void(P, Types...), s, m>: public FTP<void(P, Types...), s, m>
  {};
} // get_prefetch_functor_args

// and here we apply the traits in make_prefetch
template<std::size_t s, int m, typename F>
constexpr auto make_prefetch(size_type<s>, mode_type<m>, F f) {
    return get_prefetch_functor_args::traits<F, s, m>::make_prefetch(std::forward<F>(f));
}

} //prefetch
} //arb
