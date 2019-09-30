#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

namespace arb {
namespace prefetch {

// Internal utility
template<typename T>
using remove_qualifier_t = std::remove_cv_t<std::remove_reference_t<T>>;

// like a tuple, just for packing/unpacking typename parameter packs
template<typename... Ts>
class pack {};

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
// but this implies that *prefetched is a valid operation
template<typename P>
auto get_pointer(P prefetched) {
    return &*prefetched;
}

// if it's a plain pointer, can even be invalid
template<typename P>
auto get_pointer(P* prefetched) {
    return prefetched;
}

/*
  prefetch<size_type<prefetch-size>, mode_type<prefetch-mode>, func-to-apply-type, payload-pointers-types...>
  construct with make_prefetch utilities below

  prefetches an address-like P,
  stores a list of argument addresses, 
  and later calls a function on them when full (or at destruction)

  the concept is that you continously `store` addresses from cuts through arrays until full
  (see `store()`). Prefetch is called on some associated address (by default the first
  `store` argument) before storing.

  When full, a functor is called on all the arguments in the hope that
  the prefetch has already pulled the data associated with the prefetch-pointer.

  void do_it(vec) {
    auto p = prefetch::make_prefetch(
               prefetch::size_type<n>,
               `prefetch::read` or `prefetch::write`,
               [] (args-pointer&& args...) {
                 p->do_something(std::move(args)...);
               });

    for (obj: vec) {
      p.store(obj.p, obj.args...);
    }
  }

  Prefetching is called before the tuple is constructed from the `store` arguments, where the
  last argument has prefetch::get_pointer applied giving an address to call
  on __builtin_prefetch. For iterators or pointers, prefetch::get_pointer applies &* to the object.

  The "hard bit" is determining the range of good sizes for the capacity.
  How many prefetch should we add before calling e.process?
  Not too many or we will be pushing things out of the cache
  and not too few or we'll hit the function application before
  the data has arrived

  prefetch_size s = number of read-aheads
  prefetch_mode m = read or write, do we prefetch with a read or write expection?
  Types... =  payload types for calling functions on (pointer-like objects) with qualifiers for f
  process is applied on f: f(Types...)
*/

// _prefetch is just to rename and recombine qualified types passed in
// versus types stripped to the base
template<std::size_t s, int m, typename F, typename RawTypes, typename CookedTypes>
class _prefetch;

// unique specialization to do the pattern matching
template<std::size_t s, int m, typename F, typename... RawTypes, typename... CookedTypes>
class _prefetch<s, m, F, pack<RawTypes...>, pack<CookedTypes...>> {
public:
    static constexpr auto look_ahead = s;
    static constexpr auto mode = m;

    using element_type = std::tuple<CookedTypes...>;
    using array = std::array<element_type, look_ahead+1>; // 1 sentinel element
    using iterator = typename array::iterator;
    
    using function_type = F;
    const function_type function;

    _prefetch(F&& f): function{std::move(f)} {}
    _prefetch(const F& f): function{f} {}
    ~_prefetch() {while (begin != end) {pop();}} // clear buffer on destruct

    _prefetch(_prefetch&&) = default; //needed < C++17
    _prefetch(const _prefetch&) = delete; 
    _prefetch& operator=(const _prefetch&) = delete;
    
    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P>
    void store(CookedTypes... args, P p) {
        prefetch(p); // do our fetch
        if (begin == next) {pop();} // pop if look ahead full
        push(args...); // add look ahead
    }

    // default missing prefetch to the first argument
    void store(CookedTypes... args) {
        store(args..., [](auto&& arg0, auto&&...) {return arg0;} (args...));
    }

private:
    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() {
        apply(); // process lookahead
        if (++begin == arr.end()) {begin = arr.begin();}
    }

    // add an element to end of ring
    // precondition: begin != next
    void push(CookedTypes... args) {
        *end = element_type{args...}; // store lookahead w/ prefetch
        end = next;
        if (++next == arr.end()) {next = arr.begin();}
    }

    template<typename P>
    static void prefetch(P p) { // the only thing we really want to do
        __builtin_prefetch(get_pointer(p), mode);
    }

    static constexpr auto element_size = std::tuple_size<element_type>();
    static constexpr auto indices = std::make_index_sequence<element_size>{};
    
    // precondition: begin != end
    // apply function f to *begin
    auto apply() {apply(indices);}

    // template match to get types
    template<std::size_t... I> // unpack indices for `get`
    auto apply(std::index_sequence<I...>) {// call f with correct types: possibly rvalues or const
        return function(get_raw_value<I>()...);
    }

    // get_raw_value type extraction
    template<std::size_t n> // get nth type
    using raw_type = std::tuple_element_t<n, std::tuple<RawTypes...>>;    

    // apply type extraction and cast nth element
    template<std::size_t n>
    auto get_raw_value() { // get value with right type from n
        return std::forward<raw_type<n>>(std::get<n>(*begin));
    }

    // array pointers
    array arr; // ring buffer storage using an extra sentinel element
    iterator begin = arr.begin(); // first element to pop off
    iterator end = arr.begin(); // next element to push into
    iterator next = end+1; // sentinel: next == begin, we're out of space
};

/* prefetch class proper: */

// First types for deduction in make_prefetch
// mode types that encapsulate 0 or 1
template<int v> class mode_type {};

//   constants: read or write
static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

// size types to encapsulate lookahead
template<std::size_t sz> struct size_type {};

//   constants: size<lookahead-n>
template<std::size_t sz>
static constexpr size_type<sz> size;

// forward declaration
template<typename S, typename M, typename F, typename P, typename ... Types>
class prefetch;

// and now pull off the values with a single specialization
template<std::size_t s, int m, typename F, typename P, typename ... Types>
class prefetch<size_type<s>, mode_type<m>, F, P, Types...>:
        public _prefetch<s, m, F,
                         pack<P, Types...>,
                         pack<remove_qualifier_t<P>, remove_qualifier_t<Types>...>>
{
public:
    using parent = _prefetch<s, m, F,
                             pack<P, Types...>,
                             pack<remove_qualifier_t<P>, remove_qualifier_t<Types>...>>;

    using parent::parent;
    using parent::store;
};


/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// and now the utility functions `make_prefetch`

// make_prefetch(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (auto&& prefetch, auto&& params...) {},
//    ignored-variable-of-prefetch-type,
//    ignore-variables-of-params-types...
// )
template<typename S, typename M, typename F, typename P, typename... Types>
constexpr auto make_prefetch(S, M, F&& f, P, Types...) {
    return prefetch<S, M, F, P, Types...>{std::forward<F>(f)};
}

// make_prefetch<prefetch-type, param-types...>(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (auto&& prefetch, auto&& params...) {}
// )
template<typename P, typename... Types, typename S, typename M, typename F>
constexpr auto make_prefetch(S, M, F&& f) {
    return prefetch<S, M, F, P, Types...>{std::forward<F>(f)};
}

// make_prefetch(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (prefetch-type&&, param-types&&...) {}
// )
// first, we need to build traits to get the parameter types
namespace get_prefetch_functor_args {
  // construct prefetch from passed in P, Types...
  template<typename F, typename P, typename... Types>
  struct _traits
  {
      template<typename S, typename M>
      static constexpr auto make_prefetch(F&& f) {
          return prefetch<S, M, F, P, Types...>{std::forward<F>(f)};
      }
  };

  // for functors
  template<typename, typename>
  struct functor_traits;

  // pull off the functor argument types to construct prefetch
  template<typename F, typename T, typename P, typename... Types>
  struct functor_traits<F, void(T::*)(P, Types...) const>:
        public _traits<F, P, Types...>
  {};

  // base type, assumes F is lambda or other functor,
  // apply functor_traits to pull of P, Types..
  template<typename F>
  struct traits: public functor_traits<F, decltype(&F::operator())>
  {};

  // for function pointers: pull P, Types... immediately
  template<typename P, typename... Types>
  struct traits<void(P, Types...)>:
      public _traits<void(P, Types...), P, Types...>
  {};
} // get_prefetch_functor_args

// and here we apply the traits in make_prefetch
template<typename S, typename M, typename F>
constexpr auto make_prefetch(S, M, F&& f) {
    return get_prefetch_functor_args::traits<F>::template make_prefetch<S, M>(std::forward<F>(f));
}

} //prefetch
} //arb
