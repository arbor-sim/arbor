#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

namespace arb {
namespace prefetch {

// Internal utility

// like a tuple, just for packing/unpacking typename parameter packs
template<typename... Ts>
class pack {};

// match packs of arguments
template<typename T1s, typename T2s, typename T = void>
struct enable_if_args_match;

template<typename... T1s, typename... T2s, typename T>
struct enable_if_args_match<pack<T1s...>, pack<T2s...>, T>
    : public std::enable_if<std::is_same<
                                std::tuple<std::decay_t<T1s>...>,
                                std::tuple<std::decay_t<T2s>...>>::value,
                            T>
{};

template<typename T1s, typename T2s>
using enable_if_args_match_t = typename enable_if_args_match<T1s, T2s>::type;

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

// pull out `apply` code from prefetch 
template<typename... RawTypes>
class apply_raw {
public:
    // apply function f to t
    template<typename F, typename T>
    static auto apply(F&& f, T&& t) {return apply(std::forward<F>(f), std::forward<T>(t), indices);}

private:
    using raw_tuple = std::tuple<RawTypes...>;
    static constexpr auto size = std::tuple_size<raw_tuple>();
    static constexpr auto indices = std::make_index_sequence<size>{};
    
    // template match to get types
    template<typename F, typename T, std::size_t... I> // unpack indices for `get`
    static auto apply(F&& f, T&& t, std::index_sequence<I...>) {// call f with correct types: possibly rvalues or const
        return std::forward<F>(f)(get_raw_value<I>(std::forward<T>(t))...);
    }

    // get_raw_value type extraction
    template<std::size_t n> // get nth type
    using raw_type = std::tuple_element_t<n, raw_tuple>;

    // apply type extraction and cast nth element
    template<std::size_t n, typename T>
    static auto get_raw_value(T&& element) noexcept { // get value with right type from n
        return static_cast<raw_type<n>>(std::get<n>(std::forward<T>(element)));
    }
};

// ring_buffer: E should have trivial constructor, destructor
template<std::size_t s, typename E>
class ring_buffer
{
public:
    static constexpr auto size = s;
    using element_type = E;

    // precondition: ! is_full()
    template<typename T, typename = enable_if_args_match_t<pack<E>, pack<T>>>
    void push(T&& e) noexcept {
        *end = {std::forward<T>(e)};
        end = next;
        if (++next == arr.end()) {next = arr.begin();}
    }

    // precondition: ! is_empty()
    element_type&& pop() noexcept {
        auto head = begin;
        if (++begin == arr.end()) {begin = arr.begin();}
        return std::move(*head); // move out head, its now invalid
    }

    bool empty() const noexcept {return begin == end;}
    bool full()  const noexcept {return begin == next;}

private:
    using array = std::array<element_type, size+1>; // extra sentinel elem
    using iterator = typename array::iterator;

    // array pointers
    array arr; // ring buffer storage using an extra sentinel element
    iterator begin = arr.begin(); // first element to pop off
    iterator end   = begin; // next element to push into
    iterator next  = end+1; // sentinel: next == begin, we're out of space
};

// prefetching wrapper
// First types for deduction in make_prefetch
// mode types that encapsulate 0 or 1
template<int v> class mode_type {};

//   constants: read or write
static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
// but this implies that *prefetched is a valid operation
template<typename P>
auto get_pointer(P&& prefetched) noexcept {
    return &*std::forward<P>(prefetched);
}

// if it's a plain pointer, can even be invalid
template<typename P>
auto get_pointer(P* prefetched) noexcept {
    return prefetched;
}

// encapsulate __builtin_prefetch
// trait holds M mode_type<m>
// uses get_pointer to convert pointer-like to pointer
template<typename M>
struct prefetch_type;

template<int m>
struct prefetch_type<mode_type<m>> {
    static constexpr auto mode = m;
    
    template<typename P>
    static void apply(P&& p) noexcept { // do the prefetch
        __builtin_prefetch(get_pointer(std::forward<P>(p)), mode);
    }
};

// _prefetch is just to rename and recombine qualified types passed in
// versus types stripped to the base
template<std::size_t s, int m, typename F, typename RawTypes, typename CookedTypes>
class _prefetch;

// unique specialization to do the pattern matching
template<std::size_t s, int m, typename F, typename... RawTypes, typename... CookedTypes>
class _prefetch<s, m, F, pack<RawTypes...>, pack<CookedTypes...>> {
public:
    static constexpr auto size = s;
    static constexpr auto mode = m;

    using element_type = std::tuple<CookedTypes...>;
    using array = ring_buffer<size, element_type>;    
    using function_type = F;
    using fetch = prefetch_type<mode_type<m>>;

    _prefetch(F&& f) noexcept: function{std::move(f)} {}
    _prefetch(const F& f) noexcept: function{f} {}
    ~_prefetch() {while (! arr.empty()) {pop();}} // clear buffer on destruct

    _prefetch(_prefetch&&) = default; //needed < C++17
    _prefetch(const _prefetch&) = delete; 
    _prefetch& operator=(const _prefetch&) = delete;

    template<typename... Ts>
    using enable_if_cooked_args_t = enable_if_args_match_t<pack<Ts...>, pack<CookedTypes...>>;

    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P, typename... Ts, typename = enable_if_cooked_args_t<Ts...>>
    void store(P&& p, Ts&&... args) {
        fetch::apply(std::forward<P>(p));
        if (arr.full()) {pop();} // process and remove if look ahead full
        push(std::forward<Ts>(args)...); // add new look ahead
    }

    // default missing prefetch to the first argument
    template<typename... Ts, typename = enable_if_cooked_args_t<Ts...>>
    void store(Ts&&... args) {
        store([](auto&& arg0, auto&&...) {return arg0;} (std::forward<Ts>(args)...),
              std::forward<Ts>(args)...);
    }

private:
    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() {
        apply_raw<RawTypes...>::apply(function, arr.pop());
    }

    // add an element to end of ring
    // precondition: begin != next
    template<typename... Ts, typename = enable_if_cooked_args_t<Ts...>>
    void push(Ts&&... args) noexcept {
        arr.push(element_type{std::forward<Ts>(args)...});
    }

    array arr;
    const function_type function;
};

/* prefetch class proper: */

// size types to encapsulate lookahead
template<std::size_t sz> struct size_type {};

//   constants: size<lookahead-n>
template<std::size_t sz>
static constexpr size_type<sz> size;

// forward declaration
template<typename S, typename M, typename F, typename... Types>
class prefetch;

// and now pull off the values with a single specialization
template<std::size_t s, int m, typename F, typename... Types>
class prefetch<size_type<s>, mode_type<m>, F, Types...>:
        public _prefetch<s, m, F,
                         pack<Types...>,
                         pack<std::decay_t<Types>...>>
{
public:
    using parent = _prefetch<s, m, F,
                             pack<Types...>,
                             pack<std::decay_t<Types>...>>;

    using parent::parent;
};

/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// and now the utility functions `make_prefetch`

// make_prefetch(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (auto&& param, auto&& params...) {},
//    ignored-variable-of-param-type,
//    ignore-variables-of-params-types...
// )
template<typename S, typename M, typename F, typename Type, typename... Types>
constexpr auto make_prefetch(S, M, F&& f, Type, Types...) noexcept {
    return prefetch<S, M, F, Type, Types...>{std::forward<F>(f)};
}

// make_prefetch<param-type, param-types...>(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (auto&& param, auto&& params...) {}
// )
template<typename Type, typename... Types, typename S, typename M, typename F>
constexpr auto make_prefetch(S, M, F&& f) noexcept {
    return prefetch<S, M, F, Type, Types...>{std::forward<F>(f)};
}

// make_prefetch(
//    prefetch::size_type<n-lookaheads>,
//    prefetch::read|write,
//    [] (param-types&&...) {}
// )
// first, we need to build traits to get the parameter types
namespace get_prefetch_functor_args {
  // construct prefetch from passed in P, Types...
  template<typename F, typename... Types>
  struct _traits
  {
      template<typename S, typename M>
      static constexpr auto make_prefetch(F&& f) noexcept {
          return prefetch<S, M, F, Types...>{std::forward<F>(f)};
      }
  };

  // for functors
  template<typename, typename>
  struct functor_traits;

  // pull off the functor argument types to construct prefetch
  template<typename F, typename T, typename... Types>
  struct functor_traits<F, void(T::*)(Types...) const>:
        public _traits<F, Types...>
  {};

  // base type, assumes F is lambda or other functor,
  // apply functor_traits to pull of Types...
  template<typename F>
  struct traits: public functor_traits<F, decltype(&F::operator())>
  {};

  // for function pointers: pull Types... immediately
  template<typename... Types>
  struct traits<void(Types...)>:
      public _traits<void(Types...), Types...>
  {};
} // get_prefetch_functor_args

// and here we apply the traits in make_prefetch
template<typename S, typename M, typename F>
constexpr auto make_prefetch(S, M, F&& f) noexcept {
    return get_prefetch_functor_args::traits<F>::template make_prefetch<S, M>(std::forward<F>(f));
}

} //prefetch
} //arb
