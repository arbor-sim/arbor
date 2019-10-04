#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

namespace arb {
namespace prefetch {

/*
  two class: buffer && prefetch:

  buffer<std::size_t s, typename.. Ts>
  which is a ring buffer of that stores up to 
  s tuples of type Ts...

  and

  prefetch<int mode, buffer, function>
  which binds the buffer within scope with a function
  and prefetches every store with mode.

  a buffer is declared in an outer, permanent scope:
  buffer<n, Args...> b;

  and then:
  void doit() {
    auto&& a = make_prefetch(
      prefetch::read or prefetch::write,
      b,
      [] (auto&& element) {
        do-something(element);
      });

      // where element is a tuple<Args...>
      for (obj: vec) {
        a.store(obj.pointer, {obj.pointer, obj.arg, ...});
      }
      // and {obj.pointer, obj.arg, ...} constructs tupe<Args...>
  }
  
  prefetches an address-like P,
  stores a list of argument addresses, 
  and later calls a function on them when full (or at destruction)

  the concept is that you continously `store` addresses from cuts through arrays until full
  (see `store()`). Prefetch is called on some associated address (by default the first
  `store` argument) before storing.

  When full, a functor is called on all the arguments in the hope that
  the prefetch has already pulled the data associated with the prefetch-pointer.


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

// ring_buffer: E should have trivial constructor, destructor
template<std::size_t s, typename E>
class ring_buffer
{
public:
    static constexpr auto size = s;
    using element_type = E;

    template<typename T>
    using enable_if_element_t = std::enable_if_t<std::is_same<element_type, std::decay_t<T>>::value>;

    // precondition: ! is_full()
    template<typename T, typename = enable_if_element_t<T>>
    void push(T&& e) noexcept {
        *end = {std::forward<T>(e)};
        end = next;
        if (++next == arr.end()) {next = arr.begin();}
    }

    // precondition: ! is_empty()
    element_type& pop() noexcept {
        auto head = begin;
        if (++begin == arr.end()) {begin = arr.begin();}
        return *head; // move out head, its now invalid
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

template<typename E>
class ring_buffer<0, E>
{
public:
    static constexpr auto size = 0;
    using element_type = E;
};

template<std::size_t s, typename... Ts>
class buffer: public ring_buffer<s, std::tuple<Ts...>>
{};

// prefetching wrapper
// First types for deduction in make_prefetch
// mode types that encapsulate 0 or 1
template<int v> struct mode_type {
    static constexpr auto value = v;
    constexpr operator int() const {return value;}
};

//   constants: read or write
static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
// but this implies that *prefetched is a valid operation
template<typename P>
inline auto get_pointer(P&& prefetched) noexcept {
    return &*std::forward<P>(prefetched);
}

// if it's a plain pointer, can even be invalid
template<typename P>
inline auto get_pointer(P* prefetched) noexcept {
    return prefetched;
}

// encapsulate __builtin_prefetch
// uses get_pointer to convert pointer-like to pointer
template<int mode, typename P> // do the prefetch
static inline void fetch(P&& p) noexcept {
    __builtin_prefetch(get_pointer(std::forward<P>(p)), mode);
};

template<int mode, typename P>
static inline void fetch(mode_type<mode>, P&& p) noexcept {
    fetch<mode>(std::forward<P>(p));
};

template<typename F, typename T, std::size_t... I>
static inline auto apply_tuple_internal(F&& f, T&& t, std::index_sequence<I...>) {
    std::forward<F>(f)(std::get<I>(std::forward<T>(t))...);
}

template<typename F, typename T>
static inline auto apply_tuple(F&& f, T&& t) {
    return apply_tuple_internal(std::forward<F>(f), std::forward<T>(t), std::make_index_sequence<std::tuple_size<T>::value>());
}

template<int m, typename B, typename F>
struct prefetch_types
{
    using array = B;
    using function_type = F;    
    static constexpr auto mode = m;
    
    static constexpr auto size = array::size;
    using element_type = typename array::element_type;

    template<typename E>
    using enable_if_element_t = typename array::template enable_if_element_t<E>;
};

template<int m, typename B, typename F>
class prefetch;

// prefetch is just to rename and recombine qualified types passed in
// versus types stripped to the base
template<int m, std::size_t s, typename... Ts, typename F>
class prefetch<m, buffer<s, Ts...>, F>: public prefetch_types<m, buffer<s, Ts...>, F> {
public:
    using types = prefetch_types<m, buffer<s, Ts...>, F>;
    using typename types::array;
    using types::size;
    using types::mode;
    using typename types::element_type;
    using typename types::function_type;

    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;

    prefetch(array& arr_, function_type&& f) noexcept: arr(arr_), function{std::move(f)} {}
    prefetch(array& arr_, const function_type& f) noexcept: arr(arr_), function{f} {}
    ~prefetch() {while (! arr.empty()) {pop();}} // clear buffer on destruct

    prefetch(prefetch&&) = default; //needed < C++17
    prefetch(const prefetch&) = delete; 
    prefetch& operator=(const prefetch&) = delete;

    // default missing prefetch to the first argument
    template<typename P>
    void store(P&& p, const element_type& e) {
        store_internal(std::forward<P>(p), e);
    }

    template<typename P>
    void store(P&& p, element_type&& e) {
        store_internal(std::forward<P>(p), std::move(e));
    }

private:
    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P, typename E, typename = enable_if_element_t<E>>
    void store_internal(P&& p, E&& e) {
        fetch<mode>(std::forward<P>(p));
        if (arr.full()) {pop();} // process and remove if look ahead full
        push(std::forward<E>(e)); // add new look ahead
    }


    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() {
        function(arr.pop());
    }

    // add an element to end of ring
    // precondition: begin != next
    template<typename E, typename = enable_if_element_t<E>>
    void push(E&& e) noexcept {
        arr.push(std::forward<E>(e));
    }

    array& arr;
    const function_type function;
};

// specialization to turn off prefetch
template<int m, typename... Ts, typename F>
class prefetch<m, buffer<0, Ts...>, F>: public prefetch_types<m, buffer<0, Ts...>, F> {
public:
    using types = prefetch_types<m, buffer<0, Ts...>, F>;
    using typename types::array;
    using types::size;
    using types::mode;
    using typename types::element_type;
    using typename types::function_type;

    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;

    prefetch(array&, function_type&& f) noexcept: function{std::move(f)} {}
    prefetch(array&, const function_type& f) noexcept: function{f} {}

    prefetch(prefetch&&) = default; //needed < C++17
    prefetch(const prefetch&) = delete;
    prefetch& operator=(const prefetch&) = delete;

    template<typename P, typename E, typename = enable_if_element_t<E>>
    void store(P&& p, E&& e) {
        apply_tuple(function, std::forward<E>(e));
    }

    template<typename E, typename = enable_if_element_t<E>>
    void store(E&& e) {
        apply_tuple(function, std::forward<E>(e));
    }

private:
    const function_type function;
};

/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// and now the utility functions `make_prefetch`

// make_prefetch(
//    prefetch::read|write,
//    buffer,
//    [] (auto&& param, auto&& params...) {}
// )
template<int m, typename B, typename F>
inline constexpr auto make_prefetch(mode_type<m>, B& b, F&& f) noexcept {
    return prefetch<m, B, F>{b, std::forward<F>(f)};
}

} //prefetch
} //arb
