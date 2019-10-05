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
struct ring_buffer_types {
    static constexpr auto size = s;
    using element_type = E;

    template<typename T>
    using enable_if_element_t = std::enable_if_t<std::is_same<element_type, std::decay_t<T>>::value>;
};

template<typename U>
using is_not_trivially_destructible_t = typename std::enable_if_t<!std::is_trivially_destructible<U>::value, int>;

template<typename U>
using is_trivially_destructible_t = typename std::enable_if_t<std::is_trivially_destructible<U>::value, int>;

// requirement: s > 0, sizeof(E) > 0
template<std::size_t s, typename E>
class ring_buffer: public ring_buffer_types<s, E>
{
public:
    using types = ring_buffer_types<s, E>;
    using types::size;
    using typename types::element_type;
    
    template<typename T>
    using enable_if_element_t = typename types::template enable_if_element_t<T>;

    ring_buffer()  noexcept = default;
    ~ring_buffer() noexcept {deconstruct();} // if needed, kill elements

    // uncopyable
    ring_buffer(ring_buffer&&) = delete;
    ring_buffer(const ring_buffer&) = delete;
    ring_buffer& operator=(ring_buffer&&) = delete;
    ring_buffer& operator=(const ring_buffer&) = delete;

    // precondition: ! is_full()
    template<typename T, typename = enable_if_element_t<T>>
    void push(T&& e) noexcept {
        push_emplace(std::forward<T>(e));
    }

    // precondition: ! is_full()
    template<typename... Ts>
    void push_emplace(Ts&&... args) noexcept {
        new (stop) element_type{std::forward<Ts>(args)...};
        stop = next;
        if (++next == end) {next = begin;}
    }

    // precondition: ! is_empty()
    element_type& pop() noexcept {
        invalidate(); // if popped element alive, deconstruct if needed
        const auto popped = start;
        if (++start == end) {start = begin;}
        return *popped; // only valid until next pop happens
    }

    bool empty() const noexcept {return start == stop;}
    bool full()  const noexcept {return start == next;}

private:
    // do nothing on deconstruct for trivially destructible element_type
    template<typename U = E, is_trivially_destructible_t<U> = 0>
    void deconstruct() noexcept {}
    template<typename U = E, is_trivially_destructible_t<U> = 0>
    void invalidate() noexcept {}

    // otherwise, must handle deconstructions:
    // deconstruct all elements, if not trivially destructible
    template<typename U = E, is_not_trivially_destructible_t<U> = 0>
    void deconstruct() noexcept {
        while (valid != stop) {
            valid->~element_type();
            if (++valid == end) {valid = begin;}
        }
    }
    
    // deconstruct last popped off, if not trivially destructible
    template<typename U = E, is_not_trivially_destructible_t<U> = 0>
    void invalidate() noexcept {
        if (valid != start) {
            valid->~element_type();
            valid = start;
        }
    }

    // ring buffer storage using an extra sentinel element
    alignas(element_type) char array[sizeof(element_type)*(size+1)];    
    using iterator = element_type*;
    const iterator begin = reinterpret_cast<iterator>(array);
    const iterator end   = begin + size + 1;

    // array pointers
    iterator start = begin;  // first element to pop off
    iterator valid = begin;  // last valid element, at most one behind start
    iterator stop  = begin;  // next element to push into
    iterator next  = stop+1; // sentinel: next == start, we're out of space
};

template<typename E>
class ring_buffer<0, E>: public ring_buffer_types<0, E>
{
public:
    using types = ring_buffer_types<0, E>;
    using types::size;
    using typename types::element_type;
    template<typename T>
    using enable_if_element_t = typename types::template enable_if_element_t<T>;
};

template<std::size_t s, typename... Ts>
using buffer = ring_buffer<s, std::tuple<Ts...>>;

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
inline void fetch(P&& p) noexcept {
    __builtin_prefetch(get_pointer(std::forward<P>(p)), mode);
};

template<int mode, typename P>
inline void fetch(mode_type<mode>, P&& p) noexcept {
    fetch<mode>(std::forward<P>(p));
};

template<int m, typename B, typename F>
struct prefetch_types
{
    using buffer = B;
    using function_type = F;    
    static constexpr auto mode = m;
    
    static constexpr auto size = buffer::size;
    using element_type = typename buffer::element_type;

    template<typename E>
    using enable_if_element_t = typename buffer::template enable_if_element_t<E>;
};

template<int m, typename B, typename F, std::size_t>
class _prefetch;

template<int m, typename B, typename F>
class prefetch;

// prefetch is just to rename and recombine qualified types passed in
// versus types stripped to the base
template<int m, typename B, typename F, std::size_t>
class _prefetch: public prefetch_types<m, B, F> {
public:
    using types = prefetch_types<m, B, F>;
    using typename types::buffer;
    using types::size;
    using types::mode;
    using typename types::element_type;
    using typename types::function_type;

    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;

    _prefetch(buffer& b_, function_type&& f) noexcept: b(b_), function{std::move(f)} {}
    _prefetch(buffer& b_, const function_type& f) noexcept: b(b_), function{f} {}
    ~_prefetch() {while (! b.empty()) {pop();}} // clear buffer on destruct

    _prefetch(_prefetch&&) = default; //needed < C++17 for make_prefetch
    _prefetch(const _prefetch&) = delete;
    _prefetch& operator=(_prefetch&&) = delete;
    _prefetch& operator=(const _prefetch&) = delete;

    // allow element_type to be constructed in place
    template<typename P>
    void store(P&& p, const element_type& e) {
        store_internal(std::forward<P>(p), e);
    }

    template<typename P>
    void store(P&& p, element_type&& e) {
        store_internal(std::forward<P>(p), std::move(e));
    }

    template<typename P, typename... Ts>
    void store(P&& p, Ts&&... args) {
        store_internal(std::forward<P>(p), std::forward<Ts>(args)...);
    }

private:    
    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P, typename... Ts>
    void store_internal(P&& p, Ts&&... args) {
        fetch<mode>(std::forward<P>(p));
        if (b.full()) {pop();} // process and remove if look ahead full
        push(std::forward<Ts>(args)...); // add new look ahead
    }

    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() {
        function(b.pop());
    }

    // add an element to end of ring
    // precondition: begin != next
    template<typename E, typename = enable_if_element_t<E>>
    void push(E&& e) noexcept {
        b.push(std::forward<E>(e));
    }

    template<typename... Ts>
    void push(Ts&&... args) {
        b.push_emplace(std::forward<Ts>(args)...);
    }

    buffer& b;
    const function_type function;
};

template<int m, std::size_t s, typename... Ts, typename F>
class prefetch<m, buffer<s, Ts...>, F>: public _prefetch<m, buffer<s, Ts...>, F, s>
{
public:
    using parent = _prefetch<m, buffer<s, Ts...>, F, s>;
    using parent::parent;
};

// specialization to turn off prefetch
template<int m, typename B, typename F>
class _prefetch<m, B, F, 0>: public prefetch_types<m, B, F> {
public:
    using types = prefetch_types<m, B, F>;
    using typename types::buffer;
    using types::size;
    using types::mode;
    using typename types::element_type;
    using typename types::function_type;

    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;

    _prefetch(buffer&, function_type&& f) noexcept: function{std::move(f)} {}
    _prefetch(buffer&, const function_type& f) noexcept: function{f} {}

    _prefetch(_prefetch&&) = default; //needed < C++17 for make_prefetch
    _prefetch(const _prefetch&) = delete;
    _prefetch& operator=(_prefetch&&) = delete;
    _prefetch& operator=(const _prefetch&) = delete;

    template<typename P>
    void store(P&&, const element_type& e) {
        store_internal(e);
    }

    template<typename P>
    void store(P&&, element_type&& e) {
        store_internal(std::move(e));
    }

    template<typename P, typename... Ts>
    void store(P&& p, Ts&&... args) {
        store_internal(element_type{std::forward<Ts>(args)...});
    }
    
private:
    template<typename E, typename = enable_if_element_t<E>>
    void store_internal(E&& e) {
        function(std::forward<E>(e));
    }
    const function_type function;
};

template<int m, typename... Ts, typename F>
class prefetch<m, buffer<0, Ts...>, F>: public _prefetch<m, buffer<0, Ts...>, F, 0>
{
public:
    using parent = _prefetch<m, buffer<0, Ts...>, F, 0>;
    using parent::parent;
};

/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// and now the utility functions `make_prefetch`

// make_prefetch(
//    prefetch::read|write,
//    buffer,
//    [] (auto&& element) {}
// )
template<int m, typename B, typename F>
inline constexpr auto make_prefetch(mode_type<m>, B& b, F&& f) noexcept {
    return prefetch<m, B, F>{b, std::forward<F>(f)};
}

} //prefetch
} //arb
