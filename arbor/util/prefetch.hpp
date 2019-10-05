#pragma once

#include <array>
#include <type_traits>
#include <util/range.hpp>

namespace arb {
namespace prefetch {
// Utilities

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
    __builtin_prefetch(get_pointer(std::forward<P>(p)), mode, 3);
};

// call fetch with mode_type argument
template<int mode, typename P>
inline void fetch(mode_type<mode>, P&& p) noexcept {
    fetch<mode>(std::forward<P>(p));
};

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

// requirement: s > 0, sizeof(E) > 0
template<std::size_t s, typename E>
class ring_buffer: public ring_buffer_types<s, E>
{
public:
    using types = ring_buffer_types<s, E>;
    using typename types::element_type;
    using types::size;
    
    template<typename T>
    using enable_if_element_t = typename types::template enable_if_element_t<T>;

    ring_buffer() = default;
    ring_buffer(ring_buffer&&) = delete;
    ring_buffer(const ring_buffer&) = delete;
    ring_buffer& operator=(ring_buffer&&) = delete;
    ring_buffer& operator=(const ring_buffer&) = delete;

    ~ring_buffer() {deconstruct();} // if needed, kill elements

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
        const auto head = start;
        if (++start == end) {start = begin;}
        return *head; // only valid until next pop happens
    }

    bool empty() const noexcept {return start == stop;}
    bool full()  const noexcept {return start == next;}

private:
    template<typename U>
    using needs_destruct_t = typename std::enable_if_t<!std::is_trivially_destructible<U>::value, int>;

    template<typename U>
    using no_destruct_t = typename std::enable_if_t<std::is_trivially_destructible<U>::value, int>;

    // deconstruct all elements, if not trivially destructible
    template<typename U = element_type, needs_destruct_t<U> = 0>
    void deconstruct() noexcept {
        while (valid != stop) {
            valid->~element_type();
            if (++valid == end) {valid = begin;}
        }
    }

    // else do nothing
    template<typename U = element_type, no_destruct_t<U> = 0>
    void deconstruct() noexcept {}

    // deconstruct last popped off, if not trivially destructible
    template<typename U = element_type, needs_destruct_t<U> = 0>
    void invalidate() noexcept {
        if (valid != start) {
            valid->~element_type();
            valid = start;
        }
    }

    // else do nothing
    template<typename U = element_type, no_destruct_t<U> = 0>
    void invalidate() noexcept {}

    // ring buffer storage using an extra sentinel element
    alignas(element_type) char array[sizeof(element_type)*(size+1)];    
    typedef element_type* iterator;
    const iterator begin = reinterpret_cast<iterator>(array);
    const iterator end   = begin + size + 1;

    // array pointers
    iterator start = begin;  // first element to pop off
    iterator valid = begin;  // last valid element, at most one behind start
    iterator stop  = begin;  // next element to push into
    iterator next  = stop+1; // sentinel: next == begin, we're out of space
};

template<typename E>
class ring_buffer<0, E>: public ring_buffer_types<0, E>
{};

template<std::size_t s, typename... Ts>
using buffer = ring_buffer<s, std::tuple<Ts...>>;

template<int m, typename B, typename F>
struct prefetch_types
{
    using buffer_type = B;
    using function_type = F;    
    static constexpr auto mode = m;
    
    static constexpr auto size = buffer_type::size;
    using element_type = typename buffer_type::element_type;

    template<typename E>
    using enable_if_element_t = typename buffer_type::template enable_if_element_t<E>;
};

template<int m, typename B, typename F>
class prefetch_base: public prefetch_types<m, B, F> {
public:
    using types = prefetch_types<m, B, F>;
    using types::mode;
    using typename types::buffer_type;
    using typename types::function_type;

    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;

    prefetch_base(buffer_type& b_, function_type&& f) noexcept: b(b_), function{std::move(f)} {}
    prefetch_base(buffer_type& b_, const function_type& f) noexcept: b(b_), function{f} {}

    ~prefetch_base() noexcept { // clear buffer on destruct
        while (! b.empty()) {pop();}
    }

protected:    
    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P, typename... Ts>
    void store_internal(P&& p, Ts&&... args) noexcept {
        fetch<mode>(std::forward<P>(p));
        if (b.full()) {pop();} // process and remove if look ahead full
        push(std::forward<Ts>(args)...); // add new look ahead
    }

private:
    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() noexcept {
        function(b.pop());
    }

    // add an element to end of ring
    // precondition: begin != next
    template<typename E, typename = enable_if_element_t<E>>
    void push(E&& e) noexcept {
        b.push(std::forward<E>(e));
    }

    template<typename... Ts>
    void push(Ts&&... args) noexcept {
        b.push_emplace(std::forward<Ts>(args)...);
    }

    buffer_type& b;
    const function_type function;
};

template<int m, typename B, typename F>
class prefetch_base_zero: public prefetch_types<m, B, F>
{
public:
    using types = prefetch_types<m, B, F>;
    using typename types::buffer_type;
    using typename types::element_type;
    using typename types::function_type;

    prefetch_base_zero(buffer_type&, function_type&& f) noexcept: function{std::move(f)} {}
    prefetch_base_zero(buffer_type&, const function_type& f) noexcept: function{f} {}
    
protected:
    template<typename P, typename... Ts>
    void store_internal(P&&, Ts&&... args) noexcept {
        function(element_type{std::forward<Ts>(args)...});
    }

private:
    const function_type function;
};

// specialization to turn off prefetch
template<int m, typename... Ts, typename F>
class prefetch_base<m, buffer<0, Ts...>, F>: public prefetch_base_zero<m, buffer<0, Ts...>, F>
{
public:
    using base = prefetch_base_zero<m, buffer<0, Ts...>, F>;
    using base::base;
};

// prefetch is just to rename and recombine qualified types passed in
// versus types stripped to the base
template<int m, typename B, typename F>
class prefetch: public prefetch_base<m, B, F> {
public:
    using base = prefetch_base<m, B, F>;
    using typename base::element_type;
    using base::store_internal;

    using base::base;
    prefetch(prefetch&&) noexcept = default;

    prefetch(const prefetch&) = delete;
    prefetch& operator=(prefetch&&) = delete;
    prefetch& operator=(const prefetch&) = delete;

    // allow element_type to be constructed in place
    template<typename P>
    void store(P&& p, const element_type& e) noexcept {
        store_internal(std::forward<P>(p), e);
    }

    template<typename P>
    void store(P&& p, element_type&& e) noexcept {
        store_internal(std::forward<P>(p), std::move(e));
    }

    template<typename P, typename... Ts>
    void store(P&& p, Ts&&... args) noexcept {
        store_internal(std::forward<P>(p), std::forward<Ts>(args)...);
    }
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
