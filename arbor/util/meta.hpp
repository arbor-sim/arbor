#pragma once

/* Type utilities and convenience expressions.  */

#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <type_traits>

namespace arb {
namespace util {

// Types associated with a container or sequence

namespace impl_seqtrait {
    using std::begin;
    using std::end;

    template <typename Seq, typename = void>
    struct data_returns_pointer: std::false_type {};

    template <typename T, std::size_t N>
    struct data_returns_pointer<T (&)[N], void>: public std::true_type {};

    template <typename T>
    struct data_returns_pointer<T, std::void_t<decltype(std::declval<T>().data())>>:
        public std::is_pointer<decltype(std::declval<T>().data())>::type {};

    template <typename Seq, typename=void>
    struct size_type_ {
        using type = void;
    };

    template <typename Seq>
    struct size_type_<Seq, std::void_t<decltype(std::size(std::declval<Seq&>()))>> {
        using type = decltype(std::size(std::declval<Seq&>()));
    };

    template <typename Seq>
    struct sequence_traits {
        using iterator = decltype(begin(std::declval<Seq&>()));
        using const_iterator = decltype(begin(std::declval<const Seq&>()));
        using value_type = typename std::iterator_traits<iterator>::value_type;
        using reference = typename std::iterator_traits<iterator>::reference;
        using difference_type = typename std::iterator_traits<iterator>::difference_type;
        using size_type = typename size_type_<Seq>::type;
        // For use with heterogeneous ranges:
        using sentinel = decltype(end(std::declval<Seq&>()));
        using const_sentinel = decltype(end(std::declval<const Seq&>()));

        static constexpr bool is_contiguous = data_returns_pointer<Seq>::value;
        static constexpr bool is_regular = std::is_same<iterator, sentinel>::value;
    };

    template<typename T, typename V=void>
    struct is_sequence:
        std::false_type {};

    template<typename T>
    struct is_sequence<T, std::void_t<decltype(begin(std::declval<T>()))>>:
        std::true_type {};

}

template <typename Seq>
using sequence_traits = impl_seqtrait::sequence_traits<Seq>;

// Sequence test by checking begin.

template <typename T>
using is_sequence = impl_seqtrait::is_sequence<T>;

template <typename T>
inline constexpr bool is_sequence_v = is_sequence<T>::value;

template <typename T>
using enable_if_sequence_t = std::enable_if_t<util::is_sequence<T>::value>;

template <typename T>
using is_contiguous = std::integral_constant<bool, sequence_traits<T>::is_contiguous>;

template <typename T>
inline constexpr bool is_contiguous_v = is_contiguous<T>::value;

template <typename T>
using is_regular_sequence = std::integral_constant<bool, sequence_traits<T>::is_regular>;

template <typename T>
inline constexpr bool is_regular_sequence_v = is_regular_sequence<T>::value;

// Convenience short cuts for `enable_if`

template <typename T>
using enable_if_copy_constructible_t =
    std::enable_if_t<std::is_copy_constructible<T>::value>;

template <typename T>
using enable_if_move_constructible_t =
    std::enable_if_t<std::is_move_constructible<T>::value>;

template <typename T>
using enable_if_default_constructible_t =
    std::enable_if_t<std::is_default_constructible<T>::value>;

template <typename... T>
using enable_if_constructible_t =
    std::enable_if_t<std::is_constructible<T...>::value>;

template <typename T>
using enable_if_copy_assignable_t =
    std::enable_if_t<std::is_copy_assignable<T>::value>;

template <typename T>
using enable_if_move_assignable_t =
    std::enable_if_t<std::is_move_assignable<T>::value>;

template <typename T>
using enable_if_trivially_copyable_t =
    std::enable_if_t<std::is_trivially_copyable<T>::value>;

// Iterator class test
// (might not be portable before C++17)

template <typename T, typename = void>
struct is_iterator: public std::false_type {};

template <typename T>
struct is_iterator<T, std::void_t<typename std::iterator_traits<T>::iterator_category>>:
    public std::true_type {};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

// Random access iterator test

template <typename T, typename = void>
struct is_random_access_iterator: public std::false_type {};

template <typename T>
struct is_random_access_iterator<T, std::enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
inline constexpr bool is_random_access_iterator_v = is_random_access_iterator<T>::value;

// Bidirectional iterator test

template <typename T, typename = void>
struct is_bidirectional_iterator: public std::false_type {};

template <typename T>
struct is_bidirectional_iterator<T, std::enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::bidirectional_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
inline constexpr bool is_bidirectional_iterator_v = is_bidirectional_iterator<T>::value;

// Forward iterator test

template <typename T, typename = void>
struct is_forward_iterator: public std::false_type {};

template <typename T>
struct is_forward_iterator<T, std::enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::bidirectional_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::forward_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
inline constexpr bool is_forward_iterator_v = is_forward_iterator<T>::value;

template <typename I, typename E, typename = void, typename = void>
struct common_random_access_iterator {};

template <typename I, typename E>
struct common_random_access_iterator<
    I,
    E,
    std::void_t<decltype(false? std::declval<I>(): std::declval<E>())>,
    std::enable_if_t<
        is_random_access_iterator<
            std::decay_t<decltype(false? std::declval<I>(): std::declval<E>())>
        >::value
    >
> {
    using type = std::decay_t<
        decltype(false ? std::declval<I>() : std::declval<E>())
    >;
};

template <typename I, typename E>
using common_random_access_iterator_t = typename common_random_access_iterator<I, E>::type;

template <typename I, typename E, typename V=void>
struct has_common_random_access_iterator:
    std::false_type {};

template <typename I, typename E>
struct has_common_random_access_iterator<I, E, std::void_t<util::common_random_access_iterator_t<I, E>>>:
    std::true_type {};

template <typename I, typename E>
inline constexpr bool has_common_random_access_iterator_v = has_common_random_access_iterator<I, E>::value;

// Generic accessors:
//    * first and second for pairs and tuples;
//    * util::get<I> to forward to std::get<I> where applicable, but
//      is otherwise extensible to non-std types.

static auto first = [](auto&& pair) -> decltype(auto) { return std::get<0>(std::forward<decltype(pair)>(pair)); };
static auto second = [](auto&& pair) -> decltype(auto) { return std::get<1>(std::forward<decltype(pair)>(pair)); };

} // namespace util
} // namespace arb
