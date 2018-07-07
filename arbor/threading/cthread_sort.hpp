#pragma once

namespace arb {
namespace threading {
inline namespace cthread {

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    std::sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    std::sort(begin, end, comp);
}

template <typename Container>
void sort(Container& c) {
    std::sort(std::begin(c), std::end(c));
}

} // namespace cthread
} // namespace threading
} // namespace arb
