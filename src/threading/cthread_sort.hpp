// parallel stable sort uses threading
#include "cthread_parallel_stable_sort.h"

namespace nest {
namespace mc {
namespace threading {

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    pss::parallel_stable_sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    pss::parallel_stable_sort(begin, end ,comp);
}

template <typename Container>
void sort(Container& c) {
    pss::parallel_stable_sort(c.begin(), c.end());
}


}
}
}
