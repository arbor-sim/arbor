#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& vec) {
    for (auto& v: vec) o << " " << v;
    return o;
}

template <typename U, typename V>
std::ostream& operator<<(std::ostream& o, const std::pair<U, V>& pr) {
    return o << "<" << pr.first << ", " << pr.second << ">";
}

std::vector<int> kway_dumb(const std::vector<std::vector<int>>& lanes) {
    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    std::vector<int> m;
    m.reserve(n);

    for (auto& l: lanes) {
        m.insert(m.begin(), std::begin(l), std::end(l));
    }
    std::sort(m.begin(), m.end());

    return m;
}

std::vector<int> kway_naive(const std::vector<std::vector<int>>& lanes) {
    const unsigned k = lanes.size();
    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    std::vector<int> m;
    m.reserve(n);

    using p = std::pair<unsigned, std::vector<int>::const_iterator>;
    std::vector<p> tops;
    tops.reserve(k);
    for (auto i=0u; i<k; ++i) {
        auto& l = lanes[i];
        if (l.size()) {
            tops.push_back({i, l.begin()});
        }
    }

    while (!tops.empty()) {
        auto it = std::min_element(
                tops.begin(), tops.end(),
                [](p l, p r) {return *(l.second)<*(r.second);});
        m.push_back(*(it->second));
        ++(it->second);
        if (it->second==lanes[it->first].end()) {
            tops.erase(it);
        }
    }

    return m;
}

std::vector<int> kway_heap(const std::vector<std::vector<int>>& lanes) {
    using p = std::pair<int, unsigned>;
    auto op = [](p l, p r) {return l.first>r.first;};

    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    const unsigned k = lanes.size();

    if (lanes.size()==1) return lanes[0];

    std::vector<p> heap;

    std::vector<int> m;
    m.reserve(n);

    // Build heap with first entry in each lane, and make list of iterators that
    // tracks the entry in each lane that is currently in the queue.
    std::vector<std::vector<int>::const_iterator> tops;
    for (auto& l: lanes) {
        tops.push_back(l.begin());
        if (l.size()) {
            heap.push_back({l.front(), tops.size()-1});
            ++tops.back();
        }
    }
    std::make_heap(heap.begin(), heap.end(), op);

    while (!heap.empty()) {
        const auto l = heap.front().second;
        m.push_back(heap.front().first);
        std::pop_heap(heap.begin(), heap.end(), op);
        heap.pop_back();
        if (tops[l]!=lanes[l].end()) {
            heap.push_back({*tops[l], l});
            std::push_heap(heap.begin(), heap.end(), op);
            ++tops[l];
        }
    }

    return m;
}

template <typename F>
void test(unsigned num_lanes, unsigned n, F f) {
    std::vector<std::vector<int>> lanes(num_lanes);

    auto seed = 0u;
    for (auto& l: lanes) {
        l.reserve(n);
        std::mt19937 gen(seed++);
        std::uniform_int_distribution<> dis(1, 100);
        for (auto i=0u; i<n; ++i) {
            l.push_back(dis(gen));
        }
        std::sort(l.begin(), l.end());
    }

    std::cout << f(lanes) << "\n";
}

int main() {
    test(4, 100, kway_dumb);
    test(4, 100, kway_naive);
    test(4, 100, kway_heap);
}
