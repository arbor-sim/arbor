#pragma once

#include <vector>

#include "threading.hpp"

namespace arb {
namespace threading {

template <typename T>
class enumerable_thread_specific {
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    using storage_class = std::vector<T>;
    storage_class data;

public:
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific(const task_system_handle& ts):
        thread_ids_{ts->get_thread_ids()},
        data{std::vector<T>(ts->get_num_threads())}
    {}

    enumerable_thread_specific(const T& init, const task_system_handle& ts):
        thread_ids_{ts->get_thread_ids()},
        data{std::vector<T>(ts->get_num_threads(), init)}
    {}

    T& local() {
        return data[thread_ids_.at(std::this_thread::get_id())];
    }
    const T& local() const {
        return data[thread_ids_.at(std::this_thread::get_id())];
    }

    auto size() const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};

} // namespace threading
} // namespace arb

