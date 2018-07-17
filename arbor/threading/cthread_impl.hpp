#pragma once

#include <iostream>
#include <type_traits>

#include <thread>
#include <mutex>
#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>
#include <type_traits>
#include <functional>
#include <condition_variable>
#include <utility>
#include <unordered_map>
#include <deque>
#include <atomic>
#include <type_traits>

#include <cstdlib>
#include "arbor/execution_context.hpp"

namespace arb {

inline threading::impl::task_system* get_task_system(task_system_handle* h) {
    return (*h).get();
}

namespace threading {

// Forward declare task_group at bottom of this header
class task_group;

namespace impl {

using arb::threading::task_group;
using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;

using task = std::function<void()>;
using task_queue = std::deque<task>;

using thread_list = std::vector<std::thread>;
using thread_map = std::unordered_map<std::thread::id, std::size_t>;

class notification_queue {
private:
    // FIFO of pending tasks.
    task_queue q_tasks_;

    // Lock and signal on task availability change this is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;

public:
    // Pops a task from the task queue returns false when queue is empty or quit is set.
    bool try_pop(task& tsk);
    bool pop(task& tsk);

    // Pushes a task into the task queue and increases task group counter.
    void push(task&& tsk); // TODO: need to use value?
    bool try_push(task& tsk);

    // Stop queue from popping new tasks.
    void quit();
};

//manipulates in_flight
class task_system {
private:
    std::size_t count_;

    thread_list threads_;

    // queue of tasks
    std::vector<notification_queue> q_;

    // threads -> index
    thread_map thread_ids_;

    // total number of tasks pushed in all queues
    std::atomic<unsigned> index_{0};

public:
    // Create nthreads-1 new c std threads
    task_system(int nthreads);

    // task_system is a singleton. TODO
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    ~task_system();

    // Pushes tasks into notification queue.
    void async_(task tsk);

    // Runs tasks until quit is true.
    void run_tasks_loop(int i);

    // Request that the task_system attempts to find and run a _single_ task.
    // Will return without executing a task if no tasks available.
    void try_run_task(int i);

    // Includes master thread.
    int get_num_threads();

    // Get a stable integer for the current thread that is [0, nthreads).
    std::size_t get_current_thread();
};

} //impl

///////////////////////////////////////////////////////////////////////
// types
///////////////////////////////////////////////////////////////////////

template <typename T>
class enumerable_thread_specific {
    impl::task_system* global_task_system = nullptr;

    using storage_class = std::vector<T>;
    storage_class data;

public :
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific(task_system_handle* ts):
            global_task_system{get_task_system(ts)},
            data{std::vector<T>(global_task_system->get_num_threads())}
    {}

    T& local() {
        return data[global_task_system->get_current_thread()];
    }
    const T& local() const {
        return data[global_task_system->get_current_thread()];
    }

    auto size() const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};

inline std::string description() {
    return "CThread Pool";
}

constexpr bool multithreaded() { return true; }

using std::mutex;
using lock = std::unique_lock<mutex>;
using task = std::function<void()>;


class task_group {
private:
    std::atomic<std::size_t> in_flight_{0};
    impl::task_system* task_system_;

public:
    task_group(impl::task_system* ts):
        task_system_{ts}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    template <typename F>
    class wrap {
        F f;
        std::atomic<std::size_t>& counter;

    public:

        // Construct from a compatible function and atomic counter
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c):
                f(std::forward<F2>(other)),
                counter(c)
        {}

        wrap(wrap&& other):
                f(std::move(other.f)),
                counter(other.counter)
        {}

        // Shouldn't be used, but is required if we want to wrap with std::function
        wrap(const wrap& other):
                f(other.f),
                counter(other.counter)
        {}

        void operator()() {
            f();
            --counter;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c) {
        return wrap<callable<F>>(std::forward<F>(f), c);
    }

    template<typename F>
    void run(F&& f) {
        ++in_flight_;

        task_system_->async_(make_wrapped_function(std::forward<F>(f), in_flight_));
    }

    // wait till all tasks in this group are done
    void wait() {
        while (in_flight_) {
            task_system_->try_run_task(0);
        }
    }

    // Make sure that all tasks are done before clean up
    ~task_group() {
        wait();
    }
};

///////////////////////////////////////////////////////////////////////
// algorithms
///////////////////////////////////////////////////////////////////////
struct parallel_for {
    template <typename F>
    static void apply(int left, int right, impl::task_system* ts, F f) {
        task_group g(ts);
        for(int i = left; i < right; ++i) {
          g.run([=] {f(i);});
        }
        g.wait();
    }
};

} // namespace threading
} // namespace arb
