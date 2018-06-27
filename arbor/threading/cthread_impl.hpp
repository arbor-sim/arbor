#pragma once


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

#include <cstdlib>

namespace arb {
namespace threading {
inline namespace cthread {

// Forward declare task_group at bottom of this header
class task_group;

namespace impl {

using arb::threading::task_group;
using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;

using task = std::pair<std::function<void()>, task_group*>;
using task_queue = std::deque<task>;

using thread_list = std::vector<std::thread>;
using thread_map = std::unordered_map<std::thread::id, std::size_t>;

class notification_queue {
private:
    // fifo of pending tasks
    task_queue q_tasks_;

    // lock and signal on task availability change
    // this is the crucial bit
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // flag to handle exit from all threads
    bool quit_ = false;

public:
    // pops a task from the task queue
    // returns false when queue is empty or quit is set
    bool try_pop(task& tsk);
    bool pop(task& tsk);
    // pops a task from the task queue
    //
    template<typename B>
    bool pop_if_not(task& tsk, B condition);

    // after the function of a task has been executed
    // decrease the task counter of corresponding task_group
    void remove_from_task_group(task &tsk);

    // pushes a task into the task queue
    // and increases task group counter

    void push(task&& tsk);
    void push(const task& tsk);
    bool try_push(const task& tsk);

    //stop queue from popping new tasks
    void quit();
};

//manipulates in_flight
class task_system {
private:
    std::size_t count_;
    //thread_resource
    thread_list threads_;
    //lock for thread_map
    mutex thread_ids_mutex_;
    // queue of tasks
    std::vector<notification_queue> q_;
    // threads -> index
    thread_map thread_ids_;
    // total number of tasks pushed in all queues
    std::atomic<unsigned> index_{0};

public:
    // Create nthreads-1 new c std threads
    task_system(int nthreads);

    // task_system is a singleton FOR NOW
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    ~task_system();

    // pushes tasks into notification queue
    void async_(task&& tsk);

    // waits for all tasks in the group to be done
    void wait(task_group*);

    // runs tasks until quit is true
    void run_tasks_loop();

    // includes master thread
    int get_num_threads();

    // get a stable integer for the current thread that
    // is 0..nthreads
    std::size_t get_current_thread();

    // singleton constructor - needed to order construction
    // with other singletons (profiler)
    static task_system& get_global_task_system();
};
} //impl

///////////////////////////////////////////////////////////////////////
// types
///////////////////////////////////////////////////////////////////////
template <typename T>
class enumerable_thread_specific {
    impl::task_system& global_task_system;

    using storage_class = std::vector<T>;
    storage_class data;

public :
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific():
        global_task_system{impl::task_system::get_global_task_system()},
        data{std::vector<T>(global_task_system.get_num_threads())}
    {}

    enumerable_thread_specific(const T& init):
        global_task_system{impl::task_system::get_global_task_system()},
        data{std::vector<T>(global_task_system.get_num_threads(), init)}
    {}

    T& local() {
        return data[global_task_system.get_current_thread()];
    }
    const T& local() const {
        return data[global_task_system.get_current_thread()];
    }

    auto size() -> decltype(data.size()) const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};

template <typename T>
class parallel_vector {
    using value_type = T;
    std::vector<value_type> data_;

private:
    // lock the parallel_vector to update
    impl::mutex mutex;

    // call a function of type X f() in a lock
    template<typename F>
    auto critical(F f) -> decltype(f()) {
        impl::lock lock{mutex};
        return f();
    }

public:
    parallel_vector() = default;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    iterator begin() { return data_.begin(); }
    iterator end()   { return data_.end(); }

    const_iterator begin() const { return data_.begin(); }
    const_iterator end()   const { return data_.end(); }

    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend()   const { return data_.cend(); }

    // only guarantees the state of the vector, but not the iterators
    // unlike tbb push_back
    void push_back (value_type&& val) {
        critical([&] {
            data_.push_back(std::move(val));
        });
    }
};

inline std::string description() {
    return "CThread Pool";
}

constexpr bool multithreaded() { return true; }

using std::mutex;
using lock = std::unique_lock<mutex>;

class task_group {
private:
    std::atomic<std::size_t> in_flight{0};
    impl::task_system& global_task_system;
    mutex g_mutex_;

public:
    task_group():
        global_task_system{impl::task_system::get_global_task_system()}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    void dec_in_flight() {
        {
            lock g_lock{g_mutex_};
            in_flight--;
        }
    }

    void inc_in_flight() {
        {
            lock g_lock{g_mutex_};
            in_flight++;
        }
    }

    std::size_t get_in_flight() {
        return in_flight;
    }

    // send function void f() to threads
    template<typename F>
    void run(const F& f) {
        global_task_system.async_(impl::task{f, this});
    }

    template<typename F>
    void run(F&& f) {
        global_task_system.async_(impl::task{std::move(f), this});
    }

    // run function void f() and then wait on all threads in group
    template<typename F>
    void run_and_wait(const F& f) {
        f();
        global_task_system.wait(this);
    }

    template<typename F>
    void run_and_wait(F&& f) {
        f();
        global_task_system.wait(this);
    }

    // wait till all tasks in this group are done
    void wait() {
        global_task_system.wait(this);
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
    static void apply(int left, int right, F f) {
        task_group g;
        for(int i = left; i < right; ++i) {
          g.run([=] {f(i);});
        }
        g.wait();
    }
};

inline std::size_t thread_id() {
    return impl::task_system::get_global_task_system().get_current_thread();
}

} // namespace cthread
} // namespace threading
} // namespace arb
