#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <utility>

namespace arb {
namespace threading {

// Forward declare task_group at bottom of this header
class task_group;

using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;
using task = std::function<void()>;

namespace impl {
class notification_queue {
private:
    // FIFO of pending tasks.
    std::deque<task> q_tasks_;

    // Lock and signal on task availability change this is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;

public:
    // Pops a task from the task queue returns false when queue is empty.
    task try_pop();
    task pop();

    // Pushes a task into the task queue and increases task group counter.
    void push(task&& tsk); // TODO: need to use value?
    bool try_push(task& tsk);

    // Finish popping all waiting tasks on queue then stop trying to pop new tasks
    void quit();
};
}// namespace impl

class task_system {
private:
    unsigned count_;

    std::vector<std::thread> threads_;

    // queue of tasks
    std::vector<impl::notification_queue> q_;

    // threads -> index
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // total number of tasks pushed in all queues
    std::atomic<unsigned> index_{0};

public:
    task_system();
    // Create nthreads-1 new c std threads
    task_system(int nthreads);

    // task_system is a singleton.
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    ~task_system();

    // Pushes tasks into notification queue.
    void async(task tsk);

    // Runs tasks until quit is true.
    void run_tasks_loop(int i);

    // Request that the task_system attempts to find and run a _single_ task.
    // Will return without executing a task if no tasks available.
    void try_run_task();

    // Includes master thread.
    int get_num_threads() const;

    // Returns the thread_id map
    std::unordered_map<std::thread::id, std::size_t> get_thread_ids() const;
};

class task_group {
private:
    struct exception_struct {
        std::atomic<bool> bail{false};
        std::exception_ptr exception_ptr_;
        std::mutex exception_mutex_;

        bool have_exception() {
            return bail.load(std::memory_order_relaxed);
        }

        void set_exception(std::exception_ptr ex) {
            bail.store(true, std::memory_order_relaxed);
            lock ex_lock{exception_mutex_};
            exception_ptr_ = std::move(ex);
        }
    };

    std::atomic<std::size_t> in_flight_{0};
    /// We use a raw pointer here instead of a shared_ptr to avoid a race condition
    /// on the destruction of a task_system that would lead to a thread trying to join itself
    task_system* task_system_;
    exception_struct exception_status_;

public:
    task_group(task_system* ts):
        task_system_{ts}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    template <typename F>
    class wrap {
        F f;
        std::atomic<std::size_t>& counter;
        exception_struct& exception_status;

    public:

        // Construct from a compatible function and atomic counter
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c, exception_struct& ex):
                f(std::forward<F2>(other)),
                counter(c),
                exception_status(ex)
        {}

        wrap(wrap&& other):
                f(std::move(other.f)),
                counter(other.counter),
                exception_status(other.exception_status)
        {}

        // std::function is not guaranteed to not copy the contents on move construction
        // But the class is safe because we don't call operator() more than once on the same wrapped task
        wrap(const wrap& other):
                f(other.f),
                counter(other.counter),
                exception_status(other.exception_status)
        {}

        void operator()() {
            if(!exception_status.have_exception()) {
                try {
                    f();
                }
                catch (const std::exception &ex) {
                    exception_status.set_exception(std::current_exception());
                }
            }
            --counter;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c, exception_struct& ex) {
        return wrap<callable<F>>(std::forward<F>(f), c, ex);
    }

    template<typename F>
    void run(F&& f) {
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_, exception_status_));
    }

    // wait till all tasks in this group are done
    std::exception_ptr wait() {
        while (in_flight_) {
            task_system_->try_run_task();
        }
        return exception_status_.exception_ptr_;
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
    static void apply(int left, int right, task_system* ts, F f) {
        task_group g(ts);
        for (int i = left; i < right; ++i) {
          g.run([=] {f(i);});
        }
        auto ex = g.wait();

        if(ex) {
            try
            {
                std::rethrow_exception(ex);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(e.what());
            }
        }
    }
};
} // namespace threading

using task_system_handle = std::shared_ptr<threading::task_system>;

} // namespace arb
