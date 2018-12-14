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
    struct exception_state {
        std::atomic<bool> error_{false};
        std::exception_ptr exception_;
        std::mutex mutex_;

        operator bool() const {
            return error_.load(std::memory_order_relaxed);
        }

        void set(std::exception_ptr ex) {
            error_.store(true, std::memory_order_relaxed);
            lock ex_lock{mutex_};
            exception_ = std::move(ex);
        }

        // Clear exception state but return old state.
        // For consistency, this must only be called when there
        // are no tasks in flight that reference this exception state.
        std::exception_ptr reset() {
            auto ex = std::move(exception_);
            error_.store(false, std::memory_order_relaxed);
            exception_ = nullptr;
            return ex;
        }
    };

    std::atomic<std::size_t> in_flight_{0};

    // Set by run(), cleared by wait(). Used to check task completion status
    // in destructor.
    bool running_ = false;

    // We use a raw pointer here instead of a shared_ptr to avoid a race condition
    // on the destruction of a task_system that would lead to a thread trying to join itself.
    task_system* task_system_;
    exception_state exception_status_;

public:
    task_group(task_system* ts):
        task_system_{ts}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    template <typename F>
    class wrap {
        F f_;
        std::atomic<std::size_t>& counter_;
        exception_state& exception_status_;

    public:
        // Construct from a compatible function and atomic counter
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c, exception_state& ex):
                f_(std::forward<F2>(other)),
                counter_(c),
                exception_status_(ex)
        {}

        wrap(wrap&& other):
                f_(std::move(other.f_)),
                counter_(other.counter_),
                exception_status_(other.exception_status_)
        {}

        // std::function is not guaranteed to not copy the contents on move construction,
        // but the class is safe because we don't call operator() more than once on the same wrapped task.
        wrap(const wrap& other):
                f_(other.f_),
                counter_(other.counter_),
                exception_status_(other.exception_status_)
        {}

        void operator()() {
            if (!exception_status_) {
                try {
                    f_();
                }
                catch (...) {
                    exception_status_.set(std::current_exception());
                }
            }
            --counter_;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c, exception_state& ex) {
        return wrap<callable<F>>(std::forward<F>(f), c, ex);
    }

    template<typename F>
    void run(F&& f) {
        running_ = true;
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_, exception_status_));
    }

    // Wait till all tasks in this group are done.
    void wait() {
        while (in_flight_) {
            task_system_->try_run_task();
        }
        running_ = false;

        if (auto ex = exception_status_.reset()) {
            std::rethrow_exception(ex);
        }
    }

    ~task_group() {
        if (running_) std::terminate();
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
        g.wait();
    }
};
} // namespace threading

using task_system_handle = std::shared_ptr<threading::task_system>;

} // namespace arb
