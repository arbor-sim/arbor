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

#include <arbor/export.hpp>

namespace arb {
namespace threading {

using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;
using task = std::function<void()>;

// Tasks with priority higher than max_async_task_priority will be run synchronously.
constexpr int max_async_task_priority = 1;

// Wrap task and priority; provide move/release/reset operations and reset on run()
// to help ensure no wrapped task is run twice.
struct priority_task {
    task t;
    int priority = -1;

    priority_task() = default;
    priority_task(task&& t, int priority): t(std::move(t)), priority(priority) {}

    priority_task(priority_task&& other) noexcept {
        std::swap(t, other.t);
        priority = other.priority;
    }

    priority_task& operator=(priority_task&& other) noexcept {
        reset();
        std::swap(t, other.t);
        priority = other.priority;
        return *this;
    }

    priority_task(const priority_task&) = delete;
    priority_task& operator=(const priority_task&) = delete;

    explicit operator bool() const noexcept { return static_cast<bool>(t); }

    void run() {
        release()();
    }

    task release() {
        task u = std::move(t);
        reset();
        return u;
    }

    void reset() noexcept {
        t = nullptr;
    }
};

namespace impl {

class ARB_ARBOR_API notification_queue {
    // Number of priority levels in notification queues.
    static constexpr int n_priority = max_async_task_priority+1;

public:
    // Tries to acquire the lock to get a task of a requested priority.
    // If unsuccessful returns an empty task. If the lock is acquired
    // successfully, and the deque containing the tasks of the requested
    // priority is not empty; pops a task from that deque and returns it.
    // Otherwise returns an empty task.
    priority_task try_pop(int priority);

    // Acquires the lock and pops a task from the highest priority deque
    // that is not empty. If all deques are empty, it waits for a task to
    // be enqueued. If after a task is enqueued, it still can't acquire it
    // (because it was popped by another thread), returns an empty task.
    // If quit_ is set and the deques are all empty, returns an empty task.
    priority_task pop();

    // Acquires the lock and pushes the task into the deque containing
    // tasks of the same priority, then notifies the condition variable to
    // awaken waiting threads.
    void push(priority_task&&);

    // Tries to acquire the lock: if successful, pushes the task onto the
    // deque containing tasks of the same priority, notifies the condition
    // variable to awaken waiting threads and returns true. If unsuccessful
    // returns false.
    bool try_push(priority_task&);

    // Finish popping all waiting tasks on queue then stop trying to pop
    // new tasks
    void quit();

    // Check whether the deques are all empty.
    bool empty();

private:
    // deques of pending tasks. Each deque contains tasks of a single priority.
    // q_tasks_[i+1] has higher priority than q_tasks_[i]
    std::array<std::deque<task>, n_priority> q_tasks_;

    // Lock and signal on task availability change. This is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;
};

}// namespace impl

class ARB_ARBOR_API task_system {
private:
    // Number of notification queues.
    unsigned count_;

    // Worker threads.
    std::vector<std::thread> threads_;

    // Local thread storage: used to encode the priority of the task
    // currently executed by the thread.
    // It is initialized to -1 and reset to -1 in the destructor,
    // where a value of -1 => not running a task system task.
    static thread_local int current_task_priority_;

    // Queue index for the running thread, if any,
    // A value of -1 indicates that the executing thread is not one in
    // threads_.
    static thread_local unsigned current_task_queue_;

    // Number of priority levels in notification queues.
    static constexpr int n_priority = max_async_task_priority+1;

    // Notification queues containing n_priority deques representing
    // different priority levels.
    std::vector<impl::notification_queue> q_;

    // Map from thread id to index in the vector of threads.
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // Total number of tasks pushed in each priority level.
    // Used to index which notification queue to enqueue tasks on to,
    // to balance the workload among the queues.
    std::array<std::atomic<unsigned>, n_priority> index_;

public:
    // Create zero new threads. Only worker thread is the main thread.
    task_system();

    // Create nthreads-1 new std::threads running run_tasks_loop(tid)
    task_system(int nthreads);

    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    // Quits the notification queues. Joins the threads. Resets thread_depth_.
    // Won't wait for the existing tasks in the notification queues to be executed.
    ~task_system();

    // Pushes tasks into a notification queue, on a deque of the requested priority.
    // Will first attempt to push on all the notification queues, round-robin, starting
    // with the notification queue at index_[priority]. If unsuccessful, forces a push
    // onto the notification queue at index_[priority].

    // Public interface: run task asynchronously if priority <= max_async_task_priority,
    // else equivalent to task_system::run(priority_task) below.
    void async(priority_task ptsk);

    // Public interface: run task synchronously with current task priority set.
    void run(priority_task ptsk);

    // Convenience interfaces with priority parameter:
    void async(task t, int priority) { async({std::move(t), priority}); }
    void run(task t, int priority) { run({std::move(t), priority}); }

    // The main function that all worker std::threads execute.
    // It will try to acquire a task of the highest possible of priority from all
    // of the notification queues. If unsuccessful it will force pop any task from
    // the thread's personal queue, trying again from highest to lowest priority.
    // Note on stack overflow possibility: the force pop can seem like it could cause
    // an issue in cases when the personal queue only has low priority tasks that
    // spawn higher priority tasks that end up on other queues. In that case the thread
    // can end up in a situation where it keeps executing low priority tasks as it
    // waits for higher priority tasks to finish, causing a stack overflow. The key
    // point here is that while the thread is waiting for other tasks to finish, it
    // is not executing the run_tasks_loop, but the task_group::wait() loop which
    // doesn't use pop but always try_pop.
    // `i` is the thread idx, used to select the thread's personal notification queue.
    void run_tasks_loop(int i);

    // Public interface: try to dequeue and run a single task with at least the
    // requested priority level. Will return without executing a task if no tasks
    // are available or if the lock can't be acquired.
    //
    // Will start with queue corresponding to calling thread, if one exists.
    void try_run_task(int lowest_priority);

    // Number of threads in pool, including master thread.
    // Equivalently, number of notification queues.
    int get_num_threads() const { return (int)count_; }

    static int get_task_priority() { return current_task_priority_; }

    // Returns the thread_id map
    std::unordered_map<std::thread::id, std::size_t> get_thread_ids() const;
};

class task_group {
private:
    // For tracking exceptions raised inside the task_system.
    // If multiple tasks raise exceptions, any exception can be
    // saved and returned. Once an exception has been raised, the
    // rest of the tasks don't need to be executed, but if a few are
    // executed anyway, that's okay.
    // For the reset to work correctly using the relaxed memory order,
    // it is necessary that both task_group::run and task_group::wait
    // are synchronization points, which they are because they require
    // mutex acquisition. The reason behind this requirement is that
    // exception_state::reset is called at the end of task_group::wait,
    // and we need it to actually reset exception_state::error_ before
    // we start running any new tasks in the group.
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

    // Number of tasks that are queued but not yet executed.
    std::atomic<std::size_t> in_flight_{0};

    // Set by run(), cleared by wait(). Used to check task completion status in destructor.
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
        // Construct from a compatible function, atomic counter, and exception_state.
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

        // This is where tasks of the task_group are actually executed.
        void operator()() {
            if (!exception_status_) {
                // Execute the task. Save exceptions if they occur.
                try {
                    f_();
                }
                catch (...) {
                    exception_status_.set(std::current_exception());
                }
            }
            // Decrement the atomic counter of the tasks in the task_group;
            --counter_;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c, exception_state& ex) {
        return wrap<callable<F>>(std::forward<F>(f), c, ex);
    }

    // Adds new tasks to be executed in the task_group.
    // Use priority one higher than that of the task in the currently
    // executing thread, if any, so that tasks in nested task groups
    // are completed before any peers of the parent task. Returns this
    // priority.
    template<typename F>
    int run(F&& f) {
        int priority = task_system::get_task_priority()+1;
        run(std::forward<F>(f), priority);
        return priority;
    }

    // Adds a new task with a given priority to be executed.
    template<typename F>
    void run(F&& f, int priority) {
        running_ = true;
        ++in_flight_;
        task_system_->async(priority_task{make_wrapped_function(std::forward<F>(f), in_flight_, exception_status_), priority});
    }

    // Wait till all tasks in this group are done.
    // While waiting the thread will participate in executing the tasks.
    // It's necessary that the waiting thread participate in execution:
    // otherwise, due to nested parallelism, all the threads could become
    // stuck waiting forever, while no new tasks get executed.
    // To shorten waiting time, and reduce the chances of stack overflow,
    // the waiting thread can only execute tasks with a higher priority
    // than the task it is currently running.
    void wait() {
        int lowest_priority = task_system::get_task_priority()+1;
        while (in_flight_) {
            task_system_->try_run_task(lowest_priority);
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
    // Creates a task group, enqueues tasks and waits for their completion.
    // If a batching size if not specified, a default batch size of 1 is used.
    template <typename F>
    static void apply(int left, int right, int batch_size, task_system* ts, F f) {
        task_group g(ts);
        for (int i = left; i < right; i += batch_size) {
            g.run([=] {
                int r = i + batch_size < right ? i + batch_size : right;
                for (int j = i; j < r; ++j) {
                    f(j);
                }
            });
        }
        g.wait();
    }

    template <typename F>
    static void apply(int left, int right, task_system* ts, F f) {
        apply(left, right, 1, ts, std::move(f));
    }
};
} // namespace threading

using task_system_handle = std::shared_ptr<threading::task_system>;

} // namespace arb
