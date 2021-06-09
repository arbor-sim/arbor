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

// The maximum levels of nested parallelism allowed.
// If the user attempts to nest deeper, tasks from that level will be
// executed immediately, outside of the task system and notification queues.
constexpr int max_task_depth = 2;

class notification_queue {
public:

    // Tries to acquire the lock to get a task from the deque with the
    // requested priority. if unsuccessful returns an empty task.
    // If the lock is acquired successfully, and the requested deque
    // is not empty, pops a task from the deque; otherwise returns an
    // empty task.
    task try_pop(int priority);

    // Acquires the lock and pops a task from the highest priority deque
    // that is not empty. If all deques are empty it waits for a task to
    // be enqueued. If after a task is enqueued, it still can't acquire it
    // (because it was popped by another thread), returns an empty task.
    // If quit_ is set and the deques are all empty, returns an empty task.
    task pop();

    // Acquires the lock and pushes a task into the deque with the requested
    // priority and notifies the condition variable to awaken waiting threads.
    void push(task&& tsk, int priority); // TODO: need to use value?

    // Tries to acquire the lock; if successful, pushes a task onto the deque
    // with the requested priority, notifies the condition variable to awaken
    // waiting threads and returns true. If unsuccessful returns false.
    bool try_push(task& tsk, int priority);

    // Finish popping all waiting tasks on queue then stop trying to pop new tasks
    void quit();

    // Checks whether the deques of the queue are all empty.
    bool empty();

private:
    // FIFOs of pending tasks: q_tasks_[i+1] has higher priority than q_tasks_[i]
    std::array<std::deque<task>, max_task_depth> q_tasks_;

    // Lock and signal on task availability change this is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;

};
}// namespace impl

class task_system {
private:
    // Number of notification queues.
    unsigned count_;

    // Spawned worker std::threads.
    std::vector<std::thread> threads_;

    // Local thread storage, used to encode the priority of the task
    // currently executed by the thread: higher thread_depth_ means
    // higher priority.
    // It is initialized to -1 and reset to -1 in the destructor.
    // It is manipulated during task execution:
    // - Before a *new* task is executed, thread_depth_ (depth of
    //   *currently* executing task) is saved.
    // - thread_depth_ is then set to the depth of the *new* task.
    // - Once the *new* task has been executed, the thread_depth_
    //   is reset to its previous value.
    // (The depth of *new* tasks is set to the depth of the *current*
    // task + 1).
    static thread_local int thread_depth_;

    // Notification queues containing max_task_depth deques representing
    // different priority levels.
    std::vector<impl::notification_queue> q_;

    // Map from thread id to index.
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // Total number of tasks pushed in each priority level.
    // Used to index which notification queue to enqueue tasks on to,
    // to balance the workload among the queues.
    std::array<std::atomic<unsigned>, impl::max_task_depth> index_{0,0};

public:
    // Create zero new threads. Only worker thread is the main thread.
    task_system();

    // Create nthreads-1 new std::threads running run_tasks_loop(tid)
    task_system(int nthreads);

    // task_system is a singleton.
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    // Quits the notification queues. Joins the threads. Resets thread_depth_.
    // Won't wait for the existing tasks in the notification queues to be executed.
    ~task_system();

    // Pushes tasks into notification queue, with the requested priority level.
    // Will first attempt to push on all the queues, round-robin, starting with
    // the queue at index_[priority]. If unsuccessful, forces a push onto the
    // queue at index_[priority].
    void async(task tsk, int priority);

    // The main function that all spawned std::threads execute.
    // It will try to acquire a task of the highest possible of priority from
    // all of the notification queues. If unsuccessful it will force pop any task
    // from the thread's personal queue, trying again from highest to lowest priority.
    // Note on stack overflow possibility: the force pop can seem like it'll cause
    // an issue in cases when the personal queue only has low priority tasks, and
    // the other queues have many higher priority tasks. It seems like the thread
    // can end up in a situation where it keeps spawning low priority tasks as it
    // waits for higher priority tasks to finish. The key point here is that
    // **while the thread is waiting for other tasks to finish, it is not executing
    // the run_tasks_loop, but the task_group::wait() loop which doesn't use pop**
    // `i` is the thread id, it's the index of the first queue to check for tasks.
    void run_tasks_loop(int i);

    // Try to run a single task with at least the requested priority level.
    // Will return without executing a task if no tasks available or the
    // lock can't be acquired.
    // `i` is the thread id, it's the index of the first queue to check for tasks.
    void try_run_task(int i, int lowest_priority);

    // Includes master thread.
    int get_num_threads() const;

    // Returns the current depth of nested parallelism on the current thread.
    // 0 is the first level of parallelism, 1 is the next, so on.
    static int get_thread_depth();

    // Set the depth of the nested parallelism on the current thread.
    static void set_thread_depth(int depth);

    // Returns the thread_id map
    std::unordered_map<std::thread::id, std::size_t> get_thread_ids() const;
};

class task_group {
private:
    // For tracking exceptions raised inside the task_system.
    // If multiple tasks raise exceptions, any exception can be
    // saved and returned. Once an exception has been raised, the
    // rest of the tasks don't need to be executed, but if a few are
    // that's okay.
    // For the reset to work correctly using the relaxed memory order,
    // it is necessary that both task_group::run and task_group::wait
    // are synchronization points, which they are because they require
    // mutex acquisition. This is because exception_state::reset is
    // called at the end of task_group::wait, and we need it to actually
    // reset exception_state::error_ before we start running any new tasks
    // in the group.
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
        int depth_;
        exception_state& exception_status_;

    public:
        // Construct from a compatible function, atomic counter, priority level and exception_state.
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c, int d, exception_state& ex):
                f_(std::forward<F2>(other)),
                counter_(c),
                depth_(d),
                exception_status_(ex)
        {}

        wrap(wrap&& other):
                f_(std::move(other.f_)),
                counter_(other.counter_),
                depth_(other.depth_),
                exception_status_(other.exception_status_)
        {}

        // std::function is not guaranteed to not copy the contents on move construction,
        // but the class is safe because we don't call operator() more than once on the same wrapped task.
        wrap(const wrap& other):
                f_(other.f_),
                counter_(other.counter_),
                depth_(other.depth_),
                exception_status_(other.exception_status_)
        {}

        // This is where tasks of the task_group are actually executed.
        // Uses the local thread storage in the task system to track the priority
        // level of the currently executing task. Before a new task is executed,
        // priority of the thread is updated to the priority of the new task (depth_)
        // and then reset to the previous priority when the execution is done.
        void operator()() {
            if (!exception_status_) {
                // Save the current depth of the thread to be reset after task execution.
                auto tdepth = task_system::get_thread_depth();

                // Set the depth of the thread to the depth of the task.
                task_system::set_thread_depth(depth_);

                // Execute the task.
                try {
                    f_();
                }
                catch (...) {
                    exception_status_.set(std::current_exception());
                }

                // Reset the depth of the thread.
                task_system::set_thread_depth(tdepth);
            }
            // Decrement the atomic counter of the tasks in the task_group;
            --counter_;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c, int d, exception_state& ex) {
        return wrap<callable<F>>(std::forward<F>(f), c, d, ex);
    }

    // Enqueues new tasks belonging to the task_group.
    // The depth of nested parallelism is automatically
    // calculated used to set the priority of the task.
    // Returns the depth of the enqueued task.
    template<typename F>
    int run(F&& f) {
        running_ = true;
        int thread_depth = task_system::get_thread_depth();
        // Don't nest parallelism after a certain depth.
        if (thread_depth+1 >= impl::max_task_depth) {
            try {
                f();
            }
            catch (...) {
                exception_status_.set(std::current_exception());
            }
            return thread_depth;
        }
        auto task_depth = thread_depth+1;
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_, task_depth, exception_status_), task_depth);
        return task_depth;
    }

    // Enqueues new tasks belonging to the task_group with a given priority.
    template<typename F>
    void run(F&& f, int depth) {
        running_ = true;
        if (depth >= impl::max_task_depth) {
            try {
                f();
            }
            catch (...) {
                exception_status_.set(std::current_exception());
            }
            return;
        }
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_, depth, exception_status_), depth);
    }

    // Wait till all tasks in this group are done.
    // While waiting the thread will participate in executing the tasks.
    // It's necessary that the waiting thread participate in execution:
    // otherwise, due to nested parallelism, all the threads could become
    // stuck waiting forever, while no new tasks get executed.
    // To shorten waiting time, and reduce the chances of stack overflow,
    // the waiting thread can only execute tasks with a higher or equal
    // priority to the task it is currently running.
    void wait(int lowest_priority=0) {
        auto tid = task_system_->get_thread_ids()[std::this_thread::get_id()];
        while (in_flight_) {
            task_system_->try_run_task(tid, lowest_priority);
        }
        running_ = false;

        if (auto ex = exception_status_.reset()) {
            std::rethrow_exception(ex);
        }
    }

    ~task_group() {
        if (in_flight_&&running_) std::terminate();
    }
};

///////////////////////////////////////////////////////////////////////
// algorithms
///////////////////////////////////////////////////////////////////////
struct parallel_for {
    // Creates a task group, tasks and waits for their completion.
    // If a batching size if not specified, a default batch size of
    // num_tasks/(num_thread*32)+1 is chosen.
    // Automatically checks the nested parallelism level and executes
    // the tasks outside of the task system if it exceeds a predefined
    // threshold.
    template <typename F>
    static void apply(int left, int right, int batch_size, task_system* ts, F f) {
        int current_depth = task_system::get_thread_depth();
        task_group g(ts);
        for (int i = left; i < right; i += batch_size) {
            g.run([=] {
                int r = i + batch_size < right ? i + batch_size : right;
                for (int j = i; j < r; ++j) {
                    f(j);
                }
            }, current_depth+1);
        }
        g.wait(current_depth+1);
    }

    template <typename F>
    static void apply(int left, int right, task_system* ts, F f) {
        int batch_size = ((right - left) / (ts->get_num_threads()) * 32) + 1;
        apply(left, right, batch_size, ts, std::move(f));
    }
};
} // namespace threading

using task_system_handle = std::shared_ptr<threading::task_system>;

} // namespace arb
