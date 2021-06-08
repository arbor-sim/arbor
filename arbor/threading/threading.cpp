#include <atomic>

#include <arbor/assert.hpp>

#include "threading.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;

task notification_queue::try_pop(int priority) {
    arb_assert(priority < q_tasks_.size());
    task tsk;
    lock q_lock{q_mutex_, std::try_to_lock};
    if (!q_lock) return tsk;
    auto& q = q_tasks_.at(priority);
    if (!q.empty()) {
        tsk = std::move(q.front());
        q.pop_front();
    }
    return tsk;
}

task notification_queue::pop() {
    task tsk;
    lock q_lock{q_mutex_};
    while (empty() && !quit_) {
        q_tasks_available_.wait(q_lock);
    }
    for (auto it = q_tasks_.rbegin(); it != q_tasks_.rend(); ++it) {
        if (!it->empty()) {
            tsk = std::move(it->front());
            it->pop_front();
            break;
        }
    }
    return tsk;
}

bool notification_queue::try_push(task& tsk, int priority) {
    arb_assert(priority < q_tasks_.size());
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;
        q_tasks_.at(priority).push_front(std::move(tsk));
        tsk = 0;
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(task&& tsk, int priority) {
    arb_assert(priority < q_tasks_.size());
    {
        lock q_lock{q_mutex_};
        q_tasks_.at(priority).push_front(std::move(tsk));
    }
    q_tasks_available_.notify_all();
}

void notification_queue::quit() {
    {
        lock q_lock{q_mutex_};
        quit_ = true;
    }
    q_tasks_available_.notify_all();
}

bool notification_queue::empty() {
    for(const auto& q: q_tasks_) {
        if (!q.empty()) return false;
    }
    return true;
}

void task_system::run_tasks_loop(int i){
    while (true) {
        task tsk;
        // Loop over the levels of priority starting from highest to lowest
        for (int depth = impl::max_task_depth-1; depth >= 0; depth--) {
            // Loop over the threads trying to pop a task of the requested priority.
            for (unsigned n = 0; n != count_; n++) {
                tsk = q_[(i + n) % count_].try_pop(depth);
                if (tsk) break;
            }
            if (tsk) break;
        }
        // If a task can not be acquired, force a pop from the queue. This is a blocking action.
        if (!tsk) tsk = q_[i].pop();
        if (!tsk) break;
        tsk();
    }
}

void task_system::try_run_task(int i, int lowest_priority) {
    auto nthreads = get_num_threads();
    task tsk;
    // Loop over the levels of priority starting from highest to lowest_priority
    for (int depth = impl::max_task_depth-1; depth >= lowest_priority; depth--) {
        // Loop over the threads trying to pop a task of the requested priority.
        for (int n = 0; n != nthreads; n++) {
            tsk = q_[(i + n) % nthreads].try_pop(depth);
            if (tsk) {
                tsk();
                return;
            }
        }
    }
}

thread_local int task_system::thread_depth_ = -1;

// Default construct with one thread.
task_system::task_system(): task_system(1) {}

task_system::task_system(int nthreads): count_(nthreads), q_(nthreads) {
    if (nthreads <= 0)
        throw std::runtime_error("Non-positive number of threads in thread pool");

    // Main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;

    for (unsigned i = 1; i < count_; i++) {
        threads_.emplace_back([this, i]{run_tasks_loop(i);});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_system::~task_system() {
    for (auto& e: q_) e.quit();
    for (auto& e: threads_) e.join();
    set_thread_depth(-1);
}

void task_system::async(task tsk, int priority) {
    auto i = index_[priority]++;

    for (unsigned n = 0; n != count_; n++) {
        if (q_[(i + n) % count_].try_push(tsk, priority)) return;
    }
    q_[i % count_].push(std::move(tsk), priority);
}

int task_system::get_num_threads() const {
    return threads_.size() + 1;
}

int task_system::get_thread_depth() {
    return thread_depth_;
}

void task_system::set_thread_depth(int depth) {
    thread_depth_ = depth;
}

std::unordered_map<std::thread::id, std::size_t> task_system::get_thread_ids() const {
    return thread_ids_;
};
