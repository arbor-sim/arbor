#include <atomic>

#include "threading.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;

task notification_queue::try_pop() {
    task tsk;
    lock q_lock{q_mutex_, std::try_to_lock};
    if (q_lock && !q_tasks_.empty()) {
        tsk = std::move(q_tasks_.front());
        q_tasks_.pop_front();
    }
    return tsk;
}

task notification_queue::pop() {
    task tsk;
    lock q_lock{q_mutex_};
    while (q_tasks_.empty() && !quit_) {
        q_tasks_available_.wait(q_lock);
    }
    if (!q_tasks_.empty()) {
        tsk = std::move(q_tasks_.front());
        q_tasks_.pop_front();
    }
    return tsk;
}

bool notification_queue::try_push(task& tsk) {
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;
        q_tasks_.push_front(std::move(tsk));
        tsk = 0;
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(task&& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_front(std::move(tsk));
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
    return q_tasks_.empty();
}

void task_system::run_tasks_loop(int i){
    while (true) {
        task tsk;
        for (unsigned n = 0; n != count_; n++) {
            auto& q = q1_[(i + n) % count_].empty()? q0_[(i + n) % count_]: q1_[(i + n) % count_];
            tsk = q.try_pop();
            if (tsk) break;
        }
        if (!tsk) tsk = q1_[i].pop();
        if (!tsk) tsk = q0_[i].pop();
        if (!tsk) break;
        tsk();
    }
}

void task_system::try_run_task() {
    auto nthreads = get_num_threads();
    task tsk;
    for (int n = 0; n != nthreads; n++) {
        auto& q = q1_[n % nthreads].empty()? q0_[n % nthreads]: q1_[n % nthreads];
        tsk = q.try_pop();
        if (tsk) {
            tsk();
            break;
        }
    }
}

thread_local int task_system::thread_depth_ = 0;

// Default construct with one thread.
task_system::task_system(): task_system(1) {}

task_system::task_system(int nthreads): count_(nthreads), q0_(nthreads), q1_(nthreads) {
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
    for (auto& e: q0_) e.quit();
    for (auto& e: q1_) e.quit();
    for (auto& e: threads_) e.join();
}

void task_system::async(task tsk, bool depth) {
    auto i = index_++;

    auto& q = depth? q1_: q0_;
    for (unsigned n = 0; n != count_; n++) {
        if (q[(i + n) % count_].try_push(tsk)) return;
    }
    q[i % count_].push(std::move(tsk));
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
