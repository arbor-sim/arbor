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
        q_tasks_.push_back(std::move(tsk));
        tsk = 0;
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(task&& tsk) {
    {
        lock q_lock{q_mutex_};
        q_tasks_.push_back(std::move(tsk));
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

void task_system::run_tasks_loop(int i){
    while (true) {
        task tsk;
        for (unsigned n = 0; n != count_; n++) {
            tsk = q_[(i + n) % count_].try_pop();
            if (tsk) break;
        }
        if (!tsk) tsk = q_[i].pop();
        if (!tsk) break;
        tsk();
    }
}

void task_system::try_run_task() {
    auto nthreads = get_num_threads();
    task tsk;
    for (int n = 0; n != nthreads; n++) {
        tsk = q_[n % nthreads].try_pop();
        if (tsk) {
            tsk();
            break;
        }
    }
}

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
}

void task_system::async(task tsk) {
    auto i = index_++;

    for (unsigned n = 0; n != count_; n++) {
        if (q_[(i + n) % count_].try_push(tsk)) return;
    }
    q_[i % count_].push(std::move(tsk));
}

int task_system::get_num_threads() const {
    return threads_.size() + 1;
}

std::unordered_map<std::thread::id, std::size_t> task_system::get_thread_ids() const {
    return thread_ids_;
};
