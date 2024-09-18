#include <optional>
#include <stdexcept>
#include <cassert>

#include <arbor/assert.hpp>
#include <arbor/util/scope_exit.hpp>
#include <thread>

#include "threading/threading.hpp"
#include "affinity.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;

priority_task notification_queue::try_pop(int priority) {
    arb_assert(priority < (int)q_tasks_.size());
    lock q_lock{q_mutex_, std::try_to_lock};

    if (q_lock) {
        auto& q = q_tasks_.at(priority);
        if (!q.empty()) {
            priority_task ptsk(std::move(q.front()), priority);
            q.pop_front();
            return ptsk;
        }
    }

    return {};
}

priority_task notification_queue::pop() {
    lock q_lock{q_mutex_};

    while (empty() && !quit_) {
        q_tasks_available_.wait(q_lock);
    }
    for (int pri = n_priority-1; pri>=0; --pri) {
        auto& q = q_tasks_.at(pri);
        if (!q.empty()) {
            priority_task ptsk{std::move(q.front()), pri};
            q.pop_front();
            return ptsk;
        }
    }
    return {};
}

bool notification_queue::try_push(priority_task& ptsk) {
    arb_assert(ptsk.priority < (int)q_tasks_.size());
    {
        lock q_lock{q_mutex_, std::try_to_lock};
        if (!q_lock) return false;

        q_tasks_.at(ptsk.priority).push_front(ptsk.release());
    }
    q_tasks_available_.notify_all();
    return true;
}

void notification_queue::push(priority_task&& ptsk) {
    arb_assert(ptsk.priority < (int)q_tasks_.size());
    {
        lock q_lock{q_mutex_};
        q_tasks_.at(ptsk.priority).push_front(ptsk.release());
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

void task_system::run(priority_task ptsk) {
    arb_assert(ptsk);
    auto guard = util::on_scope_exit([pri = current_task_priority_] { current_task_priority_ = pri; });

    current_task_priority_ = ptsk.priority;
    ptsk.run();
}

void task_system::run_tasks_loop(int index) {
    auto guard = util::on_scope_exit([] { current_task_queue_ = nil; });
    current_task_queue_ = index;
    if (bind_) set_affinity(index, count_, affinity_kind::thread);
    while (true) {
        priority_task ptsk;
        // Loop over the levels of priority starting from highest to lowest
        for (int pri = n_priority-1; pri>=0; --pri) {
            // Loop over the threads trying to pop a task of the requested priority.
            for (int n = 0; n < count_; ++n) {
                ptsk = q_[(index + n) % count_].try_pop(pri);
                if (ptsk) break;
            }
            if (ptsk) break;
        }
        // If a task can not be acquired, force a pop from the queue. This is a blocking action.
        if (!ptsk) ptsk = q_[index].pop();
        if (!ptsk) break;

        run(std::move(ptsk));
    }
}

void task_system::try_run_task(int lowest_priority) {
    unsigned i = current_task_queue_+1==0? 0: current_task_queue_;
    arb_assert(i>=0 && i<count_);

    // Loop over the levels of priority starting from highest to lowest_priority
    for (int pri = n_priority-1; pri>=lowest_priority; --pri) {
        // Loop over the threads trying to pop a task of the requested priority.
        for (int n = 0; n != count_; n++) {
            if (auto ptsk = q_[(i + n) % count_].try_pop(pri)) {
                run(std::move(ptsk));
                return;
            }
        }
    }
}

thread_local int task_system::current_task_priority_ = -1;
thread_local unsigned task_system::current_task_queue_ = task_system::nil;

// Default construct with one thread.
task_system::task_system(): task_system(1) {}

task_system::task_system(int nthreads, bool bind):
    count_(nthreads),
    bind_(bind),
    q_(nthreads) {
    if (count_ <= 0) throw std::runtime_error("Non-positive number of threads in thread pool");

    for (unsigned p = 0; p<n_priority; ++p) {
        index_[p] = 0;
    }

    // Main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;
    current_task_queue_ = 0;

    // Bind the master thread
    if (bind_) set_affinity(0, count_, affinity_kind::thread);

    for (int i = 1; i < count_; i++) {
        threads_.emplace_back([this, i]{run_tasks_loop(i);});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_system::~task_system() {
    current_task_priority_ = -1;
    current_task_queue_ = nil;
    for (auto& e: q_) e.quit();
    for (auto& e: threads_) e.join();
}

void task_system::async(priority_task ptsk) {
    assert(count_ != 0);
    if (count_ == 0) throw std::runtime_error("Non-positive number of threads in thread pool");
    auto priority = ptsk.priority;
    if (priority >= n_priority) {
        run(std::move(ptsk));
    }
    else {
        auto idx = index_[priority]++;
        for (int n = 0; n < count_; n++) {
            if (q_[(idx + n) % count_].try_push(ptsk)) return;
        }
        q_[idx % count_].push(std::move(ptsk));
    }
}

std::unordered_map<std::thread::id, std::size_t> task_system::get_thread_ids() const {
    return thread_ids_;
};

std::optional<std::size_t> task_system::get_current_thread_id() const {
    const auto it = thread_ids_.find(std::this_thread::get_id());
    if(it != thread_ids_.end()) return it->second;
    return std::nullopt;
}
