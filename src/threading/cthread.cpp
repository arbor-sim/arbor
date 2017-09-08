#include <atomic>
#include <cassert>
#include <cstring>
#include <exception>
#include <iostream>
#include <regex>

#include "cthread.hpp"
#include "threading.hpp"

using namespace nest::mc::threading::impl;
using namespace nest::mc;

// RAII owner for a task in flight
struct task_pool::run_task {
    task_pool& pool;
    lock& lck;
    task tsk;

    run_task(task_pool&, lock&);
    ~run_task();
};

// Own a task in flight
// lock should be passed locked,
// and will be unlocked after call
task_pool::run_task::run_task(task_pool& pool, lock& lck):
    pool{pool},
    lck{lck},
    tsk{}
{
    std::swap(tsk, pool.tasks_.front());
    pool.tasks_.pop_front();

    lck.unlock();
    pool.tasks_available_.notify_all();
}

// Release task
// Call unlocked, returns unlocked
task_pool::run_task::~run_task() {
    lck.lock();
    tsk.second->in_flight--;

    lck.unlock();
    pool.tasks_available_.notify_all();
}

template<typename B>
void task_pool::run_tasks_loop(B finished) {
    lock lck{tasks_mutex_, std::defer_lock};
    while (true) {
        lck.lock();

        while (! quit_ && tasks_.empty() && ! finished()) {
            tasks_available_.wait(lck);
        }
        if (quit_ || finished()) {
            return;
        }

        run_task run{*this, lck};
        run.tsk.first();
    }
}

// runs forever until quit is true
void task_pool::run_tasks_forever() {
    run_tasks_loop([] {return false;});
}

// run until out of tasks for a group
void task_pool::run_tasks_while(task_group* g) {
    run_tasks_loop([=] {return ! g->in_flight;});
}

// Create pool and threads
// new threads are nthreads-1
task_pool::task_pool(std::size_t nthreads):
    tasks_mutex_{},
    tasks_available_{},
    tasks_{},
    threads_{}
{
    assert(nthreads > 0);

    // now for the main thread
    auto tid = std::this_thread::get_id();
    thread_ids_[tid] = 0;

    // and go from there
    for (std::size_t i = 1; i < nthreads; i++) {
        threads_.emplace_back([this]{run_tasks_forever();});
        tid = threads_.back().get_id();
        thread_ids_[tid] = i;
    }
}

task_pool::~task_pool() {
    {
        lock lck{tasks_mutex_};
        quit_ = true;
    }
    tasks_available_.notify_all();

    for (auto& thread: threads_) {
        thread.join();
    }
}

// push a task into pool
void task_pool::run(const task& tsk) {
    {
        lock lck{tasks_mutex_};
        tasks_.push_back(tsk);
        tsk.second->in_flight++;
    }
    tasks_available_.notify_all();
}

void task_pool::run(task&& tsk) {
  {
      lock lck{tasks_mutex_};
      tasks_.push_back(std::move(tsk));
      tsk.second->in_flight++;
  }
  tasks_available_.notify_all();
}

// call on main thread
// uses this thread to run tasks
// and waits until the entire task
// queue is cleared
void task_pool::wait(task_group* g) {
    run_tasks_while(g);
}

task_pool& task_pool::get_global_task_pool() {
    auto num_threads = threading::num_threads();
    static task_pool global_task_pool(num_threads);
    return global_task_pool;
}
