#include <cassert>
#include <exception>
#include <iostream>

#include "cthread.hpp"


using namespace nest::mc::threading::impl;

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

[[noreturn]]
static void terminate(const char *const msg) {
    std::cerr << "NMC_NUM_THREADS_ERROR: " << msg << std::endl;
    std::terminate();
}

// should check string, throw exception on missing or badly formed
static size_t global_get_num_threads() {
    const char* nthreads_str;
    // select variable to use:
    //   If NMC_NUM_THREADS_VAR is set, use $NMC_NUM_THREADS_VAR
    //   else if NMC_NUM_THREAD set, use it
    //   else if OMP_NUM_THREADS set, use it
    if (auto nthreads_var_name = std::getenv("NMC_NUM_THREADS_VAR")) {
        nthreads_str = std::getenv(nthreads_var_name);
    }
    else if (! (nthreads_str = std::getenv("NMC_NUM_THREADS"))) {
        nthreads_str = std::getenv("OMP_NUM_THREADS");
    }

    // If the selected var is unset,
    //   or no var is set,
    //   error
    if (! nthreads_str) {
        terminate("No environmental var defined");
     }

    // only composed of spaces*digits*space*
    auto nthreads_str_end{nthreads_str};
    while (std::isspace(*nthreads_str_end)) {
        ++nthreads_str_end;
    }
    while (std::isdigit(*nthreads_str_end)) {
        ++nthreads_str_end;
    }
    while (std::isspace(*nthreads_str_end)) {
        ++nthreads_str_end;
    }
    if (*nthreads_str_end) {
        terminate("Num threads is not a single integer");
    }

    // and it's got a single non-zero value
    auto nthreads{std::atoi(nthreads_str)};
    if (! nthreads) {
        terminate("Num threads is not a non-zero number");
    }
  
    return nthreads;
}

task_pool& task_pool::get_global_task_pool() {
    static task_pool global_task_pool{global_get_num_threads()};
    return global_task_pool;
}
