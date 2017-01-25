#include "cthread.hpp"
#include <cassert>

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
    std::swap(tsk, pool._tasks.front());
    pool._tasks.pop_front();
    
    lck.unlock();
    pool._tasks_available.notify_all();
}

// Release task
// Call unlocked, returns unlocked
task_pool::run_task::~run_task() {
    lck.lock();
    tsk.second->in_flight--;
    
    lck.unlock();
    pool._tasks_available.notify_all();
}

// runs forever until quit is true
void task_pool::run_tasks_loop() {
    lock lck = {_tasks_mutex, std::defer_lock};
    while (true) {
        lck.lock();

        while (! _quit && _tasks.empty()) {
            _tasks_available.wait(lck);
        }
        if (_quit) {
            return;
        }
    
        run_task run{*this, lck};
        run.tsk.first();
    }
}

// run until out of tasks for a group
void task_pool::run_tasks_while(task_group* g) {
    lock lck{_tasks_mutex, std::defer_lock};
    while (true) {  
        lck.lock();

        while (! _quit && g->in_flight && _tasks.empty()) {
            _tasks_available.wait(lck);
        }
        if (_quit || ! g->in_flight) {
            return;
        }

        run_task run{*this, lck};
        run.tsk.first();
    }
}

// Create pool and threads
// new threads are nthreads-1
task_pool::task_pool(std::size_t nthreads):
    _tasks_mutex{},
    _tasks_available{},
    _tasks{},
    _threads{}
{
    assert(nthreads > 0);
  
    // now for the main thread
    auto tid = std::this_thread::get_id();
    _thread_ids[tid] = 0;
  
    // and go from there
    for (std::size_t i = 1; i < nthreads; i++) {
        _threads.emplace_back([this]{run_tasks_loop();});
        tid = _threads.back().get_id();
        _thread_ids[tid] = i;
    }
}

task_pool::~task_pool() {
    {
        lock lck{_tasks_mutex};
        _quit = true;
    }
    _tasks_available.notify_all();
    
    for (auto& thread: _threads) {
        thread.join();
    }
}

// push a task into pool
void task_pool::run(const task& tsk) {
    {
        lock lck{_tasks_mutex};
        _tasks.push_back(tsk);
        tsk.second->in_flight++;
    }
    _tasks_available.notify_all();
}

void task_pool::run(task&& tsk) {
  {
      lock lck{_tasks_mutex};
      _tasks.push_back(std::move(tsk));
      tsk.second->in_flight++;
  }
  _tasks_available.notify_all();
}

// call on main thread
// uses this thread to run tasks
// and waits until the entire task
// queue is cleared
void task_pool::wait(task_group* g) {
    run_tasks_while(g);
}

class NMC_NUM_THREADS_ERROR {};
static auto NMC_NUM_THREADS_VAR{"NMC_NUM_THREADS_VAR"};
static auto NMC_NUM_THREADS{"NMC_NUM_THREADS"};
static auto OMP_NUM_THREADS{"OMP_NUM_THREADS"};

// should check string, throw exception on missing or badly formed
static size_t global_get_num_threads() {
    const char* nthreads_str;
    // select variable to use:
    //   If NMC_NUM_THREADS_VAR is set, use $NMC_NUM_THREADS_VAR
    //   else if NMC_NUM_THREAD set, use it
    //   else if OMP_NUM_THREADS set, use it
    if (auto nthreads_var_name = std::getenv(NMC_NUM_THREADS_VAR)) {
        nthreads_str = std::getenv(nthreads_var_name);
    }
    else if (! (nthreads_str = std::getenv(NMC_NUM_THREADS))) {
        nthreads_str = std::getenv(OMP_NUM_THREADS);
    }

    // If the selected var is unset,
    //   or no var is set,
    //   error
    if (! nthreads_str) {
        throw NMC_NUM_THREADS_ERROR();
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
        throw NMC_NUM_THREADS_ERROR();
    }

    // and it's got a single non-zero value
    auto nthreads{std::atoi(nthreads_str)};
    if (! nthreads) {
        throw NMC_NUM_THREADS_ERROR();
    }
  
    return nthreads;
}

task_pool& task_pool::get_global_task_pool() {
    static task_pool global_task_pool{global_get_num_threads()};
    return global_task_pool;
}
