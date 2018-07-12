#include <chrono>
#include <iostream>
#include <thread>

#include <arbor/threadinfo.hpp>
#include <arbor/simulation.hpp>
#include <arbor/execution_context.hpp>

using namespace arb;

struct timer_c {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    static inline time_point tic() {
        return std::chrono::system_clock::now();
    }

    static inline double toc(time_point t) {
        return std::chrono::duration<double>{tic() - t}.count();
    }

    static inline double difference(time_point b, time_point e) {
        return std::chrono::duration<double>{e-b}.count();
    }
};

void run(unsigned long us_per_task, unsigned tasks, execution_context* context) {
    auto duration = std::chrono::microseconds(us_per_task);
    arb::threading::parallel_for::apply(
            0, tasks, get_task_system(&context->task_system_),
            [&](unsigned i){std::this_thread::sleep_for(duration);});
}


int main() {
    const auto nthreads = num_threads();
    execution_context context(nthreads);
    std::cout << "## thread benchmark: " << thread_implementation() << " with " << nthreads << " threads\n";

    const unsigned long us_per_sec = 1000000;

    timer_c timer;

    // Do a dummy run to initialize the threading backend
    run(1000, 1000, &context); // 1000 * 1 ms tasks

    // pre-compute the output strings
    const std::vector<unsigned long> tasks_per_second_per_thread{100, 1000, 10000};
    const auto nruns = tasks_per_second_per_thread.size();

    std::cout << "\ngathering results..." << std::endl;
    std::vector<double> t_run;
    t_run.reserve(nruns);
    for (auto tpspt: tasks_per_second_per_thread) {
        auto us_per_task = us_per_sec/tpspt;
        std::cout << "  μs per task: " << us_per_task << std::endl;
        const auto start = timer.tic();
        run(us_per_task, tpspt*nthreads, &context);
        t_run.push_back(timer.toc(start));
    }

    std::printf("\n## %12s%12s%12s\n",
                "us-per-task", "run", "efficiency");
    for (auto i=0u; i<nruns; ++i) {
        const auto tr = t_run[i];
        unsigned us_per_task = us_per_sec / tasks_per_second_per_thread[i];
        std::printf("## %12u%12.3f%12.1f\n",
                    us_per_task, float(tr), float(1/tr*100.));
    }
}
