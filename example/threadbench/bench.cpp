#include <chrono>
#include <iostream>
#include <thread>

#include <../../arbor/threading/threading.hpp>

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

using namespace arb;

void run(unsigned long us_per_task, unsigned tasks) {
    auto duration = std::chrono::microseconds(us_per_task);
    arb::threading::parallel_for::apply(
            0, tasks,
            [&](unsigned i){std::this_thread::sleep_for(duration);});
}


int main() {
    const auto nthreads = threading::num_threads();
    std::cout << "## thread benchmark: " << threading::description() << " with " << nthreads << " threads\n";

    const unsigned long us_per_sec = 1000000;

    timer_c timer;

    // Do a dummy run to initialize the threading backend
    run(1000, 1000); // 1000 * 1 ms tasks

    // pre-compute the output strings
    const std::vector<unsigned long> tasks_per_second_per_thread{100, 1000, 10000};//, 100000};
    const auto nruns = tasks_per_second_per_thread.size();

    std::cout << "\ngathering results..." << std::endl;
    std::vector<double> t_run;
    t_run.reserve(nruns);
    for (auto tpspt: tasks_per_second_per_thread) {
        auto us_per_task = us_per_sec/tpspt;
        std::cout << "  μs per task: " << us_per_task << std::endl;
        const auto start = timer.tic();
        run(us_per_task, tpspt*nthreads);
        t_run.push_back(timer.toc(start));
    }

    // Perform one thread's worth of work in a for loop.
    // The time taken to perform this work can be used to
    // determine the best case performance for the multithreaded
    // runs.
    std::cout << "\ngathering baseline..." << std::endl;
    std::vector<double> t_baseline;
    t_baseline.reserve(nruns);
    for (auto tpspt: tasks_per_second_per_thread) {
        auto us_per_task = us_per_sec/tpspt;
        std::cout << "  μs per task: " << us_per_task << std::endl;
        auto duration = std::chrono::microseconds(us_per_task);
        const auto start = timer.tic();
        for (auto j=0u; j<tpspt; ++j) {
            std::this_thread::sleep_for(duration);
        }
        t_baseline.push_back(timer.toc(start));
    }

    std::printf("\n## %12s%12s%12s%12s\n",
                "us-per-task", "baseline", "run", "efficiency");
    for (auto i=0u; i<nruns; ++i) {
        const auto tb = t_baseline[i];
        const auto tr = t_run[i];
        unsigned us_per_task = us_per_sec / tasks_per_second_per_thread[i];
        std::printf("## %12u%12.3f%12.3f%12.1f\n",
                    us_per_task, float(tb), float(tr), float(tb/tr*100.));
    }
}