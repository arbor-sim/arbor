#include "../gtest.h"
#include "common.hpp"
#include <arbor/threadinfo.hpp>

#include <iostream>
#include <ostream>
// (Pending abstraction of threading interface)
#include <arbor/version.hpp>
#if defined(ARB_TBB_ENABLED)
    #include "threading/tbb.hpp"
#elif defined(ARB_CTHREAD_ENABLED)
    #include "threading/cthread.hpp"
#else
    #include "threading/serial.hpp"
#endif

namespace {

std::atomic<int> nmove{0};
std::atomic<int> ncopy{0};

void reset() {
    nmove = 0;
    ncopy = 0;
}

struct ftor {

    ftor() {}

    ftor(ftor&& other) {
        ++nmove;
    }

    ftor(const ftor& other) {
        ++ncopy;
    }

    void operator()() const {}
};

struct ftor_wait {

    ftor_wait() {}

    void operator()() const {
        auto duration = std::chrono::microseconds(500);
        std::this_thread::sleep_for(duration);
    }
};

struct ftor_parallel_wait {

    ftor_parallel_wait() {}

    void operator()() const {
        auto num_threads = arb::num_threads();
        auto duration = std::chrono::microseconds(500);
        arb::threading::parallel_for::apply(0, num_threads, [=](int i){ std::this_thread::sleep_for(duration);});
    }
};

}

TEST(task_group, copy_task) {
    arb::threading::task_group g;

    ftor f;
    g.run(f);
    g.wait();

    // Copy into "wrap" and move wrap into a task (std::function<void()>)
    EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(task_group, move_task) {
    arb::threading::task_group g;

    ftor f;
    g.run(std::move(f));
    g.wait();

    // Move into wrap and move wrap into a task (std::function<void()>)
    EXPECT_EQ(2, nmove);
    EXPECT_EQ(0, ncopy);
    reset();
}

TEST(task_group, individual_tasks) {
    // Simple check for deadlock
    arb::threading::task_group g;
    auto num_threads = arb::num_threads();

    ftor_wait f;
    for (int i = 0; i < 48 * num_threads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_for_tasks) {
    // Simple check for deadlock for nested parallelism
    arb::threading::task_group g;
    auto num_threads = arb::num_threads();

    ftor_parallel_wait f;
    for (int i = 0; i < 48 * num_threads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_sum_vector) {
    // Check for correctness
    arb::threading::task_group g;
    auto num_threads = arb::num_threads();
    auto total_size = num_threads*num_threads;

    std::vector<int> v, sum;
    sum.reserve(num_threads);

    for (int i = 1; i <= total_size; i++) {
        v.push_back(i);
    }

    for (int i = 0; i < num_threads; i++) {
        g.run([i, num_threads, v, &sum]{
            int temp = 0;
            for (int j = i*num_threads; j < (i+1)*num_threads; j++) {
                temp += v[j];
            }
            sum[i] = temp;
        });
    }
    g.wait();

    int final_sum = 0;
    for (int i = 0; i< num_threads; i++) {
        final_sum += sum[i];
    }

    EXPECT_EQ(final_sum, (total_size*(total_size + 1)/2));
}

TEST(task_group, parallel_for_sum_vector) {
    // Check for correctness
    arb::threading::task_group g;
    auto num_threads = arb::num_threads();
    auto total_size = num_threads*num_threads;

    std::vector<int> v, sum;
    sum.reserve(num_threads);

    for (int i = 1; i <= total_size; i++) {
        v.push_back(i);
    }

    arb::threading::parallel_for::apply(0, num_threads, [num_threads, v, &sum](int i)
            {
                int temp = 0;
                for (int j = i*num_threads; j < (i+1)*num_threads; j++) {
                    temp += v[j];
                }
                sum[i] = temp;
            });

    int final_sum = 0;
    for (int i = 0; i< num_threads; i++) {
        final_sum += sum[i];
    }

    EXPECT_EQ(final_sum, (total_size*(total_size + 1)/2));
}
