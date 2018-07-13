#include "../gtest.h"
#include "common.hpp"
#include <arbor/threadinfo.hpp>

#include <iostream>
#include <ostream>
// (Pending abstraction of threading interface)
#include <arbor/version.hpp>

#if defined(ARB_CTHREAD_ENABLED)
#include "threading/cthread.hpp"

using namespace arb::threading::impl;
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

TEST(task_system, test_copy) {
    task_system &ts = arb::threading::impl::task_system::get_global_task_system();

    ftor f;
    ts.async_(f);

    // Copy into new ftor and move ftor into a task (std::function<void()>)
    EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(task_system, test_move) {
    task_system &s = arb::threading::impl::task_system::get_global_task_system();

    ftor f;
    s.async_(std::move(f));

    // Move into new ftor and move ftor into a task (std::function<void()>)
    EXPECT_EQ(2, nmove);
    EXPECT_EQ(0, ncopy);
    reset();
}

TEST(notification_queue, test_copy) {
    notification_queue q;

    ftor f;
    q.push(f);

    // Copy into new ftor and move ftor into a task (std::function<void()>)
    EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(notification_queue, test_move) {
    notification_queue q;

    ftor f;

    // Move into new ftor and move ftor into a task (std::function<void()>)
    q.push(std::move(f));
    EXPECT_EQ(2, nmove);
    EXPECT_EQ(0, ncopy);
    reset();
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
    for (int i = 0; i < 32 * num_threads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_for_sleep) {
    // Simple check for deadlock for nested parallelism
    arb::threading::task_group g;
    auto num_threads = arb::num_threads();

    ftor_parallel_wait f;
    for (int i = 0; i < num_threads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_for) {

    for (int n = 0; n < 10000; n=!n?1:2*n) {
        std::vector<int> v(n, -1);
        arb::threading::parallel_for::apply(0, n, [&](int i) {v[i] = i;});
        for (int i = 0; i< n; i++) {
            EXPECT_EQ(i, v[i]);
        }
    }
}

TEST(task_group, nested_parallel_for) {

    for (int m = 1; m < 512; m*=2) {
        for (int n = 0; n < 1000; n=!n?1:2*n) {
            std::vector<std::vector<int>> v(n, std::vector<int>(m, -1));
            arb::threading::parallel_for::apply(0, n, [&](int i) {
                auto &w = v[i];
                arb::threading::parallel_for::apply(0, m, [&](int j) { w[j] = i + j; });
            });
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    EXPECT_EQ(i + j, v[i][j]);
                }
            }
        }
    }
}

TEST(enumerable_thread_specific, test) {
    arb::threading::enumerable_thread_specific<int> buffers(0);
    arb::threading::task_group g;

    for (int i = 0; i < 100000; i++) {
        g.run([&](){
            auto& buf = buffers.local();
            buf++;
        });
    }
    g.wait();

    int sum = 0;
    for (auto b: buffers) {
        sum += b;
    }

    EXPECT_EQ(100000, sum);
}

#endif
