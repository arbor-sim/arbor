#include "../gtest.h"
#include "common.hpp"

#include <iostream>
#include <ostream>
// (Pending abstraction of threading interface)
#include <arbor/version.hpp>

#include "threading/threading.hpp"
#include "threading/enumerable_thread_specific.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;
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
        auto duration = std::chrono::microseconds(100);
        std::this_thread::sleep_for(duration);
    }
};

struct ftor_parallel_wait {

    ftor_parallel_wait(task_system* ts): ts{ts} {}

    void operator()() const {
        auto nthreads = ts->get_num_threads();
        auto duration = std::chrono::microseconds(100);
        parallel_for::apply(0, nthreads, ts, [=](int i){ std::this_thread::sleep_for(duration);});
    }

    task_system* ts;
};

}

TEST(task_system, test_copy) {
    task_system ts;

    ftor f;
    ts.async(f);

    // Copy into new ftor and move ftor into a task (std::function<void()>)
    EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(task_system, test_move) {
    task_system ts;

    ftor f;
    ts.async(std::move(f));

    // Move into new ftor and move ftor into a task (std::function<void()>)
    EXPECT_LE(nmove, 2);
    EXPECT_LE(ncopy, 1);
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
    EXPECT_LE(nmove, 2);
    EXPECT_LE(ncopy, 1);
    reset();
}

TEST(task_group, test_copy) {
    task_system ts;
    task_group g(&ts);

    ftor f;
    g.run(f);
    g.wait();

    // Copy into "wrap" and move wrap into a task (std::function<void()>)
    EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(task_group, test_move) {
    task_system ts;
    task_group g(&ts);

    ftor f;
    g.run(std::move(f));
    g.wait();

    // Move into wrap and move wrap into a task (std::function<void()>)
    EXPECT_LE(nmove, 2);
    EXPECT_LE(ncopy, 1);
    reset();
}

TEST(task_group, individual_tasks) {
    // Simple check for deadlock
    task_system ts;
    task_group g(&ts);

    auto nthreads = ts.get_num_threads();

    ftor_wait f;
    for (int i = 0; i < 32 * nthreads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_for_sleep) {
    // Simple check for deadlock for nested parallelism
    task_system ts;
    auto nthreads = ts.get_num_threads();
    task_group g(&ts);

    ftor_parallel_wait f(&ts);
    for (int i = 0; i < nthreads; i++) {
        g.run(f);
    }
    g.wait();
}

TEST(task_group, parallel_for) {
    task_system ts;
    for (int n = 0; n < 10000; n=!n?1:2*n) {
        std::vector<int> v(n, -1);
        parallel_for::apply(0, n, &ts, [&](int i) {v[i] = i;});
        for (int i = 0; i< n; i++) {
            EXPECT_EQ(i, v[i]);
        }
    }
}

TEST(task_group, nested_parallel_for) {
    task_system ts;
    for (int m = 1; m < 512; m*=2) {
        for (int n = 0; n < 1000; n=!n?1:2*n) {
            std::vector<std::vector<int>> v(n, std::vector<int>(m, -1));
            parallel_for::apply(0, n, &ts, [&](int i) {
                auto &w = v[i];
                parallel_for::apply(0, m, &ts, [&](int j) { w[j] = i + j; });
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
    task_system_handle ts = task_system_handle(new task_system);
    enumerable_thread_specific<int> buffers(ts);
    task_group g(ts.get());

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
