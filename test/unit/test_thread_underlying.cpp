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
