#include "../gtest.h"
#include "common.hpp"

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

int nmove = 0;
int ncopy = 0;

void reset() {
    nmove = 0;
    ncopy = 0;
}

void print() {
    std::cout << "moves " << nmove << ", copies " << ncopy << "\n";
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

TEST(task_group, copy_task) {
    arb::threading::task_group g;

    ftor f;
    g.run(f);
    g.wait();

    print();
    EXPECT_LE(nmove,  2);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(task_group, move_task) {
    arb::threading::task_group g;

    ftor f;
    g.run(std::move(f));
    g.wait();

    EXPECT_LE(nmove,  3);
    EXPECT_EQ(0, ncopy);
    reset();
}

// TODO:
// * test pushing tasks straight into task_system/queue (i.e. no task_group wrapping)
// * tests for known deadlock causes
// * queues
// * enumerable_thread_specific
//      * default construction
//      * launch 100k tiny tasks that increment bucket, and check that sum(buckets) = 100k
