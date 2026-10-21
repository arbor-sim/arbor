#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <arbor/context.hpp>
#include <arbor/profile/profiler.hpp>

#include "profile/profiler_macro.hpp"

#ifdef ARB_HAVE_PROFILING

namespace {
void profiled_work() {
    PE(test_outer);
    PE(test_inner);
    PL(test_inner);
    PL(test_outer);
}

// Number of calls of the innermost region of each timer stack named name.
std::size_t count_of(const arb::profile::profile& p, const std::string& name) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < p.stacks.size(); ++i) {
        if (!p.stacks[i].empty() && p.names[p.stacks[i].back()] == name) count += p.counts[i];
    }
    return count;
}

bool check(bool ok, const std::string& what) {
    if (!ok) std::cerr << "failed: " << what << "\n";
    return ok;
}

// Profile, clear, and profile again; true if the results are as expected.
bool clear_keeps_regions() {
    arb::profile::profiler_initialize(arb::make_context());
    arb::profile::profiler_clear();
    profiled_work();
    profiled_work();
    bool ok = check(count_of(arb::profile::profiler_summary(), "test_inner") == 2, "two calls before clear");

    arb::profile::profiler_clear();
    profiled_work();

    auto p = arb::profile::profiler_summary();
    ok &= check(count_of(p, "test_outer") == 1, "one outer call after clear");
    ok &= check(count_of(p, "test_inner") == 1, "one inner call after clear");

    std::stringstream summary;
    arb::profile::print_profiler_summary(summary);
    ok &= check(summary.str().find("test_outer") != std::string::npos, "outer region in summary");
    ok &= check(summary.str().find("test_inner") != std::string::npos, "inner region in summary");
    return ok;
}
} // namespace

// PE/PL cache region indexes, so clearing must reset only the measurements.
//
// The profiler is bound to the threads of the context it is initialized with,
// for the rest of the process. To not break the other tests, run in a child
// process; the threadsafe style starts it afresh rather than forking threads.
TEST(profiler, clear_keeps_regions) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(std::exit(clear_keeps_regions()? 0: 1), ::testing::ExitedWithCode(0), "");
}

// The summary of a root region with several children. Building the tree used to
// copy the root from storage that the copy freed first with libc++, which showed
// up as garbage in the names. Long names, so that they live on the heap.
TEST(profiler, summary_with_several_children) {
    arb::profile::profile p;
    p.names = {"root_region_with_a_long_name", "first_child_with_a_long_name",
               "second_child_with_a_long_name", "third_child_with_a_long_name"};
    p.stacks = {{0}, {0, 1}, {0, 2}, {0, 3}, {0, 1, 2}};
    p.counts = {1, 2, 3, 4, 5};
    p.times = {10, 1, 2, 3, 0.5};
    p.num_threads = 1;

    std::stringstream summary;
    summary << p;
    for (const auto& name: p.names) EXPECT_NE(std::string::npos, summary.str().find(name)) << name;
}

#endif
