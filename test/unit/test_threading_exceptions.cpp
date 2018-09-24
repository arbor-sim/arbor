#include "../gtest.h"

#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/context.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/version.hpp>

#include "threading/threading.hpp"

using namespace arb::threading::impl;
using namespace arb::threading;
using namespace arb;

auto duration = std::chrono::nanoseconds(1);

struct error {
    int code;
    error(int c): code(c) {};
};

TEST(test_exception, single_task_no_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);
        g.run([](){ std::this_thread::sleep_for(duration); });
        EXPECT_NO_THROW(g.wait());
    }
}

TEST(test_exception, single_task_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(1);
        task_group g(&ts);
        g.run([](){ throw error(0);});
        try {
            g.wait();
            FAIL() << "Expected exception";
        }
        catch (error &e) {
            EXPECT_EQ(e.code, 0);
        }
        catch (...) {
            FAIL() << "Expected error type";
        }
    }
}

TEST(test_exception, many_tasks_no_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);
        for (int i = 1; i < 100; i++) {
            for (int j = 0; j < i; j++) {
                g.run([j](){ std::this_thread::sleep_for(duration); });
            }
            EXPECT_NO_THROW(g.wait());
        }
    }
}

TEST(test_exception, many_tasks_one_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);
        for (int i = 1; i < 100; i++) {
            for (int j = 0; j < i; j++) {
                g.run([j](){ if(j==0) {throw error(j);} });
            }
            try {
                g.wait();
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_EQ(e.code, 0);
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, many_tasks_many_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);
        for (int i = 1; i < 100; i++) {
            for (int j = 0; j < i; j++) {
                g.run([j](){ if(j%5 == 0) {throw error(j);} });
            }
            try {
                g.wait();
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_EQ(e.code%5, 0);
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, many_tasks_all_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);
        for (int i = 1; i < 100; i++) {
            for (int j = 0; j < i; j++) {
                g.run([j](){ throw error(j); });
            }
            try {
                g.wait();
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_TRUE((e.code >= 0) && (e.code < i));
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, parallel_for_no_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int n = 1; n < 100; n*=2) {
            EXPECT_NO_THROW(parallel_for::apply(0, n, &ts, [n](int i) { std::this_thread::sleep_for(duration); }));
        }
    }
}

TEST(test_exception, parallel_for_one_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int n = 1; n < 100; n*=2) {
            try {
                parallel_for::apply(0, n, &ts, [n](int i) { if(i==n-1) {throw error(i);} });
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_TRUE(e.code == n-1);
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, parallel_for_many_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int n = 1; n < 100; n*=2) {
            try {
                parallel_for::apply(0, n, &ts, [n](int i) { if(i%7 == 0) {throw error(i);} });
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_EQ(e.code%7, 0);
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, parallel_for_all_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int n = 1; n < 100; n*=2) {
            try {
                parallel_for::apply(0, n, &ts, [n](int i) { throw error(i); });
                FAIL() << "Expected exception";
            }
            catch (error &e) {
                EXPECT_TRUE((e.code >= 0) && (e.code < n));
            }
            catch (...) {
                FAIL() << "Expected error type";
            }
        }
    }
}

TEST(test_exception, nested_parallel_for_no_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 50; m*=2) {
            for (int n = 0; n < 50; n = !n?1:2*n) {
                parallel_for::apply(0, n, &ts, [&](int i) {
                    EXPECT_NO_THROW(parallel_for::apply(0, m, &ts, [](int j) { std::this_thread::sleep_for(duration); }));
                });
            }
        }
    }
}

TEST(test_exception, nested_parallel_for_one_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 100; m*=2) {
            for (int n = 0; n < 100; n = !n?1:2*n) {
                parallel_for::apply(0, n, &ts, [&](int i) {
                    try {
                        parallel_for::apply(0, m, &ts, [](int j) { if (j == 0) {throw error(j);} });
                        FAIL() << "Expected exception";
                    }
                    catch (error &e) {
                        EXPECT_EQ(e.code, 0);
                    }
                    catch (...) {
                        FAIL() << "Expected error type";
                    }
                });
            }
        }
    }
}

TEST(test_exception, nested_parallel_for_many_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 100; m*=2) {
            for (int n = 0; n < 100; n = !n?1:2*n) {
                parallel_for::apply(0, n, &ts, [&](int i) {
                    try {
                        parallel_for::apply(0, m, &ts, [](int j) { if (j%10 == 0) {throw error(j);} });
                        FAIL() << "Expected exception";
                    }
                    catch (error &e) {
                        EXPECT_TRUE(e.code%10==0);
                    }
                    catch (...) {
                        FAIL() << "Expected error type";
                    }
                });
            }
        }
    }
}

TEST(test_exception, nested_parallel_for_all_throw) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 100; m*=2) {
            for (int n = 0; n < 100; n = !n?1:2*n) {
                parallel_for::apply(0, n, &ts, [&](int i) {
                    try {
                        parallel_for::apply(0, m, &ts, [](int j) { throw error(j); });
                        FAIL() << "Expected exception";
                    }
                    catch (error &e) {
                        EXPECT_TRUE(e.code >= 0 && e.code < m);
                    }
                    catch (...) {
                        FAIL() << "Expected error type";
                    }
                });
            }
        }
    }
}
