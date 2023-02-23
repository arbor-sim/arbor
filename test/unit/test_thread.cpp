#include <gtest/gtest.h>
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
    int duration_us = 100;

    ftor_wait() {}
    ftor_wait(int us): duration_us(us) {}

    void operator()() const {
        auto duration = std::chrono::microseconds(duration_us);
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
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);

        ftor f;
        ts.async(f, 0);

        // Copy into new ftor and move ftor into a task (std::function<void()>)
        // move ctor is elided with some compilers
        //EXPECT_EQ(1, nmove);
        EXPECT_EQ(1, ncopy);
        reset();
    }
}

TEST(task_system, test_move) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);

        ftor f;
        ts.async(std::move(f), 0);

        // Move into new ftor and move ftor into a task (std::function<void()>)
        EXPECT_LE(nmove, 2);
        EXPECT_LE(ncopy, 1);
        reset();
    }
}

TEST(notification_queue, test_copy) {
    notification_queue q;

    ftor f;
    q.push({task(f), 0});

    // Copy into new ftor and move ftor into a task (std::function<void()>)
    // move ctor is elided with some compilers
    //EXPECT_EQ(1, nmove);
    EXPECT_EQ(1, ncopy);
    reset();
}

TEST(notification_queue, test_move) {
    notification_queue q;

    ftor f;

    // Move into new ftor and move ftor into a task (std::function<void()>)
    q.push({task(std::move(f)), 1});
    EXPECT_LE(nmove, 2);
    EXPECT_LE(ncopy, 1);
    reset();
}

TEST(task_group, test_copy) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);

        ftor f;
        g.run(f);
        g.wait();

        // Copy into "wrap" and move wrap into a task (std::function<void()>)
        EXPECT_EQ(1, nmove);
        EXPECT_EQ(1, ncopy);
        reset();
    }
}

TEST(task_group, test_move) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);

        ftor f;
        g.run(std::move(f));
        g.wait();

        // Move into wrap and move wrap into a task (std::function<void()>)
        EXPECT_LE(nmove, 2);
        EXPECT_LE(ncopy, 1);
        reset();
    }
}

TEST(task_group, individual_tasks) {
    // Simple check for deadlock
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);

        ftor_wait f;
        for (int i = 0; i < 32 * nthreads; i++) {
            g.run(f);
        }
        g.wait();
    }
}

TEST(task_group, parallel_for_sleep) {
    // Simple check for deadlock for nested parallelism
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        task_group g(&ts);

        ftor_parallel_wait f(&ts);
        for (int i = 0; i < nthreads; i++) {
            g.run(f);
        }
        g.wait();
    }
}

TEST(task_group, parallel_for) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int n = 0; n < 10000; n = !n ? 1 : 2 * n) {
            std::vector<int> v(n, -1);
            parallel_for::apply(0, n, &ts, [&](int i) { v[i] = i; });
            for (int i = 0; i < n; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
    }
}


TEST(task_group, manual_nested_parallel_for) {
    // Check for deadlock or stack overflow
    const int ntasks = 100000;
    {
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            std::vector<int> v(ntasks);
            task_system ts(nthreads);

            auto nested = [&](int j) {
              task_group g1(&ts);
              g1.run([&](){v[j] = j;});
              g1.wait();
            };

            task_group g0(&ts);
            for (int i = 0; i < ntasks; i++) {
                g0.run([=](){nested(i);});
            }
            g0.wait();
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
    }
    {
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            std::vector<int> v(ntasks);
            task_system ts(nthreads);

            auto double_nested = [&](int i) {
              task_group g2(&ts);
              g2.run([&](){v[i] = i;});
              g2.wait();
            };

            auto nested = [&](int i) {
              task_group g1(&ts);
              g1.run([=](){double_nested(i);});
              g1.wait();
            };

            task_group g0(&ts);
            for (int i = 0; i < ntasks; i++) {
                g0.run([=](){nested(i);});
            }
            g0.wait();
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
    }
}

TEST(task_group, nested_parallel_for) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 512; m *= 2) {
            for (int n = 0; n < 1000; n = !n ? 1 : 2 * n) {
                std::vector<std::vector<int>> v(n, std::vector<int>(m, -1));
                parallel_for::apply(0, n, &ts, [&](int i) {
                    auto& w = v[i];
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
}

TEST(task_group, multi_nested_parallel_for) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system ts(nthreads);
        for (int m = 1; m < 512; m *= 2) {
            for (int n = 0; n < 128; n = !n ? 1 : 2 * n) {
                for (int p = 0; p < 16; p = !p ? 1 : 2 * p) {
                    std::vector<std::vector<std::vector<int>>> v(n, std::vector<std::vector<int>>(m, std::vector<int>(p, -1)));
                    parallel_for::apply(0, n, &ts, [&](int i) {
                        auto& w = v[i];
                        parallel_for::apply(0, m, &ts, [&](int j) {
                            auto& u = w[j];
                            parallel_for::apply(0, p, &ts, [&](int k) {
                                u[k] = i + j + k;
                            });
                        });
                    });
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < m; j++) {
                            for (int k = 0; k < p; k++) {
                                EXPECT_EQ(i + j + k, v[i][j][k]);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(task_group, nested_parallel_for_unbalanced) {
    // Top level parallel for has many more tasks than lower level
    const int ntasks = 100000;
    {
        // Default batching
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            task_system ts(nthreads);
            std::vector<int> v(ntasks);
            parallel_for::apply(0, ntasks, &ts, [&](int i) {
                parallel_for::apply(0, 1, &ts, [&](int j) { v[i] = i; });
            });
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
        // 128 tasks per batch
        const int batch_size = 128;
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            task_system ts(nthreads);
            std::vector<int> v(ntasks);
            parallel_for::apply(0, ntasks, batch_size, &ts, [&](int i) {
              parallel_for::apply(0, 1, batch_size, &ts, [&](int j) { v[i] = i; });
            });
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
    }
    // lower level parallel for has many more tasks than top level
    {
        // Default batching
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            task_system ts(nthreads);
            std::vector<int> v(ntasks);
            parallel_for::apply(0, 1, &ts, [&](int i) {
                parallel_for::apply(0, ntasks, &ts, [&](int j) { v[j] = j; });
            });
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
        // 128 tasks per batch
        const int batch_size = 128;
        for (int nthreads = 1; nthreads < 20; nthreads *= 4) {
            task_system ts(nthreads);
            std::vector<int> v(ntasks);
            parallel_for::apply(0, 1, batch_size, &ts, [&](int i) {
                parallel_for::apply(0, ntasks, batch_size, &ts, [&](int j) { v[j] = j; });
            });
            for (int i = 0; i < ntasks; i++) {
                EXPECT_EQ(i, v[i]);
            }
        }
    }
}

TEST(task_group, multi_nested_parallel_for_unbalanced) {
    // Top level parallel for has many more tasks than lower level
    const int ntasks = 100000;
    for (int nthreads = 1; nthreads < 20; nthreads*=4) {
        task_system ts(nthreads);
        std::vector<int> v(ntasks);
        parallel_for::apply(0, ntasks, &ts, [&](int i) {
            parallel_for::apply(0, 1, &ts, [&](int j) {
                parallel_for::apply(0, 1, &ts, [&](int k) {
                    v[i] = i;
                });
            });
        });
        for (int i = 0; i < ntasks; i++) {
            EXPECT_EQ(i, v[i]);
        }
    }
    // lower level parallel for has many more tasks than top level
    for (int nthreads = 1; nthreads < 20; nthreads*=4) {
        task_system ts(nthreads);
        std::vector<int> v(ntasks);
        parallel_for::apply(0, 1, &ts, [&](int i) {
            parallel_for::apply(0, 1, &ts, [&](int j) {
                parallel_for::apply(0, ntasks, &ts, [&](int k) {
                    v[k] = k;
                });
            });
        });
        for (int i = 0; i < ntasks; i++) {
            EXPECT_EQ(i, v[i]);
        }
    }
}

TEST(enumerable_thread_specific, test) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        task_system_handle ts = task_system_handle(new task_system(nthreads));
        enumerable_thread_specific<int> buffers(ts);
        task_group g(ts.get());

        for (int i = 0; i < 100000; i++) {
            g.run([&]() {
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
}
