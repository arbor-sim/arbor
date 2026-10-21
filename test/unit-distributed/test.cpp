#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#if __has_include(<execinfo.h>) && __has_include(<pthread.h>) && __has_include(<cxxabi.h>)
#include <cxxabi.h>
#include <execinfo.h>
#include <pthread.h>
#define ARB_TEST_BACKTRACE
#endif

#include <gtest/gtest.h>

#include <arbor/context.hpp>

#include <sup/ioutil.hpp>
#include <tinyopt/tinyopt.h>

#ifdef TEST_MPI
#include <arborenv/with_mpi.hpp>
#endif

#include "distributed_context.hpp"
#include "execution_context.hpp"

#include "distributed_listener.hpp"

#include "test.hpp"

using namespace arb;

context g_context = make_context();

namespace {
// Fails fast if a test hangs, e.g. because ranks wait in different collectives,
// instead of running into the CI job timeout without any output. If a test runs
// longer than ARB_TEST_TIMEOUT seconds (default 600), every rank prints the test
// and a backtrace of its main thread, and exits, which makes mpirun stop all ranks.
class watchdog: public testing::EmptyTestEventListener {
public:
    explicit watchdog(int rank): rank_(rank), limit_(timeout_from_env()) {
#ifdef ARB_TEST_BACKTRACE
        main_thread_ = pthread_self();
        // The first call may allocate, which must not happen in the signal handler.
        void* frame;
        backtrace(&frame, 1);
        std::signal(SIGUSR1, record_backtrace);
#endif
        thread_ = std::thread([this] { watch(); });
    }

    ~watchdog() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void OnTestStart(const testing::TestInfo& info) override {
        std::lock_guard<std::mutex> lock(mutex_);
        test_ = std::string(info.test_suite_name()) + "." + info.name();
        deadline_ = clock::now() + limit_;
        cv_.notify_one();
    }

    void OnTestEnd(const testing::TestInfo&) override {
        std::lock_guard<std::mutex> lock(mutex_);
        deadline_.reset();
    }

private:
    using clock = std::chrono::steady_clock;

    static std::chrono::seconds timeout_from_env() {
        auto env = std::getenv("ARB_TEST_TIMEOUT");
        long seconds = env? std::atol(env): 0;
        return std::chrono::seconds(seconds > 0? seconds: 600);
    }

    void watch() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!done_) {
            if (!deadline_) {
                cv_.wait(lock);
                continue;
            }
            auto deadline = *deadline_;
            cv_.wait_until(lock, deadline);
            // Woken for another reason: done, or the test ended or changed.
            if (done_ || deadline_ != deadline || clock::now() < deadline) continue;

            std::fprintf(stderr, "rank %d: test %s did not finish within %lld s\n",
                         rank_, test_.c_str(), (long long)limit_.count());
            print_main_thread_backtrace();
            std::fflush(stderr);
            std::_Exit(124);
        }
    }

#ifdef ARB_TEST_BACKTRACE
    static constexpr int max_frames = 64;
    static inline void* frames_[max_frames];
    static inline volatile std::sig_atomic_t num_frames_ = 0;

    // Signal handler, runs on the main thread: only record the return addresses.
    static void record_backtrace(int) {
        num_frames_ = backtrace(frames_, max_frames);
    }

    // Demangle the C++ symbol in a line of backtrace_symbols, if any.
    static std::string demangle(const std::string& line) {
        auto begin = line.find("_Z");
        if (begin == std::string::npos) return line;
        auto end = line.find_first_of(" +)", begin);
        auto mangled = line.substr(begin, end == std::string::npos? std::string::npos: end - begin);
        int status = 0;
        char* name = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
        if (status != 0 || !name) return line;
        std::string result = line.substr(0, begin) + name + (end == std::string::npos? "": line.substr(end));
        std::free(name);
        return result;
    }

    void print_main_thread_backtrace() {
        pthread_kill(main_thread_, SIGUSR1);
        for (int i = 0; i < 100 && !num_frames_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        int n = num_frames_;
        if (!n) {
            std::fprintf(stderr, "rank %d: no backtrace of the main thread\n", rank_);
            return;
        }
        char** symbols = backtrace_symbols(frames_, n);
        std::fprintf(stderr, "rank %d: backtrace of the main thread:\n", rank_);
        for (int i = 0; i < n; ++i) {
            std::fprintf(stderr, "rank %d:   %s\n", rank_, symbols? demangle(symbols[i]).c_str(): "?");
        }
        std::free(symbols);
    }

    pthread_t main_thread_;
#else
    void print_main_thread_backtrace() {
        std::fprintf(stderr, "rank %d: backtraces not supported on this platform\n", rank_);
    }
#endif

    int rank_;
    std::chrono::seconds limit_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    std::string test_;
    std::optional<clock::time_point> deadline_;
    std::thread thread_;
};
} // namespace

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -d, --dryrun        Number of dry run ranks\n"
"  -h, --help          Display usage information and exit\n";

int main(int argc, char **argv) {
    proc_allocation alloc;
    alloc.gpu_id = -1;

#ifdef TEST_MPI
    arbenv::with_mpi guard(argc, argv, false);
    g_context = arb::make_context(alloc, MPI_COMM_WORLD);
#elif defined(TEST_LOCAL)
    g_context = arb::make_context(alloc);
#else
#error "define TEST_MPI or TEST_LOCAL for distributed test"
#endif

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    // replace original printer with our custom printer
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new distributed_listener("run_"+g_context->distributed->name(), g_context));
    listeners.Append(new watchdog(g_context->distributed->id()));

    int return_value = 0;
    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto comm_size = to::parse<unsigned>(arg, "-d", "--dryrun")) {
                if (*comm_size==0) {
                    throw to::user_option_error("number of dry run ranks must be positive");
                }
                // Note that this must be set again for each test that uses a different
                // number of cells per domain, e.g.
                //      policy::set_sizes(policy::size(), new_cells_per_rank)
                // TODO: fix when dry run mode reimplemented
                //policy::set_sizes(*comm_size, 0);
            }
            else if (to::parse(arg, "-h", "--help")) {
                to::usage(argv[0], usage_str);
                return 0;
            }
            else {
                throw to::option_error("unrecognized option", *arg);
            }
        }

        // record the local return value for tests run on this mpi rank
        //      0 : success
        //      1 : failure
        return_value = RUN_ALL_TESTS();
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return_value = 1;
    }
    catch (std::exception& e) {
        std::cout << "caught exception: " << e.what() << std::endl;
        return_value = 1;
    }

    // perform global collective, to ensure that all ranks return
    // the same exit code
    return g_context->distributed->max(return_value);
}
