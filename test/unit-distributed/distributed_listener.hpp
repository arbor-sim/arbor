#pragma once

#include <fstream>
#include <string>
#include <utility>

#include <arbor/context.hpp>

#include "../gtest.h"

/// A specialized listener desinged for printing test results with MPI
/// or in other distributed contexts.
///
/// When tests are run with e.g. MPI, one instance of each test is run on
/// each rank. The default behavior of Google Test is for each test
/// instance to print to stdout. With more than one rank, this creates
/// the usual mess of output.
///
/// This specialization has the first rank (rank 0) print to stdout, and all
/// ranks print their output to separate text files.
/// For each test a message is printed showing:
///     - detailed messages about errors on rank 0,
///     - a head count of errors that occured on other ranks.

class distributed_listener: public testing::EmptyTestEventListener {
    using UnitTest = testing::UnitTest;
    using TestCase = testing::TestCase;
    using TestInfo = testing::TestInfo;
    using TestPartResult = testing::TestPartResult;

public:
    distributed_listener(std::string f_base, const arb::context &ctx);

    /// Messages that are printed at the start and end of the test program.
    /// i.e. once only.
    virtual void OnTestProgramStart(const UnitTest&) override;
    virtual void OnTestProgramEnd(const UnitTest&) override;

    /// Messages that are printed at the start and end of each test case.
    /// On startup a counter that counts the number of tests that fail in
    /// this test case is initialized to zero, and will be incremented for each
    /// test that fails.
    virtual void OnTestCaseStart(const TestCase& test_case) override;
    virtual void OnTestCaseEnd(const TestCase& test_case) override;

    // Called before a test starts.
    virtual void OnTestStart(const TestInfo& test_info) override;

    // Called after a failed assertion or a SUCCEED() invocation.
    virtual void OnTestPartResult(const TestPartResult& test_part_result) override;

    // Called after a test ends.
    virtual void OnTestEnd(const TestInfo& test_info) override;

private:
    struct printer {
        std::ofstream fid_;
        bool cout_;

        printer() = default;
        printer(std::string base_name, int rank);
    };

    template <typename T>
    friend printer& operator<<(printer&, const T&);

    const arb::context& context_;
    int rank_;
    int size_;
    bool mpi_;
    printer emit_;

    int test_case_failures_;
    int test_case_tests_;
    int test_failures_;
};

