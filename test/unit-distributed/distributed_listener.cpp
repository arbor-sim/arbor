#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <communication/distributed_context.hpp>

#include "../gtest.h"

#include "distributed_listener.hpp"

distributed_listener::printer::printer(std::string base_name, int rank) {
    base_name += "_"+std::to_string(rank)+".txt";

    fid_.open(base_name);
    if (!fid_) {
        throw std::runtime_error("could not open file " + base_name + " for test output");
    }

    cout_ = rank==0;
}

template <typename T>
distributed_listener::printer& operator<<(distributed_listener::printer& p, const T& item) {
    if (p.fid_) p.fid_ << item;
    if (p.cout_) std::cout << item;
    return p;
}

distributed_listener::distributed_listener(std::string f_base, const arb::distributed_context* ctx):
    context_(ctx),
    rank_(context_->id()),
    size_(context_->size()),
    emit_(std::move(f_base), rank_)
{}

void distributed_listener::OnTestProgramStart(const UnitTest&) {
    emit_ << "*** test output for rank " << rank_ << " of " << size_ << "\n\n";
}

void distributed_listener::OnTestProgramEnd(const UnitTest&) {
    emit_ << "*** end test output for rank " << rank_ << " of " << size_ << "\n\n";
}

void distributed_listener::OnTestCaseStart(const TestCase& test_case) {
    test_case_failures_ = 0;
    test_case_tests_ = 0;
}

void distributed_listener::OnTestCaseEnd(const TestCase& test_case) {
    emit_
        << "    PASSED " << test_case_tests_-test_case_failures_
        << " of " << test_case_tests_ << "tests"
        << " in " << test_case.name() << "\n";

    if (test_case_failures_>0) {
        emit_
            << "    FAILED " << test_case_failures_
            << " of " << test_case_tests_ << "tests"
            << " in " << test_case.name() << "\n";
    }

    emit_ << "\n";
}

void distributed_listener::OnTestStart(const TestInfo& test_info) {
    emit_
        << "TEST:  " << test_info.test_case_name()
        << "::" << test_info.name() << "\n";

    test_failures_ = 0;
}

void distributed_listener::OnTestPartResult(const TestPartResult& test_part_result) {
    // indent all lines in the summary by 4 spaces
    std::string summary = "    " + std::string(test_part_result.summary());
    auto pos = summary.find("\n");
    while (pos!=summary.size() && pos!=std::string::npos) {
        summary.replace(pos, 1, "\n    ");
        pos = summary.find("\n", pos+1);
    }

    emit_
        << " LOCAL_" << (test_part_result.failed()? "FAIL": "SUCCESS") << "\n"
        << test_part_result.file_name() << ':' << test_part_result.line_number() << "\n"
        << summary << "\n";

    // note that there was a failure in this test case
    if (test_part_result.failed()) {
        ++test_failures_;
    }
}

void distributed_listener::OnTestEnd(const TestInfo& test_info) {
    ++test_case_tests_;

    // count the number of ranks that had errors
    int global_errors = context_->sum(test_failures_>0 ? 1 : 0);
    if (global_errors>0) {
        ++test_case_failures_;
        emit_ << "  GLOBAL_FAIL on " << global_errors << "ranks\n";
    }
}

