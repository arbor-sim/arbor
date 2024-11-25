#pragma once

// Lockable ostream over a provided streambuf.

#include <ostream>
#include <memory>
#include <mutex>

namespace arb {
namespace io {

struct locked_ostream: std::ostream {
    locked_ostream(std::streambuf *b);
    locked_ostream(locked_ostream&& other);

    ~locked_ostream();

    std::unique_lock<std::mutex> guard();

private:
    std::shared_ptr<std::mutex> mex;
};

} // namespace io
} // namespace arb
