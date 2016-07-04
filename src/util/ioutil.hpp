#pragma once

#include <ios>

namespace nest {
namespace mc {
namespace util {

class iosfmt_guard {
public:
    explicit iosfmt_guard(std::ios &stream): save_(nullptr), stream_(stream) {
        save_.copyfmt(stream_);
    }

    ~iosfmt_guard() {
        stream_.copyfmt(save_);
    }

private:
    std::ios save_;
    std::ios& stream_;
};


} // namespace util
} // namespace mc
} // namespace nest

