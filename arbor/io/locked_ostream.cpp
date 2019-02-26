#include <ostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

#include "locked_ostream.hpp"

namespace arb {
namespace io {

using tbl_type = std::unordered_map<std::streambuf*, std::weak_ptr<std::mutex>>;

static tbl_type& g_mex_tbl() {
    static tbl_type tbl;
    return tbl;
}

static std::mutex& g_mex_tbl_mex() {
    static std::mutex mex;
    return mex;
}

static std::shared_ptr<std::mutex> register_sbuf(std::streambuf* b) {
    if (b) {
        std::lock_guard<std::mutex> lock(g_mex_tbl_mex());

        auto& wptr = g_mex_tbl()[b];
        auto mex = wptr.lock();
        if (!mex) {
            mex = std::shared_ptr<std::mutex>(new std::mutex);
            wptr = mex;
        }
        return mex;
    }
    else {
        return std::shared_ptr<std::mutex>();
    }
}

static void deregister_sbuf(std::streambuf* b) {
    if (b) {
        std::lock_guard<std::mutex> lock(g_mex_tbl_mex());

        auto i = g_mex_tbl().find(b);
        if (i!=g_mex_tbl().end() && !(i->second.use_count())) {
            g_mex_tbl().erase(i);
        }
    }
}

locked_ostream::locked_ostream(std::streambuf *b):
    std::ostream(b),
    mex(register_sbuf(b))
{}


locked_ostream::locked_ostream(locked_ostream&& other):
    std::ostream(std::move(other)),
    mex(std::move(other.mex))
{
    set_rdbuf(other.rdbuf());
    other.set_rdbuf(nullptr);
}

locked_ostream::~locked_ostream() {
    mex.reset();
    deregister_sbuf(rdbuf());
}

std::unique_lock<std::mutex> locked_ostream::guard() {
    return std::unique_lock<std::mutex>(*mex);
}

} // namespace io
} // namespace arb
