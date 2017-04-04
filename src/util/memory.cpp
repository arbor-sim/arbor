#include "memory.hpp"

#if defined(__linux__)
extern "C" {
    #include <malloc.h>
}
#endif

namespace nest {
namespace mc {
namespace util {

#if defined(__linux__)

memory_size_type allocated_memory() {
    auto m = mallinfo();
    return m.hblkhd + m.uordblks;
}

#else

memory_size_type allocated_memory() {
    return -1;
}

#endif

} // namespace util
} // namespace mc
} // namespace nest
