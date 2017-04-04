#include "memory.hpp"

#ifdef __linux__
extern "C" {
    #include <malloc.h>
}
#endif

namespace nest {
namespace mc {
namespace util {

#ifdef __linux__

memory_size_type allocated_memory() {
    struct mallinfo m = mallinfo();
    return m.hblkhd + m.uordblks;
}

#else

memory_size_type allocated_memory() {
    return bad_memory_reading;
}

#endif

} // namespace util
} // namespace mc
} // namespace nest
