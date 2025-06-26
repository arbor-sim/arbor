#include "memory.hpp"
#include <arbor/version.hpp>

#ifdef __linux__
extern "C" {
    #include <malloc.h>
}
#endif

#ifdef __APPLE__
extern "C" {
    #include<mach/mach.h>
}
#endif

#ifdef ARB_GPU_ENABLED
    #include <arbor/gpu/gpu_api.hpp>
#endif

namespace arb {
namespace hw {

#if defined(__linux__) && defined(__GLIBC__)
memory_size_type allocated_memory() {
#if __GLIBC__ > 2 || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 33))
    auto m = mallinfo2();
#else
    auto m = mallinfo();
#endif
    return m.hblkhd + m.uordblks;
}
#elif defined(__APPLE__)
memory_size_type allocated_memory() {
    task_basic_info res;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    auto rc = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t) &res, &t_info_count);
    if (KERN_SUCCESS != rc) return -1;
    // return res.virtual_size;
    return res.resident_size;
}
#else
memory_size_type allocated_memory() {
    return -1;
}
#endif

#ifdef ARB_HAVE_GPU
memory_size_type gpu_allocated_memory() {
    std::size_t free;
    std::size_t total;
    auto success = gpu::device_mem_get_info(&free, &total);

    return success? total-free: -1;
}
#else
memory_size_type gpu_allocated_memory() {
    return -1;
}
#endif

} // namespace hw
} // namespace arb
