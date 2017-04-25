#pragma once

#include <cstdint>

namespace nest {
namespace mc {
namespace util {

// Use a signed type to store memory sizes because it can be used to store
// the difference between two readings, which may be negative.
// A 64 bit type is large enough to store any amount of memory that will
// reasonably be used.
using memory_size_type = std::int64_t;

// Returns the amount of memory currently allocated in bytes.
// Returns a negative value on error, or if the operation is not supported on
// the target architecture.
memory_size_type allocated_memory();

// Returns the amount of memory currently allocated on the gpu in bytes.
// Returns a negative value on error, or if not using the gpu
memory_size_type gpu_allocated_memory();

} // namespace util
} // namespace mc
} // namespace nest
