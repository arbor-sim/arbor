#pragma once

namespace memory {
namespace gpu {

void fill8(uint8_t* v, uint8_t value, std::size_t n);
void fill16(uint16_t* v, uint16_t value, std::size_t n);
void fill32(uint32_t* v, uint32_t value, std::size_t n);
void fill64(uint64_t* v, uint64_t value, std::size_t n);

} // namespace gpu
} // namespace memory
