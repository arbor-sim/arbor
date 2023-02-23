#include <cstddef>

#include "memory/memory.hpp"
#include "memory/copy.hpp"
#include "util/meta.hpp"

#pragma once

namespace arb {
namespace gpu {
namespace {
template <typename T>
struct chunk_writer {
    T* end; // device ptr
    const std::size_t stride;

    chunk_writer(T* data, std::size_t stride): end(data), stride(stride) {}

    template <typename Seq, typename = std::enable_if_t<util::is_contiguous_v<Seq>>>
    T* append(Seq&& seq) {
        arb_assert(std::size(seq)==stride);
        return append_freely(std::forward<Seq>(seq));
    }

    template <typename Seq, typename = std::enable_if_t<util::is_contiguous_v<Seq>>>
    T* append_freely(Seq&& seq) {
        std::size_t n = std::size(seq);
        memory::copy(memory::host_view<T>(const_cast<T*>(std::data(seq)), n), memory::device_view<T>(end, n));
        auto p = end;
        end += n;
        return p;
    }

    T* fill(T value) {
        memory::fill(memory::device_view<T>(end, stride), value);
        auto p = end;
        end += stride;
        return p;
    }

    template <typename Seq, typename = std::enable_if_t<util::is_contiguous_v<Seq>>>
    T* append_with_padding(Seq&& seq, typename util::sequence_traits<Seq>::value_type value) {
        std::size_t n = std::size(seq);
        arb_assert(n <= stride);
        std::size_t r = stride - n;
        auto p = append_freely(std::forward<Seq>(seq));
        memory::fill(memory::device_view<typename util::sequence_traits<Seq>::value_type>(end, r), value);
        end += r;
        return p;
    }
};
} // anonymous namespace
} // namespace gpu
} // namespace arb
