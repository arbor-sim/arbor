#pragma once

// Format a sequence delimitted by some separator.

#include <iostream>

namespace arb {
namespace io {

namespace impl {
    template <typename Seq, typename Separator>
    struct sepval {
        const Seq& seq;
        Separator sep;

        sepval(const Seq& seq, Separator sep): seq(seq), sep(std::move(sep)) {}

        friend std::ostream& operator<<(std::ostream& out, const sepval& sv) {
            bool emitsep = false;
            for (const auto& v: sv.seq) {
                if (emitsep) out << sv.sep;
                emitsep = true;
                out << v;
            }
            return out;
        }
    };
}

// Adapt a sequence with arbitrary delimiter.

template <typename Seq, typename Separator>
impl::sepval<Seq, Separator> sepval(const Seq& seq, Separator sep) {
    return impl::sepval<Seq, Separator>(seq, std::move(sep));
}

// Adapt a sequence with delimiter ", ".

template <typename Seq>
impl::sepval<Seq, const char*> csv(const Seq& seq) {
    return sepval(seq, ", ");
}

} // namespace io
} // namespace arb

