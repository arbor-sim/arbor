#pragma once

// Eliminated successive identical elements in-place in a SequenceConteiner.
// (Can wrap or be replaced by std::unique in C++17.)
//
// If we ever implement a unique_view, it should go here too.

#include <utility>

namespace arb {
namespace util {

template <typename SeqContainer, typename EqPred = std::equal_to<>>
void unique_in_place(SeqContainer& c, EqPred eq = EqPred{}) {
    if (c.empty()) return;

    auto end = c.end();
    auto write = c.begin();
    auto read = write;

    while (++read!=end) {
        if (eq(*read, *write)) continue;
        if (++write!=read) *write = std::move(*read);
    }

    c.erase(++write, end);
}

} // namespace util
} // namespace arb
