#pragma once

// Represent an mcable_list where each cable has an associated value.
//
// The only mutating operations are insert, emplace, and clear.

#include <algorithm>
#include <optional>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

template <typename T>
struct mcable_map {
    // Forward subset of vector interface:

    using value_type = std::pair<mcable, T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using const_iterator = typename std::vector<value_type>::const_iterator;
    using iterator = const_iterator;

    using const_reverse_iterator = typename std::vector<value_type>::const_reverse_iterator;
    using reverse_iterator = const_reverse_iterator;

    const_iterator begin() const noexcept { return elements_.cbegin(); }
    const_iterator cbegin() const noexcept { return elements_.cbegin(); }

    const_reverse_iterator rbegin() const noexcept { return elements_.crbegin(); }
    const_reverse_iterator crbegin() const noexcept { return elements_.crbegin(); }

    const_iterator end() const noexcept { return elements_.cend(); }
    const_iterator cend() const noexcept { return elements_.cend(); }

    const_reverse_iterator rend() const noexcept { return elements_.crend(); }
    const_reverse_iterator crend() const noexcept { return elements_.crend(); }

    const value_type& operator[](size_type i) const { return elements_[i]; }

    decltype(auto) front() const { return *begin(); }
    decltype(auto) back() const { return *rbegin(); }

    std::size_t size() const noexcept { return elements_.size(); }
    bool empty() const noexcept { return !size(); }

    void clear() { elements_.clear(); }

    // mcable_map-specific operations:

    // Insertion is successful iff c intersects with cables in at most two discrete points.
    // insert() and emplace() return true on success.

    bool insert(const mcable& c, T value) {
        auto opt_it = insertion_point(c);
        if (!opt_it) return false;
        elements_.emplace(*opt_it, c, std::move(value));
        assert_invariants();
        return true;
    }

    template <typename... Args>
    bool emplace(const mcable& c, Args&&... args) {
        auto opt_it = insertion_point(c);
        if (!opt_it) return false;
        elements_.emplace(*opt_it,
           std::piecewise_construct,
           std::forward_as_tuple(c),
           std::forward_as_tuple(std::forward<Args>(args)...));
        assert_invariants();
        return true;
    }

    mcable_list support() const {
        mcable_list s;
        s.reserve(elements_.size());
        std::transform(elements_.begin(), elements_.end(), std::back_inserter(s),
            [](const auto& x) { return x.first; });
        return s;
    }

private:
    std::vector<value_type> elements_;

    std::optional<typename std::vector<value_type>::iterator> insertion_point(const mcable& c) {
        struct as_mcable {
            mcable value;
            as_mcable(const value_type& x): value(x.first) {}
            as_mcable(const mcable& x): value(x) {}
        };

        auto it = std::lower_bound(elements_.begin(), elements_.end(), c,
            [](as_mcable a, as_mcable b) { return a.value<b.value; });

        if (it!=elements_.begin()) {
            mcable prior = std::prev(it)->first;
            if (prior.branch==c.branch && prior.dist_pos>c.prox_pos) {
                return std::nullopt;
            }
        }
        if (it!=elements_.end()) {
            mcable next = it->first;
            if (c.branch==next.branch && c.dist_pos>next.prox_pos) {
                return std::nullopt;
            }
        }
        return it;
    }

    void assert_invariants() {
        arb_assert(test_invariants(support()));
    }
};

} // namespace arb
