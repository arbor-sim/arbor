#pragma once

#include <map>

#include <arbor/assert.hpp>

namespace arb {
namespace util {

// A map for containers
// Caches the containers to avoid unneccessary re-allocations in case of reuse.
template<
    class Key,
    class Container,
    class Compare = std::less<Key>
    //class Allocator = std::allocator<std::pair<const Key, T>
>
class container_map
{
private:
    using map_type = std::map<Key, Container, Compare>;

public:
    using key_type               = Key;
    using mapped_type            = Container;
    using value_type             = typename map_type::value_type;
    using size_type 	         = typename map_type::size_type;
    using difference_type        = typename map_type::difference_type;
    using key_compare            = typename map_type::key_compare;
    using allocator_type         = typename map_type::allocator_type;
    using reference              = typename map_type::reference;
    using const_reference        = typename map_type::const_reference;
    using pointer                = typename map_type::pointer;
    using const_pointer          = typename map_type::const_pointer;
    using iterator               = typename map_type::iterator;
    using const_iterator         = typename map_type::const_iterator;
    using reverse_iterator       = typename map_type::reverse_iterator;
    using const_reverse_iterator = typename map_type::const_reverse_iterator;

public: // ctors
    container_map() {}

    container_map(const container_map&) = default;
    container_map(container_map&&) = default;

    container_map (std::initializer_list<value_type> init) : m_{std::move(init)} {}

    container_map& operator=(const container_map&) = default;
    container_map& operator=(container_map&&) noexcept = default;

public: // element access
    Container& operator[](const Key& key) {
        if (auto it = m_.find(key); it!=end())
            return it->second;
        if (auto it = m_cache_.find(key); it!=m_cache_.end())
            return m_.insert(m_cache_.extract(it)).position->second;
        return m_.try_emplace(key).first->second;
    }
    Container& operator[](Key&& key) {
        if (auto it = m_.find(key); it!=end())
            return it->second;
        if (auto it = m_cache_.find(key); it!=m_cache_.end())
            return m_.insert(m_cache_.extract(it)).position->second;
        return m_.try_emplace(std::move(key)).first->second;
    }

public: // iterators
    iterator begin() noexcept { return m_.begin(); }
    const_iterator begin() const noexcept { return m_.begin(); }
    const_iterator cbegin() const noexcept { return m_.cbegin(); }
    iterator end() noexcept { return m_.end(); }
    const_iterator end() const noexcept { return m_.end(); }
    const_iterator cend() const noexcept { return m_.cend(); }

    reverse_iterator rbegin() noexcept { return m_.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return m_.rbegin(); }
    const_reverse_iterator crbegin() const noexcept { return m_.crbegin(); }
    reverse_iterator rend() noexcept { return m_.rend(); }
    const_reverse_iterator rend() const noexcept { return m_.rend(); }
    const_reverse_iterator crend() const noexcept { return m_.crend(); }

public: // capacity
    [[nodiscard]] bool empty() const noexcept { return m_.empty(); }
    size_type size() const noexcept { return m_.size(); }
    size_type aggregate_size() const noexcept {
        size_type s = 0u;
        for (const auto& c : m_) s += c.second.size();
        return s;
    }
    size_type max_size() const noexcept { return m_.max_size(); }

public: // modifiers
    void clear() {
        auto iter=m_.begin();
        while(iter != m_.end()) {
            auto i = iter++;
            auto r = m_cache_.insert(m_.extract(i));
            arb_assert(r.inserted);
            r.position->second.clear();
        }
        arb_assert(m_.size() == 0);
    }

public: // lookup
    iterator find(const Key& key ) { return m_.find(key); }
    const_iterator find(const Key& key) const { return m_.find(key); }

private:
    map_type m_;
    map_type m_cache_;
};

} // namespace util
} // namespace arb
