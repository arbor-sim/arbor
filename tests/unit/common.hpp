#pragma once

/*
 * Convenience functions, structs used across
 * more than one unit test.
 */

namespace testing {

// sentinel for use with range-related tests

struct null_terminated_t {
    bool operator==(const char *p) const { return !*p; }
    bool operator!=(const char *p) const { return !!*p; }

    friend bool operator==(const char *p, null_terminated_t x) {
        return x==p;
    }

    friend bool operator!=(const char *p, null_terminated_t x) {
        return x!=p;
    }

    constexpr null_terminated_t() {}
};

constexpr null_terminated_t null_terminated;

// wrap a value type, with copy operations disabled

template <typename V>
struct nocopy {
    V value;

    nocopy(): value{} {}
    nocopy(V v): value(v) {}
    nocopy(const nocopy& n) = delete;

    nocopy(nocopy&& n) {
        value=n.value;
        n.value=V{};
        ++move_ctor_count;
    }

    nocopy& operator=(const nocopy& n) = delete;
    nocopy& operator=(nocopy&& n) {
        value=n.value;
        n.value=V{};
        ++move_assign_count;
        return *this;
    }

    bool operator==(const nocopy& them) const { return them.value==value; }
    bool operator!=(const nocopy& them) const { return !(*this==them); }

    static int move_ctor_count;
    static int move_assign_count;
    static void reset_counts() {
        move_ctor_count = 0;
        move_assign_count = 0;
    }
};

template <typename V>
int nocopy<V>::move_ctor_count;

template <typename V>
int nocopy<V>::move_assign_count;

// wrap a value type, with move operations disabled

template <typename V>
struct nomove {
    V value;

    nomove(): value{} {}
    nomove(V v): value(v) {}
    nomove(nomove&& n) = delete;

    nomove(const nomove& n): value(n.value) {
        ++copy_ctor_count;
    }

    nomove& operator=(nomove&& n) = delete;

    nomove& operator=(const nomove& n) {
        value=n.value;
        ++copy_assign_count;
        return *this;
    }

    bool operator==(const nomove& them) const { return them.value==value; }
    bool operator!=(const nomove& them) const { return !(*this==them); }

    static int copy_ctor_count;
    static int copy_assign_count;
    static void reset_counts() {
        copy_ctor_count = 0;
        copy_assign_count = 0;
    }
};

template <typename V>
int nomove<V>::copy_ctor_count;

template <typename V>
int nomove<V>::copy_assign_count;

} // namespace testing
