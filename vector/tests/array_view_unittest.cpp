#include <numeric>

#include "gtest.h"

#include <Vector.hpp>

// check that const views work
TEST(array_view, const_view) {
    using vector = memory::HostVector<int>;
    using view   = vector::view_type;
    using const_view  = vector::const_view_type;

    vector v(10);
    std::iota(v.begin(), v.end(), 0);

    view v_non_const = v;

    {
        const_view v_const(v);
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
    {
        const_view v_const(v_non_const);
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
    {
        const_view v_const(v(memory::all));
        for(auto i : v.range()) {
            EXPECT_EQ(v[i], v_const[i]);
        }
    }
}

// this struct provides const member functions that return
// const_views of private HostVector data
struct A {
    using vector = memory::HostVector<int>;
    using view   = vector::view_type;
    using const_view   = vector::const_view_type;

    A(int size)
    :   data_(size), data_view_(data_)
    { }

    const_view const_data() const {
        return data_;
    }

    const_view const_data_view() const {
        return data_view_;
    }

    private :

    vector data_;
    view data_view_;
};

TEST(array_view, const_members) {
    {
        A a(10);
        auto view1 = a.const_data();
        auto view2 = a.const_data_view();
    }

    {
        const A a(10);
        auto view1 = a.const_data();
        auto view2 = a.const_data_view();
    }
}
