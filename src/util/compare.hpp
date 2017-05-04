#pragma once

namespace nest {
namespace mc {
namespace util {

template<typename E>
struct lessthan {
    template<typename T1, typename T2>
    bool operator() (const T1& t1, const T2& t2) const {
        return E()(t1) < E()(t2);
    }
};

template<typename E>
struct greaterthan {
    template<typename T1, typename T2>
    bool operator() (const T1& t1, const T2& t2) const {
        return E()(t1) > E()(t2);
    }
};

template<typename E>
struct equalto {
    template<typename T1, typename T2>
    bool operator() (const T1& t1, const T2& t2) const {
        return E()(t1) == E()(t2);
    }
};

}
}
}
