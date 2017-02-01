#pragma once

#include <chrono>

namespace nest {
namespace mc {
namespace threading {
namespace impl{

struct timer {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    static inline time_point tic() {
        return std::chrono::system_clock::now();
    }

    static inline double toc(time_point t) {
        return std::chrono::duration<double>{tic() - t}.count();
    }

    static inline double difference(time_point b, time_point e) {
        return std::chrono::duration<double>{e-b}.count();
    }
};

}
}
}
}
