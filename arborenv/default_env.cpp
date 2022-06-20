#include <limits>
#include <optional>

#include <arbor/version.hpp>
#include <arborenv/arbenvexcept.hpp>
#include <arborenv/concurrency.hpp>
#include <arborenv/default_env.hpp>

#ifdef ARB_GPU_ENABLED
#include "gpu_api.hpp"
#endif

#include "read_envvar.hpp"

namespace arbenv {

ARB_ARBORENV_API unsigned long default_concurrency() {
    unsigned long env_thread = get_env_num_threads();
    return env_thread? env_thread: thread_concurrency();
}

ARB_ARBORENV_API unsigned long get_env_num_threads() {
    constexpr const char* env_var = "ARBENV_NUM_THREADS";
    std::optional<long long> env_val = read_env_integer(env_var, throw_on_invalid);
    if (!env_val) return 0;

    if (*env_val<1 || static_cast<unsigned long long>(*env_val)>std::numeric_limits<unsigned long>::max()) {
        throw invalid_env_value(env_var, std::getenv(env_var));
    }
    return *env_val;
}

#ifdef ARB_GPU_ENABLED

ARB_ARBORENV_API int default_gpu() {
    constexpr const char* env_var = "ARBENV_GPU_ID";
    int n_device = -1;
    get_device_count(&n_device); // error => leave n_device == -1

    std::optional<long long> env_val = read_env_integer(env_var, throw_on_invalid);
    if (env_val) {
        if (*env_val<0) return -1;
        if (env_val > std::numeric_limits<int>::max()) {
            throw invalid_env_value(env_var, std::getenv(env_var));
        }

        int id = static_cast<int>(*env_val);
        if (id>=n_device) {
            throw arbenv::no_such_gpu(id);
        }

        return id;
    }

    return n_device>0? 0: -1;
}

#else

ARB_ARBORENV_API int default_gpu() {
    return -1;
}

#endif // def ARB_GPU_ENABLED

} // namespace arbenv

