#include <stdexcept>
#include <string>

#include <arborenv/arbenvexcept.hpp>

using namespace std::literals;

namespace arbenv {
using std::string;

invalid_env_value::invalid_env_value(const std::string& variable, const std::string& value):
    arborenv_exception("environment variable \""s + variable + R"(" has invalid value ")" + value + "\""s),
    env_variable(variable),
    env_value(value)
{}

// GPU enumeration, selection.

no_such_gpu::no_such_gpu(int gpu_id):
    arborenv_exception("no gpu with id "s+std::to_string(gpu_id)),
    gpu_id(gpu_id)
{}

gpu_uuid_error::gpu_uuid_error(const string& what)
    : arborenv_exception("error determining GPU uuids: "s + what) {}

} // namespace arbenv
