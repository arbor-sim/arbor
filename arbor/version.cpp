#include <arbor/version.hpp>
#include <arbor/export.hpp>
#include <string>
#include <map>
#include <variant>

namespace arb {
ARB_ARBOR_API const char* source_id = ARB_SOURCE_ID;
ARB_ARBOR_API const char* arch = ARB_ARCH;
ARB_ARBOR_API const char* build_config = ARB_BUILD_CONFIG;
ARB_ARBOR_API const char* version = ARB_VERSION;
#ifdef ARB_VERSION_DEV
ARB_ARBOR_API const char* version_dev = ARB_VERSION_DEV;
#else
ARB_ARBOR_API const char* version_dev = "";
#endif
ARB_ARBOR_API const char* full_build_id = ARB_FULL_BUILD_ID;

using config_map = std::map<std::string, std::variant<std::string, bool, int>>;

// returns Arbor configuration map
ARB_ARBOR_API config_map get_arbor_config() {
    config_map config;

    #ifdef ARB_MPI_ENABLED
        config["mpi"] = true;
    #else
        config["mpi"] = false;
    #endif
    #ifdef ARB_NVCC_ENABLED
        config["cuda"] = true;
    #endif
    #ifdef ARB_CUDA_CLANG_ENABLED
        config["cuda-clang"] = true;
    #endif
    #ifdef ARB_HIP_ENABLED
        config["hip"] = true;
    #endif
    #ifndef ARB_GPU_ENABLED
        config["gpu"] = true;
    #endif
    #ifdef ARB_VECTORIZE_ENABLED
        config["vectorize"] = true;
    #else
        config["vectorize"] = false;
    #endif
    #ifdef ARB_PROFILE_ENABLED
        config["profiiling"] = true;
    #else
        config["profiiling"] = false;
    #endif
    #ifdef ARB_NEUROML_ENABLED
        config["neuroml"] = true;
    #else
        config["neuroml"] = false;
    #endif
    #ifdef ARB_BUNDLED_ENABLED
        config["bundled"] = true;
    #else
        config["bundled"] = true;
    #endif
    config["version"] = arb::version;
    config["source"] = arb::source_id;
    config["build_config"] = arb::build_config;
    config["arch"] = arb::arch;

    return config;
}

// return pretty-printed string representation of the Arbor configuration map
ARB_ARBOR_API std::string get_arbor_config_str() {
    config_map config = get_arbor_config();
    std::string config_str = "";

    struct value_str {
        std::string operator()(const std::string& value) const {
            return "'" + value + "'";
        }

        std::string operator()(bool value) const {
            return value ? "true" : "false";
        }

        std::string operator()(int value) const {
            return std::to_string(value);
        }
    };

    for (auto it = config.begin(); it != config.end(); ++it) {
        config_str += it->first + "=" + std::visit(value_str{}, it->second);

        if (std::next(it) != config.end()) {
            config_str += ", ";
        }
    }

    return config_str;
}

// return JSON representation of the Arbor configuration map
ARB_ARBOR_API std::string get_arbor_config_json() {
    config_map config = get_arbor_config();
    std::string config_json_str = "{";

    struct value_str {
        std::string operator()(const std::string& value) const {
            return "\"" + value + "\"";
        }

        std::string operator()(bool value) const {
            return value ? "true" : "false";
        }

        std::string operator()(int value) const {
            return std::to_string(value);
        }
    };

    for (auto it = config.begin(); it != config.end(); ++it) {
        config_json_str += "\"" + it->first + "\" : " + std::visit(value_str{}, it->second);

        if (std::next(it) != config.end()) {
            config_json_str += ", ";
        }
    }
    config_json_str += "}";

    return config_json_str;
}

// return JSON representation of the Arbor configuration map
/*ARB_ARBOR_API std::string get_arbor_config_json() {
    config_map config = get_arbor_config();
    
    nlohmann::json config_json = config;
    std::string config_json_str = config_json.dump();

    return config_json_str;
}*/
}