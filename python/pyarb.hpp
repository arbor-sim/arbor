#pragma once

// Common module-global data for use by the pyarb implementation.

#include <functional>
#include <memory>
#include <typeinfo>
#include <unordered_map>

#include <arbor/arbexcept.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

#include <pybind11/pybind11.h>

// Version check
#define mk_tok(x) #x
#define mk_ver(M, m, p) mk_tok(M) "." mk_tok(m) "." mk_tok(p)
#define PB11_ERR(M, m, p) "Required version of pybind11 is 2.8.1 <= version < 3.0.0 Found " mk_ver(M, m, p)
static_assert((PYBIND11_VERSION_HEX >= 0x02080100)
              &&
              (PYBIND11_VERSION_HEX  < 0x03000000),
              PB11_ERR(PYBIND11_VERSION_MAJOR, PYBIND11_VERSION_MINOR, PYBIND11_VERSION_PATCH));
#undef PB11_ERR
#undef mk_ver
#undef mk_tok

namespace pyarb {

// Sample recorder object interface.

struct sample_recorder {
    virtual void record(arb::util::any_ptr meta, std::size_t n_sample, const arb::sample_record* records) = 0;
    virtual pybind11::object samples() const = 0;
    virtual pybind11::object meta() const = 0;
    virtual void reset() = 0;
    virtual ~sample_recorder() {}
};

// Recorder 'factory' type: given an any_ptr to probe metadata of a specific subset of types,
// return a corresponding sample_recorder instance.

using sample_recorder_factory = std::function<std::unique_ptr<sample_recorder> (arb::util::any_ptr)>;

// Holds map: probe metadata pointer type → recorder object factory.

struct recorder_factory_map {
    std::unordered_map<std::type_index, sample_recorder_factory> map_;

    template <typename Meta>
    void assign(sample_recorder_factory rf) {
        map_[typeid(const Meta*)] = std::move(rf);
    }

    std::unique_ptr<sample_recorder> make_recorder(arb::util::any_ptr meta) const {
        try {
            return map_.at(meta.type())(meta);
        }
        catch (std::out_of_range&) {
            std::string ty = meta.type().name();
            throw arb::arbor_internal_error("unrecognized probe metadata type " + ty);
        }
    }
};

// Probe metadata to Python object converter.

using probe_meta_converter = std::function<pybind11::object (arb::util::any_ptr)>;

struct probe_meta_cvt_map {
    std::unordered_map<std::type_index, probe_meta_converter> map_;

    template <typename Meta>
    void assign(probe_meta_converter cvt) {
        map_[typeid(const Meta*)] = std::move(cvt);
    }

    pybind11::object convert(arb::util::any_ptr meta) const {
        if (auto iter = map_.find(meta.type()); iter!=map_.end()) {
            return iter->second(meta);
        }
        else {
            return pybind11::none();
        }
    }
};

// Collection of module-global data.

struct pyarb_global {
    recorder_factory_map recorder_factories;
    probe_meta_cvt_map probe_meta_converters;
};

using pyarb_global_ptr = std::shared_ptr<pyarb_global>;

} // namespace pyarb
