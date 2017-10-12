#ifndef MINIAPP
#define MINIAPP

#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <json/json.hpp>

#include <common_types.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <io/exporter_spike_file.hpp>
#include <model.hpp>
#include <profiling/profiler.hpp>
#include <profiling/meter_manager.hpp>
#include <threading/threading.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <util/ioutil.hpp>
#include <util/nop.hpp>

#include "io.hpp"
#include "miniapp_recipes.hpp"
#include "trace_sampler.hpp"

int main(int argc, char** argv);

#endif

