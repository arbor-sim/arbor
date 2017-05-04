#pragma once

#if defined(NMC_USE_LINEAR_CONNECTION_QUEUE)
#include <communication/linear_connection_queue_policy.hpp>
#elif defined(NMC_USE_LINEAR_SPIKE_QUEUE)
#include <communication/linear_spike_queue_policy.hpp>
#elif defined(NMC_USE_GLOBAL_SEARCH_QUEUE)
#include <communication/global_search_queue_policy.hpp>
#elif defined(NMC_USE_DOMAIN_SEARCH_QUEUE)
#include <communication/domain_search_queue_policy.hpp>
#else
#error "No NMC_USE_*_QUEUE defined"
#endif
