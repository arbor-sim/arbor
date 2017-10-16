#pragma once

#if !defined(NMC_HAVE_CTHREAD)
    #error "this header can only be loaded if NMC_HAVE_CTHREAD is set"
#endif

// task_group definition
#include "cthread_impl.hpp"

// and sorts use cthread_parallel_stable_sort
#include "cthread_sort.hpp"

static size_t global_get_num_threads();