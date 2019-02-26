# FindThreads improperly passes -pthread to nvcc instead of e.g. -Xcompiler=-pthread.
# (see: https://gitlab.kitware.com/cmake/cmake/issues/18008)

function(find_threads_cuda_fix)
    if(TARGET Threads::Threads)
        get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
        if("CUDA" IN_LIST _languages)
            get_property(_threads_options TARGET Threads::Threads PROPERTY INTERFACE_COMPILE_OPTIONS)
            if(_threads_options STREQUAL "-pthread")
                set_property(TARGET Threads::Threads
                    PROPERTY INTERFACE_COMPILE_OPTIONS
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>
                    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>)
            endif()
        endif()
    endif()
endfunction()
