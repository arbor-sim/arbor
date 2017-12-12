from libc.stdlib cimport malloc, free
from cpython cimport bool

####### _stringify ##############
# Wraps a "main" signature function
# with a conversion from python objects
#
ctypedef int _Func(int, char**)
#
cdef bool _stringify(_Func func, str arg0, tuple argv):
    # this is because I'm lazy
    cdef list argv_full = [arg0] + list(argv)
    cdef int argc = <int> len(argv_full)

    # convert all elements of argv and convert to strings, bytes, and store
    cdef list argv_str   = [str(argvi)     for argvi in argv_full]
    cdef list argv_bytes = [argvi.encode() for argvi in argv_str]

    # allocate the argv buffer: remember the extra NULL at the end!
    cdef char** argv_chars = <char**> malloc((argc+1) * sizeof(char*))
    try:
        # args require a null at the end
        argv_chars[argc] = NULL
        # encode the c strings correctly (and store them)
        # and then put them in the array
        for i, argvi in enumerate(argv_bytes):
            argv_chars[i] = argvi # c-string ref extracted

        # Now call our function
        return func(argc, argv_chars) == 0
    
    finally:
        # finally, clean up... whooowh
        free(argv_chars)

####### _miniapp #################################
# C function defined in miniapp-base
# miniapp main -> callable function
cdef extern from "miniapp-base.hpp":
    cdef int _miniapp(int argc, char** argv)
    cdef int _test_miniapp()

######## miniapp ###################################
# miniapp(arg1, ...) in python
#  calls the miniapp main function
#  with a argv[0] of "pyarbor"
def miniapp(*argv):
    return _stringify(_miniapp, "pyarbor", argv)


def test_miniapp():
    return _test_miniapp() == 0
