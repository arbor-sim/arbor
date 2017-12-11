from libc.stdlib cimport malloc, free

cdef extern from "miniapp-base.hpp":
    cdef int miniapp(int argc, char** argv)

def pyarbor_miniapp(argv):
    cdef int argc = <int> len(argv)
    cdef char* arg0 = "pyarbor\0"
    cdef char** argv_chars = <char**> malloc((argc+1) * sizeof(char*))
    cdef char** argv_chars_off = argv_chars + 1
    try:
        argv_chars[0] = arg0
        argv_chars[argc] = NULL

        argv_bytes = [argvi.encode() for argvi in argv[1:]]
        for i, argvi in enumerate(argv_bytes):
            argv_chars_off[i] = argvi # c-string ref extracted

        return miniapp(argc, argv_chars) == 0
    finally: free(argv_chars)
