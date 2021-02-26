#pragma once

// #define MARK_REGIONS 1
#ifdef MARK_REGIONS
#define ENTER(stream) (stream) << " /* " << __FUNCTION__ << ":enter*/ "
#define EXIT(stream)  (stream) << " /* " << __FUNCTION__ << ":exit*/ "
#define ENTERM(stream, m) (stream) << " /* " << __FUNCTION__ << ":" << m << ":enter */ "
#define EXITM(stream, m)  (stream) << " /* " << __FUNCTION__ << ":" << m << ":exit */ "
#else
#define ENTER(stream)
#define EXIT(stream)
#define ENTERM(stream, m)
#define EXITM(stream, m)
#endif
