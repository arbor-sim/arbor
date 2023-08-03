// Used to extract SVE vector register size (in bits)
#include <iostream>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
inline int sve_length() noexcept { std::cout << svcntb()*8; return 1; }
#else
inline int sve_length() noexcept { std::cout << 16*8; return 0; }
#endif

int main() { return sve_length(); }
