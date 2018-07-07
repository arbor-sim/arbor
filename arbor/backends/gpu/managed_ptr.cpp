namespace arb {
namespace gpu {

// TODO: make this a runtime check

bool device_concurrent_managed_access() {
    return (ARB_CUDA_ARCH >= 600); // all GPUs from P100
}

} // namespace gpu
} // namespace arb

