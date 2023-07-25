#include "common.cuh"
#include "logging.h"
#include <cuda_runtime.h>
#include <cuda.h>

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        __logCritical("CUDA ASSERT FAILURE :: %s %s %d", cudaGetErrorString(code), file, line);

        if (abort)
            exit(code);
    }
}

void gpuErrorCheck(const char *file, int line)
{
    // --- Errors
    auto errSync = cudaGetLastError();
    auto errAsync = cudaDeviceSynchronize();

    // --- Status
    bool success = true;
    std::stringstream ss;

    // --- Error Check : synchronous
    if (errSync != cudaSuccess)
    {
        success = false;
        __logCritical("Synchronous Kernel Error :: %s", cudaGetErrorString(errSync));
        ss << "CUDA Error :: " << cudaGetErrorString(errSync);
    }

    // --- Error Check : asynchronous
    if (errAsync != cudaSuccess)
    {
        success = false;
        __logCritical("Synchronous Kernel Error :: %s", cudaGetErrorString(errAsync));
        ss << "CUDA Error Async :: " << cudaGetErrorString(errAsync);
    }

    if(not success)
        throw std::runtime_error(ss.str());
}

void* strict_malloc(size_t mem_size, const char* filename, int line)
{
    void* host_ptr = malloc(mem_size);

    // --- Error Checking
    if (host_ptr == nullptr) {
        __logCritical("MALLOC FAILURE :: Allocating %zu bytes of memory", mem_size);
        exit(EXIT_FAILURE);
    }

    return host_ptr;
}
