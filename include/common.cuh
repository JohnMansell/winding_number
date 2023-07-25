//
// Created by john on 7/24/23.
//

#ifndef COMMON_CUH
#define COMMON_CUH

// --- C
#include <complex>
#include <cstdlib>


// --- CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>


// --- OS
#ifdef _WIN32
#include <windows.h>
#else // linux
#include <unistd.h>
#endif // _WIN32


// ------------------------------------
//          Macros
// ------------------------------------
#ifdef __DEBUG__
    #define __gpuAssert(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
    #define __gpuErrCheck() {gpuErrorCheck(__FILE__, __LINE__); }
    #define __gpuMalloc(count) { gpuMalloc(count, __FILE__, __LINE__) }
    #define __gpuMallocManaged(count) { gpuMallocManaged(count, __FILE__, __LINE__) }
    #define __hostMalloc(count) { hostMalloc(count, __FILE__, __LINE__) }
    #define __gpuMemcpy(dest, source, count) { __gpuAssert(gpuMemcpy(dest, source, count)) }
#else
    #define __gpuAssert(ans) (ans);
    #define __gpuErrCheck() {}
    #define __gpuMalloc(count) { gpuMalloc(count)}
    #define __gpuMallocManaged(count) {gpuMallocManaged(count)}
    #define __hostMalloc(count) hostMalloc(count)
    #define __gpuMemcpy(dest, source, count) gpuMemcpy(dest, source, count)
#endif

// ------------------------------------
//          Host Functions
// ------------------------------------
void sleep_ms(int milliseconds);
void* strict_malloc(size_t size, const char *filename, int line);

// ------------------------------------
//          GPU Functions
// ------------------------------------
void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
void gpuErrorCheck(const char *file, int line);

template <class T>
cudaError_t gpuMemcpy(T *dest, T *source, size_t count)
{
    size_t mem_size = sizeof(T) * count;
    return cudaMemcpy(dest, source, mem_size, cudaMemcpyDefault);
}

template <class T>
auto cpu_malloc_t(size_t count, const char *filename, int line)
{
    size_t mem_size = sizeof(T) * count;
    return (T*) strict_malloc(mem_size, filename, line);
}

struct hostMalloc
{
    // --- Members
    size_t count = 1;
    int line = 0;
    const char * filename;

    // --- Copy Constructor
    template <class T>
    operator T() {
        using pointer_type = typename std::remove_pointer<T>::type;
        return cpu_malloc_t<pointer_type>(count, filename, line);
    }

    // --- Constructors
    explicit hostMalloc(size_t _count)
            : count(_count) {};

    // --- Debug Constructor
    explicit hostMalloc(size_t _count, const char* _filename, int _line)
            : count(_count),  filename(_filename), line(_line) {};
};

template <class T>
auto gpu_malloc_t(size_t count, const char* filename, int line)
{
    T* d_ptr;
    //        #ifdef __DEBUG__
    //                gpuAssert(cudaMalloc((void**) &d_ptr, count * sizeof(T)), filename, line, false);
    //        #else
    //                cudaMalloc((void**) &d_ptr, count * sizeof(T));
    //        #endif

    cudaMalloc((void**) &d_ptr, count * sizeof(T));

    return d_ptr;
}

struct gpuMalloc
{
    // --- Member
    size_t count = 1;
    int line = 0;
    const char * filename;

    // --- Copy constructor (pretty much the only place C++ will let you infer type from return type)
    template<class T>
    operator T(){
        using pointer_type = typename std::remove_pointer<T>::type;
        return gpu_malloc_t<pointer_type>(count, filename, line);
    }

    // --- Constructor
    explicit gpuMalloc(size_t _count)
            : count(_count) {};

    // --- Debug Constructor
    explicit gpuMalloc(size_t _count, const char* _filename, int _line)
            : count(_count),  filename(_filename), line(_line) {};
};

template <class T>
auto gpu_malloc_managed_t(size_t count, const char* filename, int line)
{
    T* managed_ptr;
    cudaMallocManaged((void**) &managed_ptr, count * sizeof(T));

    return managed_ptr;
}

struct gpuMallocManaged
{
    // --- Members
    size_t count = 1;
    int line = 0;
    const char * filename;

    // --- Copy Constructor  : infer type T
    template <class T>
    operator T() {
        using pointer_type = typename std::remove_pointer<T>::type;
        return gpu_malloc_managed_t<pointer_type>(count, filename, line);
    }

    // --- Constructor
    explicit gpuMallocManaged(size_t _count)
            : count(_count) {};

    // --- Debug Constructor
    explicit gpuMallocManaged(size_t _count, const char* _filename, int _line)
            : count(_count), filename(_filename), line(_line) {};
};


#endif // COMMON_CUH

