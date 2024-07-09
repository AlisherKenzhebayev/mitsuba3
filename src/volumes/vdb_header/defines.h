#pragma once
#include <drjit/jit.h>

#define EPS 1e-6

#define PNANOVDB_BUF_BOUNDS_CHECK

// Constants
#define OPENVDB_FILE "vdbfiles/bunny_cloud.vdb"
#define OPENVDB_GRID "density"
#define NANOVDB_FILE "vdbfiles/converted.nvdb"

// ------------------------------------------------ Configuration ----------------------------------------------------------- LINE 17

// platforms
#define PNANOVDB_C
//#define PNANOVDB_HLSL
//#define PNANOVDB_GLSL

// addressing mode
// PNANOVDB_ADDRESS_32
// PNANOVDB_ADDRESS_64
#if defined(PNANOVDB_C)
#ifndef PNANOVDB_ADDRESS_32
#define PNANOVDB_ADDRESS_64
#endif
// #elif defined(PNANOVDB_HLSL)
// #ifndef PNANOVDB_ADDRESS_64
// #define PNANOVDB_ADDRESS_32
// #endif
// #elif defined(PNANOVDB_GLSL)
// #ifndef PNANOVDB_ADDRESS_64
// #define PNANOVDB_ADDRESS_32
// #endif
#endif

// -- DRJIT --

#define DRJIT_USE_AD
#define DRJIT_USE_LLVM
#ifndef DRJIT_USE_LLVM
    #define DRJIT_USE_CUDA
#endif

#if defined(DRJIT_USE_AD)
#if defined(DRJIT_USE_LLVM)
    using Bool =     drjit::DiffArray<drjit::LLVMArray<bool>>;
    using Float =    drjit::DiffArray<drjit::LLVMArray<float>>;
    using Int32 =    drjit::DiffArray<drjit::LLVMArray<int32_t>>;
    using Int64 =    drjit::DiffArray<drjit::LLVMArray<int64_t>>;
    using UInt32 =   drjit::DiffArray<drjit::LLVMArray<uint32_t>>;
    using UInt64 =   drjit::DiffArray<drjit::LLVMArray<uint64_t>>;
#elif defined(DRJIT_USE_CUDA)
    using Bool =     drjit::DiffArray<JitBackend::CUDA, bool>;
    using Float =    drjit::DiffArray<JitBackend::CUDA, float>;
    using Int32 =    drjit::DiffArray<JitBackend::CUDA, int32_t>;
    using Int64 =    drjit::DiffArray<JitBackend::CUDA, int64_t>;
    using UInt32 =   drjit::DiffArray<JitBackend::CUDA, uint32_t>;
    using UInt64 =   drjit::DiffArray<JitBackend::CUDA, uint64_t>;
#endif
#else
#if defined(DRJIT_USE_LLVM)
    using Bool =     drjit::LLVMArray<bool>;
    using Float =    drjit::LLVMArray<float>;
    using Int32 =    drjit::LLVMArray<int32_t>;
    using Int64 =    drjit::LLVMArray<int64_t>;
    using UInt32 =   drjit::LLVMArray<uint32_t>;
    using UInt64 =   drjit::LLVMArray<uint64_t>;
#elif defined(DRJIT_USE_CUDA)
    using Bool =     drjit::CUDAArray<bool>;
    using Float =    drjit::CUDAArray<float>;
    using Int32 =    drjit::CUDAArray<int32_t>;
    using Int64 =    drjit::CUDAArray<int64_t>;
    using UInt32 =   drjit::CUDAArray<uint32_t>;
    using UInt64 =   drjit::CUDAArray<uint64_t>;
#endif
#endif