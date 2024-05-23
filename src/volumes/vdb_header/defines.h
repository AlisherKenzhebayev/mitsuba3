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

#define DRJIT_USE_LLVM
#ifndef DRJIT_USE_LLVM
    #define DRJIT_USE_CUDA
#endif

#if defined(DRJIT_USE_LLVM)
    using BoolJit =     drjit::LLVMArray<bool>;
    using FloatJit =    drjit::LLVMArray<float>;
    using Int32Jit =    drjit::LLVMArray<int32_t>;
    using Int64Jit =    drjit::LLVMArray<int64_t>;
    using UInt32Jit =   drjit::LLVMArray<uint32_t>;
    using UInt64Jit =   drjit::LLVMArray<uint64_t>;
#elif defined(DRJIT_USE_CUDA)
    using BoolJit =     drjit::CUDAArray<bool>;
    using FloatJit =    drjit::CUDAArray<float>;
    using Int32Jit =    drjit::CUDAArray<int32_t>;
    using Int64Jit =    drjit::CUDAArray<int64_t>;
    using UInt32Jit =   drjit::CUDAArray<uint32_t>;
    using UInt64Jit =   drjit::CUDAArray<uint64_t>;
#endif
