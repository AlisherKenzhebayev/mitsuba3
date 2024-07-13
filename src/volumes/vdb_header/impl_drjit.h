#include "defines.h"

#include <drjit/jit.h>
#include <drjit/util.h>

#include <nanovdb/PNanoVDB.h>

#define PNANOVDB_BUF_FORCE_INLINE static inline __attribute__((always_inline))

#pragma region Buffer + Reads

template <typename Float, typename Spectrum>
class DrJitImplementation{
public: 
    MI_IMPORT_CORE_TYPES();

    static constexpr bool IsCUDA = drjit::is_cuda_v<Float>;
    static constexpr bool IsLLVM = drjit::is_llvm_v<Float>;
    static constexpr bool IsJIT = IsLLVM || IsCUDA;

    struct drjit_buf_t
    {
        UInt32 data;
        UInt64 data64;
    #ifdef PNANOVDB_BUF_BOUNDS_CHECK
        UInt64 size_in_words;
    #endif
    };

    PNANOVDB_BUF_FORCE_INLINE drjit_buf_t drjit_make_buf(const void* data, uint64_t size_in_words)
    {
        // static_assert(!IsJIT);

        uint32_t byteSize = size_in_words;
        uint32_t data32Size = byteSize / 4;

        // printf("SIZEaSD %d, %d", byteSize, data32Size);

        drjit_buf_t ret;
        ret.data = drjit::load<UInt32>(data, data32Size);
        if constexpr(IsJIT) 
            ret.data64 = drjit::map<UInt64>(ret.data.data(), data32Size / 2);
        // if constexpr(IsJIT) {
        //     ret.data64 = UInt64::borrow(ret.data.index());
        //     // ret.data64 = UInt64::borrow(ret.data.index());
        // }
        // ret.data64 = drjit::empty<UInt64>(data32Size << 1u);
        // ret.data = data;        // Creates a copy

//         if constexpr(IsJIT) {
//             if constexpr(IsLLVM) {
// // #if defined(DRJIT_USE_LLVM)
//                 jit_memcpy(JitBackend::LLVM, ret.data.data(), data, byteSize); 
//                 // jit_memcpy(JitBackend::LLVM, ret.data64.data(), data, byteSize); 
//             } else {
// // #elif defined(DRJIT_USE_CUDA)
//                 jit_memcpy(JitBackend::CUDA, ret.data.data(), data, byteSize); 
//                 // jit_memcpy(JitBackend::CUDA, ret.data64.data(), data, byteSize); 
//             }
//         }
// #endif

    #ifdef PNANOVDB_BUF_BOUNDS_CHECK
        // ret.size_in_words = size_in_words;
        ret.size_in_words = drjit::full<UInt64>(size_in_words, 1);
    #endif
        return ret;
    }

    PNANOVDB_BUF_FORCE_INLINE UInt32 drjit_buf_read_uint32(drjit_buf_t buf, UInt64 byte_offset, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        //UInt64 wordAddress = (byte_offset >> 2u);
        UInt32 wordAddress = byte_offset >> 2u;// drjit::sr<2u>(byte_offset);
        UInt32 value = drjit::full<UInt32>(0u);
    #ifdef PNANOVDB_BUF_BOUNDS_CHECK
        // if((wordAddress < buf.size_in_words).data()[0]){
        //     return drjit::gather<UInt32>(buf.data, wordAddress);
        // }else{
        //     return value;
        // }
        Bool mask = wordAddress.lt_(buf.size_in_words);
        wordAddress = drjit::select(mask, wordAddress, value); // Jit side optimization to avoid out-of-bounds reads
        UInt32 data = drjit::gather<UInt32>(buf.data, wordAddress);
        UInt32 out = drjit::select(mask, data, value);
        return out;
    #else
        return drjit::gather<UInt32>(dataCopy, wordAddress);
    #endif
    }

    PNANOVDB_BUF_FORCE_INLINE UInt64 drjit_buf_read_uint64(drjit_buf_t buf, UInt64 byte_offset, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        // uint64_t offset64 = byte_offset.data()[0] >> 3u;
        // UInt64 wordAddress64 = drjit::full<UInt64>(offset64, 1);
        UInt64 wordAddress64 = byte_offset >> 3u;// drjit::sr<3u>(byte_offset);
        
        // UInt64 wordAddress = (wordAddress64 << 1u);
        // UInt32 wordAddress32 = wordAddress64 << 1u;// drjit::sl<1u>(wordAddress64);
        // drjit::resize(wordAddress32, 2);
        // UInt32 wordAddress = drjit::repeat(wordAddress32, 2);
        // TODO: confused about the size? 
        // Is it array or matrix?
        // Also the approach I use is to load the gathered data by size
        // Will it still work with the UInt64?
        
        // UInt32 scalarOffset = drjit::linspace<UInt64>(0llu, 1llu, 2);
        // UInt32 addedOffset = drjit::tile(scalarOffset, byte_offset.size());
        // printf("SIZES %lu %lu\n", wordAddress.size(), addedOffset.size());
        // wordAddress = wordAddress + addedOffset;
        
        UInt64 value = drjit::full<UInt64>(uint64_t(0));
    
    #ifdef PNANOVDB_BUF_BOUNDS_CHECK
        // UInt64 sizeInWord64 = buf.size_in_words >> 1u;
        UInt64 sizeInWord64 = buf.size_in_words >> 1u;
        
        Bool mask = wordAddress64 < sizeInWord64;;
        wordAddress64 = drjit::select(mask, wordAddress64, value); // Jit side optimization to avoid out-of-bounds reads
        UInt64 gather64 = drjit::gather<UInt64>(buf.data64, wordAddress64);
        UInt64 out = drjit::select(mask, gather64, value);
        return out;
    #else
        UInt64 gather64 = drjit::gather<UInt64>(buf.data64, wordAddress64);
        
        return gather64;
    #endif
    }
    #pragma endregion Buffer + Reads

    // [TODO] LATER: Line 198 -> Line 1602 For now left as is
    typedef uint32_t drjit_grid_type_t;
    // #define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) drjit_grid_type_constants[grid_typeIn].nameIn

    // [TODO] LATER: Recheck, hereon coding out of PNANOVDB_C assumption 
    #pragma region Basic Types
    #if defined(__CUDACC__)
    #define PNANOVDB_FORCE_INLINE static __host__ __device__ __forceinline__
    #elif defined(_WIN32)
    #define PNANOVDB_FORCE_INLINE static inline __forceinline
    #else
    #define PNANOVDB_FORCE_INLINE static inline __attribute__((always_inline))
    #endif

    #define PNANOVDB_STRUCT_TYPEDEF(X) typedef struct X X;
    #define PNANOVDB_STATIC_CONST static const
    #define PNANOVDB_INOUT(X) X*
    #define PNANOVDB_IN(X) const X*
    #define PNANOVDB_DEREF(X) (*X)
    #define PNANOVDB_REF(X) &X

    // basic types, type conversion
    #define PNANOVDB_NATIVE_64
    #ifndef __CUDACC_RTC__
    #include <stdint.h>
    #endif

    #define PNANOVDB_FALSE 0
    #define PNANOVDB_TRUE 1
    typedef struct drjit_coord_t
    {
        Int32 x, y, z;
    }drjit_coord_t;

    PNANOVDB_FORCE_INLINE Int32 drjit_uint32_as_int32(UInt32 v, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        return (Int32)v; 
    }
    PNANOVDB_FORCE_INLINE Int64 drjit_uint64_as_int64(UInt64 v, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return (Int64)v; 
    }
    PNANOVDB_FORCE_INLINE UInt64 drjit_int64_as_uint64(Int64 v, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return (UInt64)v; 
    }
    PNANOVDB_FORCE_INLINE UInt32 drjit_int32_as_uint32(Int32 v, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return (UInt32)v; 
    }
    PNANOVDB_FORCE_INLINE Float drjit_uint32_as_float(UInt32 v, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        Float vf = drjit::reinterpret_array<Float, UInt32>(v);
        // Float vf = drjit::empty<Float>(1); 
        // vf = drjit::load<Float>(v.data(), v.size());
        return vf;
    }
    PNANOVDB_FORCE_INLINE UInt32 drjit_uint64_low(UInt64 v, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return (UInt32)v; 
    }
    PNANOVDB_FORCE_INLINE UInt64 drjit_uint32_as_uint64_low(UInt32 x, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        return ((UInt64)x); 
    }

    // Mainly used as if case, so Bool should work well
    // [MODIFIED]
    PNANOVDB_FORCE_INLINE Bool drjit_uint64_is_equal(UInt64 a, UInt64 b, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return drjit::eq(a, b); 
    }
    PNANOVDB_FORCE_INLINE Bool drjit_int64_is_zero(Int64 a, Mask active) 
    { 
        MI_MASK_ARGUMENT(active);
        return drjit::eq(a, 0l); 
    }
    #pragma endregion Basic Types

    #pragma region Address Type
    #pragma region x32
        // [TODO] x32 code
    #pragma endregion x32
    #pragma region x64
    struct drjit_address_t
    {
        UInt64 byte_offset;
    };
    PNANOVDB_STRUCT_TYPEDEF(drjit_address_t)

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset(drjit_address_t address, UInt32 byte_offset, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t ret = address;
        ret.byte_offset += byte_offset;
        return ret;
    }
    PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset_neg(drjit_address_t address, UInt32 byte_offset, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t ret = address;
        ret.byte_offset -= byte_offset;
        return ret;
    }
    PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset64(drjit_address_t address, UInt64 byte_offset, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t ret = address;
        ret.byte_offset += byte_offset;
        return ret;
    }
    PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_null(Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t ret = { drjit::zeros<UInt64>(1) };
        return ret;
    }
    PNANOVDB_FORCE_INLINE Bool drjit_address_is_null(drjit_address_t address, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        // Gets broadcasted
        return drjit::eq(address.byte_offset, 0);
    }
    #pragma endregion x64
    #pragma endregion Address Type


    #pragma region HL Buffer Read
    PNANOVDB_FORCE_INLINE UInt32 drjit_read_uint32(drjit_buf_t buf, drjit_address_t address, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        return drjit_buf_read_uint32(buf, address.byte_offset, active);
    }
    PNANOVDB_FORCE_INLINE UInt64 drjit_read_uint64(drjit_buf_t buf, drjit_address_t address, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        return drjit_buf_read_uint64(buf, address.byte_offset, active);
    }
    PNANOVDB_FORCE_INLINE Int64 drjit_read_int64(drjit_buf_t buf, drjit_address_t address, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        return drjit_uint64_as_int64(drjit_read_uint64(buf, address, active), active);
    }
    PNANOVDB_FORCE_INLINE Float drjit_read_float(drjit_buf_t buf, drjit_address_t address, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        return drjit_uint32_as_float(drjit_read_uint32(buf, address, active), active);
    }
    #pragma endregion HL Buffer Read


    #pragma region Core structures
    // [TODO] Skipped, later included with header from pnano 
    #pragma endregion Core structures


    #pragma region Grid + Tree Handle
    struct drjit_grid_handle_t { drjit_address_t address = {drjit::full<UInt64>(uint64_t(0))}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_grid_handle_t)

    struct drjit_tree_handle_t { drjit_address_t address = {drjit::full<UInt64>(uint64_t(0))}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_tree_handle_t)

    PNANOVDB_FORCE_INLINE UInt64 drjit_tree_get_node_offset_root(drjit_buf_t buf, drjit_tree_handle_t p, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        return drjit_read_uint64(buf, drjit_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT, active), active);
    }
    #pragma endregion Grid + Tree Handle


    #pragma region Root Handle
    struct drjit_root_handle_t { drjit_address_t address = {drjit::full<UInt64>(0ul)}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_root_handle_t)

    PNANOVDB_FORCE_INLINE UInt32 drjit_root_get_tile_count(drjit_buf_t buf, drjit_root_handle_t p, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        return drjit_read_uint32(buf, drjit_address_offset(p.address, PNANOVDB_ROOT_OFF_TABLE_SIZE, active), active);
    }
    #pragma endregion Root Handle


    #pragma region Root Tile
    struct drjit_root_tile_handle_t { drjit_address_t address = {drjit::full<UInt64>(0)}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_root_tile_handle_t)

    PNANOVDB_FORCE_INLINE UInt64 drjit_root_tile_get_key(drjit_buf_t buf, drjit_root_tile_handle_t p, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        UInt32 byte_offset = drjit::full<UInt32>(uint32_t(PNANOVDB_ROOT_TILE_OFF_KEY));
        return drjit_read_uint64(buf, drjit_address_offset(p.address, byte_offset, active), active);
    }
    PNANOVDB_FORCE_INLINE Int64 drjit_root_tile_get_child(drjit_buf_t buf, drjit_root_tile_handle_t p, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        UInt32 byte_offset = drjit::full<UInt32>(uint32_t(PNANOVDB_ROOT_TILE_OFF_CHILD));
        return drjit_read_int64(buf, drjit_address_offset(p.address, byte_offset, active), active);
    }
    #pragma endregion Root Tile

    #pragma region Upper Handle
    struct drjit_upper_handle_t { drjit_address_t address = {drjit::full<UInt64>(0)}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_upper_handle_t)

    PNANOVDB_FORCE_INLINE Bool drjit_upper_get_child_mask(drjit_buf_t buf, drjit_upper_handle_t p, UInt32 bit_index, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        UInt32 bit_index_shift = bit_index >> 5u;// drjit::sr<5u>(bit_index);
        UInt32 upper_off_child = drjit::full<UInt32>(uint32_t(PNANOVDB_UPPER_OFF_CHILD_MASK));
        UInt32 byte_offset = drjit::fmadd(4u, bit_index_shift, upper_off_child);
        
        // byte_offset = PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u);
        drjit_address_t tempAddr = drjit_address_offset(p.address, byte_offset, active);
        UInt32 value = drjit_read_uint32(buf, tempAddr, active);
        UInt32 and_bit_index = bit_index & drjit::full<UInt32>(31u);
        UInt32 shifted_value = value >> and_bit_index; // Unable to call the drjit::sr<> for the and_bit_index
        // return ((value >> (bit_index & 31u)) & 1) != 0u;
        // return (shifted_value & drjit::full<UInt32>(1u)) != drjit::zeros<UInt32>();
        return drjit::neq<UInt32, UInt32>
            (
                (shifted_value & drjit::full<UInt32>(1)),
                drjit::zeros<UInt32>()
            );
    }
    #pragma endregion Upper Handle


    #pragma region Lower Handle
    struct drjit_lower_handle_t { drjit_address_t address = {drjit::full<UInt64>(0)}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_lower_handle_t)


    PNANOVDB_FORCE_INLINE Bool drjit_lower_get_child_mask(drjit_buf_t buf, drjit_lower_handle_t p, UInt32 bit_index, Mask active) 
    {
        MI_MASK_ARGUMENT(active);
        UInt32 bit_index_shift = bit_index >> 5u;// drjit::sr<5u>(bit_index);
        // UInt32 bit_index_shift = drjit::sr<5u>(bit_index);
        UInt32 lower_off_child = drjit::full<UInt32>(uint32_t(PNANOVDB_LOWER_OFF_CHILD_MASK));
        UInt32 byte_offset = drjit::fmadd(4u, bit_index_shift, lower_off_child);
        
        // byte_offset = PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u);

        UInt32 value = drjit_read_uint32(buf, drjit_address_offset(p.address, byte_offset, active), active);
        UInt32 and_bit_index = bit_index & drjit::full<UInt32>(31u);
        UInt32 shifted_value = value >> and_bit_index; // Unable to call the drjit::sr<> for the and_bit_index
        // return ((value >> (bit_index & 31u)) & 1) != 0u;    
        // return (shifted_value & drjit::full<UInt32>(uint32_t(1))) != drjit::zeros<UInt32>();
        return drjit::neq<UInt32, UInt32>
            (
                (shifted_value & drjit::full<UInt32>(1)),
                drjit::zeros<UInt32>()
            );
    }
    #pragma endregion Lower Handle


    #pragma region Leaf
    struct drjit_leaf_handle_t { drjit_address_t address = {drjit::full<UInt64>(0)}; };
    PNANOVDB_STRUCT_TYPEDEF(drjit_leaf_handle_t)
    #pragma endregion Leaf


    #pragma region Get Handle (Tree, Root)
    PNANOVDB_FORCE_INLINE drjit_tree_handle_t drjit_grid_get_tree(drjit_buf_t buf, drjit_grid_handle_t grid, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_tree_handle_t tree = { grid.address };
        tree.address = drjit_address_offset(grid.address, drjit::full<UInt32>(uint32_t(PNANOVDB_GRID_SIZE)), active);
        return tree;
    }

    PNANOVDB_FORCE_INLINE drjit_root_handle_t drjit_tree_get_root(drjit_buf_t buf, drjit_tree_handle_t tree, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_root_handle_t root = { tree.address };
        UInt64 byte_offset = drjit_tree_get_node_offset_root(buf, tree, active);
        root.address = drjit_address_offset64(root.address, byte_offset, active);
        return root;
    }

    PNANOVDB_FORCE_INLINE drjit_root_tile_handle_t drjit_root_get_tile_zero(drjit_grid_type_t grid_type, drjit_root_handle_t root, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_root_tile_handle_t tile = { root.address };
        tile.address = drjit_address_offset(tile.address, drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, root_size)), active);
        return tile;
    }

    PNANOVDB_FORCE_INLINE drjit_upper_handle_t drjit_root_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, drjit_root_tile_handle_t tile, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_upper_handle_t upper = { root.address };
        upper.address = drjit_address_offset64(upper.address, drjit_int64_as_uint64(drjit_root_tile_get_child(buf, tile, active), active), active);
        return upper;
    }
    #pragma endregion Get Handle (Tree, Root)


    #pragma region Coord To Key
    PNANOVDB_FORCE_INLINE UInt64 drjit_coord_to_key(PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
    #if defined(PNANOVDB_NATIVE_64)
        UInt64 iu = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).x, active) >> 12u;
        UInt64 ju = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).y, active) >> 12u;
        UInt64 ku = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).z, active) >> 12u;
        return ku.or_(ju << 21u).or_(iu << 42u);
    #else
    // TODO: x32
        // pnanovdb_uint32_t iu = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x) >> 12u;
        // pnanovdb_uint32_t ju = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).y) >> 12u;
        // pnanovdb_uint32_t ku = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).z) >> 12u;
        // pnanovdb_uint32_t key_x = ku | (ju << 21);
        // pnanovdb_uint32_t key_y = (iu << 10) | (ju >> 11);
        // return pnanovdb_uint32_as_uint64(key_x, key_y);
    #endif
    }
    #pragma endregion Coord To Key


    #pragma region Root Find Tile
    PNANOVDB_FORCE_INLINE drjit_root_tile_handle_t drjit_root_find_tile(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_buf_t buffer = buf;

        UInt32 tileCount = drjit_uint32_as_int32(drjit_root_get_tile_count(buf, root, active), active);    
        UInt32 tile_count = tileCount;

        drjit_root_tile_handle_t tile = drjit_root_get_tile_zero(grid_type, root, active);
        
        UInt32 tileFixed = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size));
        UInt32 tileFixedOffset = tileFixed;
        
        UInt64 coordKey = drjit_coord_to_key(ijk, active);
        UInt64 key = coordKey;
        
        uint32_t size = coordKey.size();
        UInt64 tileAddressOffset = tile.address.byte_offset;

        UInt32 byte_offset = drjit::full<UInt32>(uint32_t(PNANOVDB_ROOT_TILE_OFF_KEY));
        
        UInt32 i = drjit::full<UInt32>(0);
        Bool foundMask = drjit::full<Bool>(false);
        Bool curMask = drjit::full<Bool>(false);
        Bool modifyMask  = drjit::full<Bool>(false);
        drjit::Loop<Bool> 
            loop("Root Find Tile", i, key, tile_count, curMask, foundMask, modifyMask, tileFixedOffset, tileAddressOffset, byte_offset, buffer);

        while(loop(i < tile_count)){
            drjit_root_tile_handle_t dummyTileHandleInLoop = 
                {drjit_address_t{drjit::fmadd(i, tileFixedOffset, tileAddressOffset)}};
            
            UInt64 currentKey = drjit_root_tile_get_key(buffer, dummyTileHandleInLoop, active);
            
            curMask = drjit_uint64_is_equal(key, currentKey, active);
            Bool pos = foundMask.and_(curMask);
            Bool neg = foundMask.not_().and_(curMask);
            modifyMask = pos.or_(neg);
            foundMask = foundMask.or_(curMask);
            
            tileAddressOffset = drjit::select(
                modifyMask, 
                drjit::fmadd(i, tileFixedOffset, tileAddressOffset), 
                tileAddressOffset);

            i += 1;
        }
        
        drjit_root_tile_handle_t null_handle = { drjit_address_null(active) };
        tile.address.byte_offset = drjit::select(foundMask, tileAddressOffset, null_handle.address.byte_offset);
        return tile;
    }
    #pragma endregion Root Find Tile


    #pragma region Leaf Node
    PNANOVDB_FORCE_INLINE UInt32 drjit_leaf_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        Int32 constN = drjit::full<Int32>(7);
        drjit_coord_t coord = PNANOVDB_DEREF(ijk);

        UInt32 x = (constN.and_(coord.x)) >> 0 << 6;
        UInt32 y = (constN.and_(coord.y)) >> 0 << 3;
        UInt32 z = (constN.and_(coord.z)) >> 0;

        return x + y + z;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_leaf_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_leaf_handle_t node, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 byte_offset = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_table));
        UInt32 strideBits = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits)) * n;
        strideBits = strideBits >> 3u;
        byte_offset += strideBits;
        return drjit_address_offset(node.address, byte_offset, active);
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_leaf_get_value_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_leaf_handle_t leaf, PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 n = drjit_leaf_coord_to_offset(ijk, active);
        return drjit_leaf_get_table_address(grid_type, buf, leaf, n, active);
    }
    #pragma endregion Leaf Node


    #pragma region Lower Node
    PNANOVDB_FORCE_INLINE UInt32 drjit_lower_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        Int32 constN = drjit::full<Int32>(127);
        drjit_coord_t coord = PNANOVDB_DEREF(ijk);
        
        UInt32 x = (constN.and_(coord.x)) >> 3 << 8;
        UInt32 y = (constN.and_(coord.y)) >> 3 << 4;
        UInt32 z = (constN.and_(coord.z)) >> 3;
        
        return x + y + z;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_lower_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t node, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 byte_offset = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_table));
        UInt32 strideBits = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, table_stride)) * n;
        byte_offset += strideBits;
        return drjit_address_offset(node.address, byte_offset, active);
    }

    PNANOVDB_FORCE_INLINE Int64 drjit_lower_get_table_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t node, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t table_address = drjit_lower_get_table_address(grid_type, buf, node, n, active);
        return drjit_read_int64(buf, table_address, active);
    }

    PNANOVDB_FORCE_INLINE drjit_leaf_handle_t drjit_lower_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t lower, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_leaf_handle_t leaf = { lower.address };
        leaf.address = drjit_address_offset64(leaf.address, drjit_int64_as_uint64(drjit_lower_get_table_child(grid_type, buf, lower, n, active), active), active);
        return leaf;
    }
    #pragma endregion Lower Node

    #pragma region Last8
    PNANOVDB_FORCE_INLINE drjit_address_t drjit_lower_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t lower, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(UInt32) level, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 n = drjit_lower_coord_to_offset(ijk, active);
        drjit_address_t value_address;
        drjit_address_t leaf_address;
        drjit_address_t lower_address;
        Bool mask = drjit_lower_get_child_mask(buf, lower, n, active);

        drjit_leaf_handle_t child = drjit_lower_get_child(grid_type, buf, lower, n, active);
        leaf_address = drjit_leaf_get_value_address(grid_type, buf, child, ijk, active);
        lower_address = drjit_lower_get_table_address(grid_type, buf, lower, n, active);

        value_address.byte_offset = drjit::select(mask, leaf_address.byte_offset, lower_address.byte_offset);
        PNANOVDB_DEREF(level) = drjit::select(mask, 0u, 1u);

        return value_address;
    }

    PNANOVDB_FORCE_INLINE UInt32 drjit_upper_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        Int32 constN = drjit::full<Int32>(4095);
        drjit_coord_t coord = PNANOVDB_DEREF(ijk);
        
        UInt32 x = (constN.and_(coord.x)) >> 7 << 10;
        UInt32 y = (constN.and_(coord.y)) >> 7 << 5;
        UInt32 z = (constN.and_(coord.z)) >> 7;
        
        return x + y + z;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_upper_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t node, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 byte_offset = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_table));
        UInt32 strideBits = drjit::full<UInt32>(PNANOVDB_GRID_TYPE_GET(grid_type, table_stride)) * n;
        byte_offset += strideBits;
        return drjit_address_offset(node.address, byte_offset, active);
    }

    PNANOVDB_FORCE_INLINE Int64 drjit_upper_get_table_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t node, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_address_t bufAddress = drjit_upper_get_table_address(grid_type, buf, node, n, active);
        return drjit_read_int64(buf, bufAddress, active);
    }

    PNANOVDB_FORCE_INLINE drjit_lower_handle_t drjit_upper_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t upper, UInt32 n, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        drjit_lower_handle_t lower = { upper.address };
        lower.address = drjit_address_offset64(lower.address, drjit_int64_as_uint64(drjit_upper_get_table_child(grid_type, buf, upper, n, active), active), active);
        return lower;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_upper_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t upper, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(UInt32) level, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 n = drjit_upper_coord_to_offset(ijk, active);
        drjit_address_t value_address;
        drjit_address_t lower_address;
        UInt32 lower_level;
        drjit_address_t upper_address;
        Bool mask = drjit_upper_get_child_mask(buf, upper, n, active);
        
        drjit_lower_handle_t child = drjit_upper_get_child(grid_type, buf, upper, n, active);
        lower_address = drjit_lower_get_value_address_and_level(grid_type, buf, child, ijk, &lower_level, active);
        upper_address = drjit_upper_get_table_address(grid_type, buf, upper, n, active);
        
        value_address.byte_offset = drjit::select(mask, lower_address.byte_offset, upper_address.byte_offset);
        PNANOVDB_DEREF(level) = drjit::select(mask, lower_level, 2u);

        return value_address;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_root_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(UInt32) level, Mask active)
    {
        MI_MASK_ARGUMENT(active);
    // default = FF, rewrite1 = T*, rewrite2 = FT
        drjit_root_tile_handle_t tile = drjit_root_find_tile(grid_type, buf, root, ijk, active);
        drjit_address_t ret;
        
        drjit_upper_handle_t child = drjit_root_get_child(grid_type, buf, root, tile, active);
        
        UInt32 dflt_level;
        drjit_address_t dflt = drjit_upper_get_value_address_and_level(grid_type, buf, child, ijk, &dflt_level, active);
        drjit_address_t valT = drjit_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background), active);
        drjit_address_t valFT = drjit_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value), active);

        Bool mask1 = drjit_address_is_null(tile.address, active);
        Bool mask2 = drjit_int64_is_zero(drjit_root_tile_get_child(buf, tile, active), active);
        
        ret.byte_offset = drjit::select(mask1, valT.byte_offset, dflt.byte_offset);
        ret.byte_offset = drjit::select(mask2, valFT.byte_offset, ret.byte_offset);
        PNANOVDB_DEREF(level) = drjit::select(mask1, 4u, dflt_level);
        PNANOVDB_DEREF(level) = drjit::select(mask2, 3u, PNANOVDB_DEREF(level));

        return ret;
    }

    PNANOVDB_FORCE_INLINE drjit_address_t drjit_root_get_value_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk, Mask active)
    {
        MI_MASK_ARGUMENT(active);
        UInt32 level;
        return drjit_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level), active);
    }
    #pragma endregion Last8

    // jit_var_lt
    // uint32_t v0 = jit_var_literal(/* backend  = */ JitBackendCUDA,
    //                               /* type     = */ VarTypeFloat32,
    //                               /* value    = */ &value,
    //                               /* size     = */ 1,
    //                               /* eval     = */ 0,
    //                               /* is_class = */ 0);
    // );

};