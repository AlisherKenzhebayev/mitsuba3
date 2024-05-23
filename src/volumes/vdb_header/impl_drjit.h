#include "defines.h"

#include <drjit/jit.h>
#include <drjit/util.h>

#include <nanovdb/PNanoVDB.h>

#define PNANOVDB_BUF_FORCE_INLINE static inline __attribute__((always_inline))

#pragma region Buffer + Reads
typedef struct drjit_buf_t
{
    UInt32Jit data;
    UInt64Jit data64;
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    UInt64Jit size_in_words;
#endif
}drjit_buf_t;

PNANOVDB_BUF_FORCE_INLINE drjit_buf_t drjit_make_buf(uint32_t* data, uint64_t size_in_words)
{
    uint32_t byteSize = size_in_words;
    uint32_t data32Size = byteSize / 4;

    drjit_buf_t ret;
    ret.data = drjit::empty<UInt32Jit>(data32Size);
    ret.data64 = drjit::empty<UInt64Jit>(data32Size >> 1u);
    // ret.data = data;        // Creates a copy

#if defined(DRJIT_USE_LLVM)
    jit_memcpy(JitBackend::LLVM, ret.data.data(), data, byteSize); 
    jit_memcpy(JitBackend::LLVM, ret.data64.data(), data, byteSize >> 1u); 
#elif defined(DRJIT_USE_CUDA)
    jit_memcpy(JitBackend::CUDA, ret.data.data(), data, byteSize); 
    jit_memcpy(JitBackend::CUDA, ret.data64.data(), data, byteSize >> 1u); 
#endif

#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    // ret.size_in_words = size_in_words;
    ret.size_in_words = drjit::full<UInt64Jit>(size_in_words, 1);
#endif
    return ret;
}

PNANOVDB_BUF_FORCE_INLINE UInt32Jit drjit_buf_read_uint32(drjit_buf_t buf, UInt64Jit byte_offset)
{
    //UInt64Jit wordAddress = (byte_offset >> 2u);
    UInt64Jit wordAddress = byte_offset.sr_(2u);// drjit::sr<2u>(byte_offset);
    UInt32Jit value = drjit::full<UInt32Jit>(0u, byte_offset.size());
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    // if((wordAddress < buf.size_in_words).data()[0]){
    //     return drjit::gather<UInt32Jit>(buf.data, wordAddress);
    // }else{
    //     return value;
    // }
    BoolJit mask = wordAddress.lt_(buf.size_in_words);
    UInt32Jit data = drjit::gather<UInt32Jit>(buf.data, wordAddress);
    UInt32Jit out = drjit::select(mask, data, value);
    return out;
#else
    return drjit::gather<UInt32Jit>(dataCopy, wordAddress);
#endif
}

PNANOVDB_BUF_FORCE_INLINE UInt64Jit drjit_buf_read_uint64(drjit_buf_t buf, UInt64Jit byte_offset)
{
    // uint64_t offset64 = byte_offset.data()[0] >> 3u;
    // UInt64Jit wordAddress64 = drjit::full<UInt64Jit>(offset64, 1);
    UInt64Jit wordAddress64 = byte_offset.sr_(3u);// drjit::sr<3u>(byte_offset);
    
    // UInt64Jit wordAddress = (wordAddress64 << 1u);
    UInt64Jit wordAddress32 = wordAddress64.sl_(1u);// drjit::sl<1u>(wordAddress64);
    // drjit::resize(wordAddress32, 2);
    UInt64Jit wordAddress = drjit::repeat(wordAddress32, 2);
    // TODO: confused about the size? 
    // Is it array or matrix?
    // Also the approach I use is to load the gathered data by size
    // Will it still work with the UInt64?
    
    UInt64Jit scalarOffset = drjit::linspace<UInt64Jit>(0llu, 1llu, 2);
    UInt64Jit addedOffset = drjit::tile(scalarOffset, byte_offset.size());
    printf("SIZES %lu %lu\n", wordAddress.size(), addedOffset.size());
    wordAddress = wordAddress + addedOffset;

    UInt64Jit value = drjit::full<UInt64Jit>(0llu, byte_offset.size());
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    // UInt64Jit sizeInWord64 = buf.size_in_words >> 1u;
    UInt64Jit sizeInWord64 = buf.size_in_words.sr_(1u);// drjit::sr<1u>(buf.size_in_words);
    
    BoolJit mask = wordAddress64.lt_(sizeInWord64);
    UInt32Jit gather32 = drjit::gather<UInt32Jit>(buf.data, wordAddress);
    // UInt64Jit gather64 = drjit::load<UInt64Jit>(gather32.data(), byte_offset.size());
    UInt64Jit gather64 = drjit::gather<UInt64Jit>(buf.data64, wordAddress64);
    UInt64Jit out = drjit::select(mask, gather64, value);
    return out; 
    //     // BEST IDEA -
    //     // 1. offset all the doubled indices idx, idx, idx2, idx2 by 1 for every second
    //     // 2. Load the gathered data into half sized uint64 format, in order as they come
    //     // Q: how to offset every second? Mask???
    
    // if((wordAddress64 < sizeInWord64).data()[0]){
    //     UInt32Jit gather = drjit::gather<UInt32Jit>(buf.data, wordAddress);
    //     // UInt64Jit gather64 = drjit::gather<UInt64Jit>(buf.data, wordAddress);
    //     // UInt64Jit gather64x8 = drjit::gather<UInt64Jit>(buf.data, wordAddress64);
    //     value = drjit::load<UInt64Jit>(gather.data(), byte_offset.size());
    //     printf("GATHER     - %u | %u\n", gather.data()[0], gather.data()[1]);
    //     printf("DATA VALUE - %lu\n", value.data()[0]);
    //     // UInt64Jit conv2Copy = drjit::empty<UInt64Jit>(1);
    //     // conv2Copy = UInt64Jit(gather.data());
    //     // printf("DATA CONV - %lu\n", conv2Copy.data()[0]);
    //     return value;
    // }else{
    //     return value;
    // }
#else
    UInt32Jit gather32 = drjit::gather<UInt32Jit>(buf.data, wordAddress);
    UInt64Jit gather64 = drjit::load<UInt64Jit>(gather32.data(), byte_offset.size());
    
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

// [TODO] remove this double abstraction of types, as simply repeats types defined in header
typedef UInt32Jit drjit_uint32_t;
typedef Int32Jit drjit_int32_t;
typedef BoolJit drjit_bool_t;
typedef FloatJit drjit_float_t;
#define PNANOVDB_FALSE 0
#define PNANOVDB_TRUE 1
typedef UInt64Jit drjit_uint64_t;
typedef Int64Jit drjit_int64_t;
typedef struct drjit_coord_t
{
    drjit_int32_t x, y, z;
}drjit_coord_t;

PNANOVDB_FORCE_INLINE drjit_int32_t drjit_uint32_as_int32(drjit_uint32_t v) { return (drjit_int32_t)v; }
PNANOVDB_FORCE_INLINE drjit_int64_t drjit_uint64_as_int64(drjit_uint64_t v) { return (drjit_int64_t)v; }
PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_int64_as_uint64(drjit_int64_t v) { return (drjit_uint64_t)v; }
PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_int32_as_uint32(drjit_int32_t v) { return (drjit_uint32_t)v; }
PNANOVDB_FORCE_INLINE drjit_float_t drjit_uint32_as_float(drjit_uint32_t v) 
{ 
    drjit_float_t vf = drjit::empty<drjit_float_t>(1); 
    // TODO - here
    vf = drjit::load<drjit_float_t>(v.data(), v.size());
    return vf;
}
PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_uint64_low(drjit_uint64_t v) { return (drjit_uint32_t)v; }
PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_uint32_as_uint64_low(drjit_uint32_t x) { return ((drjit_uint64_t)x); }

// Mainly used as if case, so BoolJit should work well
// [MODIFIED]
PNANOVDB_FORCE_INLINE drjit_bool_t drjit_uint64_is_equal(drjit_uint64_t a, drjit_uint64_t b) { return drjit::eq(a, b); }
PNANOVDB_FORCE_INLINE drjit_bool_t drjit_int64_is_zero(drjit_int64_t a) { return drjit::eq(a, 0); }
#pragma endregion Basic Types

#pragma region Address Type
#pragma region x32
    // [TODO] x32 code
#pragma endregion x32
#pragma region x64
struct drjit_address_t
{
    drjit_uint64_t byte_offset;
};
PNANOVDB_STRUCT_TYPEDEF(drjit_address_t)

PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset(drjit_address_t address, drjit_uint32_t byte_offset)
{
    drjit_address_t ret = address;
    ret.byte_offset += byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset_neg(drjit_address_t address, drjit_uint32_t byte_offset)
{
    drjit_address_t ret = address;
    ret.byte_offset -= byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_offset64(drjit_address_t address, drjit_uint64_t byte_offset)
{
    drjit_address_t ret = address;
    ret.byte_offset += byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE drjit_address_t drjit_address_null()
{
    // [TODO] need to check for broadcasting
    drjit_address_t ret = { drjit::zeros<UInt64Jit>(1) };
    return ret;
}
PNANOVDB_FORCE_INLINE drjit_bool_t drjit_address_is_null(drjit_address_t address)
{
    // Gets broadcasted
    return drjit::eq(address.byte_offset, 0);
}
#pragma endregion x64
#pragma endregion Address Type


#pragma region HL Buffer Read
PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_read_uint32(drjit_buf_t buf, drjit_address_t address)
{
    return drjit_buf_read_uint32(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_read_uint64(drjit_buf_t buf, drjit_address_t address)
{
    return drjit_buf_read_uint64(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE drjit_int64_t drjit_read_int64(drjit_buf_t buf, drjit_address_t address)
{
    return drjit_uint64_as_int64(drjit_read_uint64(buf, address));
}
PNANOVDB_FORCE_INLINE drjit_float_t drjit_read_float(drjit_buf_t buf, drjit_address_t address)
{
    return drjit_uint32_as_float(drjit_read_uint32(buf, address));
}
#pragma endregion HL Buffer Read


#pragma region Core structures
// [TODO] Skipped, later included with header from pnano 
#pragma endregion Core structures


#pragma region Grid + Tree Handle
struct drjit_grid_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_grid_handle_t)

struct drjit_tree_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_tree_handle_t)

PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_tree_get_node_offset_root(drjit_buf_t buf, drjit_tree_handle_t p) {
    return drjit_read_uint64(buf, drjit_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT));
}
#pragma endregion Grid + Tree Handle


#pragma region Root Handle
struct drjit_root_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_root_handle_t)

PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_root_get_tile_count(drjit_buf_t buf, drjit_root_handle_t p) {
    return drjit_read_uint32(buf, drjit_address_offset(p.address, PNANOVDB_ROOT_OFF_TABLE_SIZE));
}
#pragma endregion Root Handle


#pragma region Root Tile
struct drjit_root_tile_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_root_tile_handle_t)

PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_root_tile_get_key(drjit_buf_t buf, drjit_root_tile_handle_t p) {
    drjit_uint32_t byte_offset = drjit::full<drjit_uint32_t>(PNANOVDB_ROOT_TILE_OFF_KEY);
    return drjit_read_uint64(buf, drjit_address_offset(p.address, byte_offset));
}
PNANOVDB_FORCE_INLINE drjit_int64_t drjit_root_tile_get_child(drjit_buf_t buf, drjit_root_tile_handle_t p) {
    drjit_uint32_t byte_offset = drjit::full<drjit_uint32_t>(PNANOVDB_ROOT_TILE_OFF_CHILD);
    return drjit_read_int64(buf, drjit_address_offset(p.address, byte_offset));
}
#pragma endregion Root Tile


#pragma region Upper Handle
struct drjit_upper_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_upper_handle_t)

PNANOVDB_FORCE_INLINE drjit_bool_t drjit_upper_get_child_mask(drjit_buf_t buf, drjit_upper_handle_t p, drjit_uint32_t bit_index) {
    drjit_uint32_t bit_index_shift = bit_index.sr_(5u);// drjit::sr<5u>(bit_index);
    drjit_uint32_t upper_off_child = drjit::full<drjit_uint32_t>(PNANOVDB_UPPER_OFF_CHILD_MASK);
    drjit_uint32_t byte_offset = drjit::fmadd(4u, bit_index_shift, upper_off_child);
    
    // byte_offset = PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u);

    drjit_uint32_t value = drjit_read_uint32(buf, drjit_address_offset(p.address, byte_offset));
    drjit_uint32_t and_bit_index = bit_index & drjit::full<drjit_uint32_t>(31u);
    drjit_uint32_t shifted_value = value.sr_(and_bit_index); // Unable to call the drjit::sr<> for the and_bit_index
    // return ((value >> (bit_index & 31u)) & 1) != 0u;
    return drjit::neq<drjit_uint32_t, drjit_uint32_t>
        (
            (shifted_value & drjit::full<drjit_uint32_t>(1)),
            drjit::zeros<drjit_uint32_t>()
        );
}
#pragma endregion Upper Handle


#pragma region Lower Handle
struct drjit_lower_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_lower_handle_t)


PNANOVDB_FORCE_INLINE drjit_bool_t drjit_lower_get_child_mask(drjit_buf_t buf, drjit_lower_handle_t p, drjit_uint32_t bit_index) {
    drjit_uint32_t bit_index_shift = bit_index.sr_(5u);// drjit::sr<5u>(bit_index);
    // drjit_uint32_t bit_index_shift = drjit::sr<5u>(bit_index);
    drjit_uint32_t lower_off_child = drjit::full<drjit_uint32_t>(PNANOVDB_LOWER_OFF_CHILD_MASK);
    drjit_uint32_t byte_offset = drjit::fmadd(4u, bit_index_shift, lower_off_child);
    
    // byte_offset = PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u);

    drjit_uint32_t value = drjit_read_uint32(buf, drjit_address_offset(p.address, byte_offset));
    drjit_uint32_t and_bit_index = bit_index & drjit::full<drjit_uint32_t>(31u);
    drjit_uint32_t shifted_value = value.sr_(and_bit_index); // Unable to call the drjit::sr<> for the and_bit_index
    // return ((value >> (bit_index & 31u)) & 1) != 0u;
    return drjit::neq<drjit_uint32_t, drjit_uint32_t>
        (
            (shifted_value & drjit::full<drjit_uint32_t>(1)),
            drjit::zeros<drjit_uint32_t>()
        );
}
#pragma endregion Lower Handle


#pragma region Leaf
struct drjit_leaf_handle_t { drjit_address_t address = {drjit::full<UInt64Jit>(0)}; };
PNANOVDB_STRUCT_TYPEDEF(drjit_leaf_handle_t)
#pragma endregion Leaf


#pragma region Get Handle (Tree, Root)
PNANOVDB_FORCE_INLINE drjit_tree_handle_t drjit_grid_get_tree(drjit_buf_t buf, drjit_grid_handle_t grid)
{
    drjit_tree_handle_t tree = { grid.address };
    tree.address = drjit_address_offset(grid.address, drjit::full<UInt32Jit>(PNANOVDB_GRID_SIZE));
    return tree;
}

PNANOVDB_FORCE_INLINE drjit_root_handle_t drjit_tree_get_root(drjit_buf_t buf, drjit_tree_handle_t tree)
{
    drjit_root_handle_t root = { tree.address };
    drjit_uint64_t byte_offset = drjit_tree_get_node_offset_root(buf, tree);
    root.address = drjit_address_offset64(root.address, byte_offset);
    return root;
}

PNANOVDB_FORCE_INLINE drjit_root_tile_handle_t drjit_root_get_tile_zero(drjit_grid_type_t grid_type, drjit_root_handle_t root)
{
    drjit_root_tile_handle_t tile = { root.address };
    tile.address = drjit_address_offset(tile.address, drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, root_size)));
    return tile;
}

PNANOVDB_FORCE_INLINE drjit_upper_handle_t drjit_root_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, drjit_root_tile_handle_t tile)
{
    drjit_upper_handle_t upper = { root.address };
    upper.address = drjit_address_offset64(upper.address, drjit_int64_as_uint64(drjit_root_tile_get_child(buf, tile)));
    return upper;
}
#pragma endregion Get Handle (Tree, Root)


#pragma region Coord To Key
PNANOVDB_FORCE_INLINE drjit_uint64_t drjit_coord_to_key(PNANOVDB_IN(drjit_coord_t) ijk)
{
#if defined(PNANOVDB_NATIVE_64)
    drjit_uint64_t iu = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).x).sr_(12u);
    drjit_uint64_t ju = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).y).sr_(12u);
    drjit_uint64_t ku = drjit_int32_as_uint32(PNANOVDB_DEREF(ijk).z).sr_(12u);
    return ku.or_(ju.sl_(21u)).or_(iu.sl_(42u));
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
PNANOVDB_FORCE_INLINE drjit_root_tile_handle_t drjit_root_find_tile(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk)
{
    UInt32Jit tile_count = drjit_uint32_as_int32(drjit_root_get_tile_count(buf, root));
    drjit_root_tile_handle_t tile = drjit_root_get_tile_zero(grid_type, root);
    UInt32Jit tileFixedOffset = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size));
    UInt64Jit key = drjit_coord_to_key(ijk);
    
    uint32_t size = key.size();
    UInt32Jit offset = drjit::zeros<UInt32Jit>(size);
    // Loop seems to have an issue handling reads/tracked variables 
    // that are larget than tracked sample 
    // Because the max tile_count is predefined as well as zeroth tile
    // It is possible to overcome the read restriction inside of the loop to the outside
    // And only iterate over needed using i in the loop

    drjit_uint32_t byte_offset = drjit::full<drjit_uint32_t>(PNANOVDB_ROOT_TILE_OFF_KEY);
    // TODO - here:
    UInt32Jit temp = drjit::arange<UInt32Jit>(0, 8);
    UInt32Jit preReadTileOffset = drjit::fmadd(
        temp,
        tileFixedOffset, 
        byte_offset);

// [TODO] breaks here
    drjit_root_tile_handle_t dummyTileHandle = {drjit_address_t{drjit_address_offset(tile.address, preReadTileOffset)}};
    UInt64Jit preReadKeys = drjit_root_tile_get_key(buf, dummyTileHandle);
    
    UInt32Jit i = drjit::full<UInt32Jit>(0);
    BoolJit foundMask = drjit::full<BoolJit>(false);
    jit_var_set_label(foundMask.index(), "foundMask");
    BoolJit curMask = drjit::full<BoolJit>(false);
    jit_var_set_label(curMask.index(), "curMask");
    BoolJit modifyMask  = drjit::full<BoolJit>(false);
    jit_var_set_label(modifyMask.index(), "modifyMask");
    drjit::Loop<FloatJit::MaskType> loop("Root Find Tile", i, curMask, foundMask, modifyMask, tile, tile.address.byte_offset);
    
    while(loop(i < tile_count)){
        UInt64Jit currentKey = drjit::gather<UInt64Jit>(preReadKeys, i);
        curMask = drjit_uint64_is_equal(key, currentKey);
        BoolJit pos = foundMask.and_(curMask);
        BoolJit neg = foundMask.not_().and_(curMask);
        modifyMask = pos.or_(neg);
        foundMask = foundMask.or_(curMask);
        
        tile.address.byte_offset = drjit::select(
            modifyMask, 
            drjit::fmadd(i, tileFixedOffset, tile.address.byte_offset), 
            tile.address.byte_offset);
            
        i += 1;
    }

    drjit_root_tile_handle_t null_handle = { drjit_address_null() };
    tile.address.byte_offset = drjit::select(foundMask, tile.address.byte_offset, null_handle.address.byte_offset);
    return tile;
}
#pragma endregion Root Find Tile


#pragma region Leaf Node
PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_leaf_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk)
{
    Int32Jit constN = drjit::full<Int32Jit>(7);
    drjit_coord_t coord = PNANOVDB_DEREF(ijk);

    UInt32Jit x = (constN.and_(coord.x)).sr_(0).sl_(6);
    UInt32Jit y = (constN.and_(coord.y)).sr_(0).sl_(3);
    UInt32Jit z = (constN.and_(coord.z)).sr_(0);
        
    return x + y + z;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_leaf_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_leaf_handle_t node, drjit_uint32_t n)
{
    UInt32Jit byte_offset = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_table));
    UInt32Jit strideBits = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits)) * n;
    strideBits = strideBits.sr_(3u);
    byte_offset += strideBits;
    return drjit_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_leaf_get_value_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_leaf_handle_t leaf, PNANOVDB_IN(drjit_coord_t) ijk)
{
    UInt32Jit n = drjit_leaf_coord_to_offset(ijk);
    return drjit_leaf_get_table_address(grid_type, buf, leaf, n);
}
#pragma endregion Leaf Node


#pragma region Lower Node
PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_lower_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk)
{
    Int32Jit constN = drjit::full<Int32Jit>(127);
    drjit_coord_t coord = PNANOVDB_DEREF(ijk);
    
    UInt32Jit x = (constN.and_(coord.x)).sr_(3).sl_(8);
    UInt32Jit y = (constN.and_(coord.y)).sr_(3).sl_(4);
    UInt32Jit z = (constN.and_(coord.z)).sr_(3);
    
    return x + y + z;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_lower_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t node, drjit_uint32_t n)
{
    UInt32Jit byte_offset = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_table));
    UInt32Jit strideBits = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, table_stride)) * n;
    byte_offset += strideBits;
    return drjit_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE drjit_int64_t drjit_lower_get_table_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t node, drjit_uint32_t n)
{
    drjit_address_t table_address = drjit_lower_get_table_address(grid_type, buf, node, n);
    return drjit_read_int64(buf, table_address);
}

PNANOVDB_FORCE_INLINE drjit_leaf_handle_t drjit_lower_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t lower, drjit_uint32_t n)
{
    drjit_leaf_handle_t leaf = { lower.address };
    leaf.address = drjit_address_offset64(leaf.address, drjit_int64_as_uint64(drjit_lower_get_table_child(grid_type, buf, lower, n)));
    return leaf;
}
#pragma endregion Lower Node

#pragma region Last8
PNANOVDB_FORCE_INLINE drjit_address_t drjit_lower_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_lower_handle_t lower, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(drjit_uint32_t) level)
{
    drjit_uint32_t n = drjit_lower_coord_to_offset(ijk);
    drjit_address_t value_address;
    drjit_address_t leaf_address;
    drjit_address_t lower_address;
    BoolJit mask = drjit_lower_get_child_mask(buf, lower, n);

    drjit_leaf_handle_t child = drjit_lower_get_child(grid_type, buf, lower, n);
    leaf_address = drjit_leaf_get_value_address(grid_type, buf, child, ijk);
    lower_address = drjit_lower_get_table_address(grid_type, buf, lower, n);

    value_address.byte_offset = drjit::select(mask, leaf_address.byte_offset, lower_address.byte_offset);
    PNANOVDB_DEREF(level) = drjit::select(mask, 0u, 1u);

    return value_address;
}

PNANOVDB_FORCE_INLINE drjit_uint32_t drjit_upper_coord_to_offset(PNANOVDB_IN(drjit_coord_t) ijk)
{
    Int32Jit constN = drjit::full<Int32Jit>(4095);
    drjit_coord_t coord = PNANOVDB_DEREF(ijk);
    
    UInt32Jit x = (constN.and_(coord.x)).sr_(7).sl_(10);
    UInt32Jit y = (constN.and_(coord.y)).sr_(7).sl_(5);
    UInt32Jit z = (constN.and_(coord.z)).sr_(7);
    
    return x + y + z;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_upper_get_table_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t node, drjit_uint32_t n)
{
    UInt32Jit byte_offset = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_table));
    UInt32Jit strideBits = drjit::full<UInt32Jit>(PNANOVDB_GRID_TYPE_GET(grid_type, table_stride)) * n;
    byte_offset += strideBits;
    return drjit_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE drjit_int64_t drjit_upper_get_table_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t node, drjit_uint32_t n)
{
    drjit_address_t bufAddress = drjit_upper_get_table_address(grid_type, buf, node, n);
    return drjit_read_int64(buf, bufAddress);
}

PNANOVDB_FORCE_INLINE drjit_lower_handle_t drjit_upper_get_child(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t upper, drjit_uint32_t n)
{
    drjit_lower_handle_t lower = { upper.address };
    lower.address = drjit_address_offset64(lower.address, drjit_int64_as_uint64(drjit_upper_get_table_child(grid_type, buf, upper, n)));
    return lower;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_upper_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_upper_handle_t upper, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(drjit_uint32_t) level)
{
    drjit_uint32_t n = drjit_upper_coord_to_offset(ijk);
    drjit_address_t value_address;
    drjit_address_t lower_address;
    drjit_uint32_t lower_level;
    drjit_address_t upper_address;
    BoolJit mask = drjit_upper_get_child_mask(buf, upper, n);
    
    drjit_lower_handle_t child = drjit_upper_get_child(grid_type, buf, upper, n);
    lower_address = drjit_lower_get_value_address_and_level(grid_type, buf, child, ijk, &lower_level);
    upper_address = drjit_upper_get_table_address(grid_type, buf, upper, n);
    
    value_address.byte_offset = drjit::select(mask, lower_address.byte_offset, upper_address.byte_offset);
    PNANOVDB_DEREF(level) = drjit::select(mask, lower_level, 2u);

    return value_address;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_root_get_value_address_and_level(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk, PNANOVDB_INOUT(drjit_uint32_t) level)
{
 // default = FF, rewrite1 = T*, rewrite2 = FT
    drjit_root_tile_handle_t tile = drjit_root_find_tile(grid_type, buf, root, ijk);
    drjit_address_t ret;
    
    drjit_upper_handle_t child = drjit_root_get_child(grid_type, buf, root, tile);
    
    drjit_uint32_t dflt_level;
    drjit_address_t dflt = drjit_upper_get_value_address_and_level(grid_type, buf, child, ijk, &dflt_level);
    drjit_address_t valT = drjit_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background));
    drjit_address_t valFT = drjit_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value));

    BoolJit mask1 = drjit_address_is_null(tile.address);
    BoolJit mask2 = drjit_int64_is_zero(drjit_root_tile_get_child(buf, tile));
    
    ret.byte_offset = drjit::select(mask1, valT.byte_offset, dflt.byte_offset);
    ret.byte_offset = drjit::select(mask2, valFT.byte_offset, ret.byte_offset);
    PNANOVDB_DEREF(level) = drjit::select(mask1, 4u, dflt_level);
    PNANOVDB_DEREF(level) = drjit::select(mask2, 3u, PNANOVDB_DEREF(level));

    return ret;
}

PNANOVDB_FORCE_INLINE drjit_address_t drjit_root_get_value_address(drjit_grid_type_t grid_type, drjit_buf_t buf, drjit_root_handle_t root, PNANOVDB_IN(drjit_coord_t) ijk)
{
    drjit_uint32_t level;
    return drjit_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level));
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