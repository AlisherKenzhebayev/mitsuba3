#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/volumegrid.h>
#include <drjit/dynamic.h>
#include <drjit/texture.h>

#include <nanovdb/util/CreateNanoGrid.h>   // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/IO.h>
#include <random>

#include "vdb_header/impl_drjit.h"
// #define NANOVDB_NEW_ACCESSOR_METHODS
#define PNANOVDB_C
#define _WIN64
#define PNANOVDB_ADDRESS_64
#define PNANOVDB_C
#define PNANOVDB_BUF_BOUNDS_CHECK
// #include <nanovdb/PNanoVDB.h>

using namespace drjit;

using FloatArrayC  = dr::CUDAArray<float>;
using Int32ArrayC  = dr::CUDAArray<int32_t>;
using UInt32ArrayC = dr::CUDAArray<uint32_t>;
using UInt64ArrayC = dr::CUDAArray<uint64_t>;
using MaskArrayC   = dr::CUDAArray<bool>;
using FloatArrayL  = dr::LLVMArray<float>;
using Int32ArrayL  = dr::LLVMArray<int32_t>;
using UInt32ArrayL = dr::LLVMArray<uint32_t>;
using UInt64ArrayL = dr::LLVMArray<uint64_t>;
using MaskArrayL   = dr::LLVMArray<bool>;

// Include work from before on NanoVDB snippets, to at least enable reads from the file.

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class GridPnano final : public Volume<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Volume, update_bbox, m_to_local, m_bbox, m_channel_count)
    MI_IMPORT_TYPES(VolumeGrid)

    static constexpr bool IsCUDA = is_cuda_v<Float>;
    static constexpr bool IsLLVM = is_llvm_v<Float>;
    static constexpr bool IsJIT = IsLLVM || IsCUDA;

    GridPnano(const Properties &props) : Base(props) {
        // So that it matches some of grid types in PNanoVDB
        std::string allowed_grid_types[3] = {
            "unknown",
            "float",
            "double",
        }; 
        
        std::string grid_type_str = props.string("grid_type", "unknown");
        int index_found = -1;
        for (size_t i = 0; i < allowed_grid_types->size(); i++)
            if(allowed_grid_types[i] == grid_type_str){
                index_found = i;
                break;
            }
        
        if(index_found == -1 || grid_type_str == "unknown")
            Throw("Invalid grid type \"%s\", must be one of allowed ones!", grid_type_str);

        int grid_n = props.get<int>("grid_n", 0);
        
        // Load openVDB data by filename
        std::string file_path = props.string("vdb_filename", "");
        if(file_path.empty()){
            Throw("No filename under \"vdb_filename\" key is provided!");
        }

        if(!std::filesystem::exists(file_path))
            Log(Error, "\"%s\": file does not exist!", file_path);
        
        auto _gridCountHandle = nanovdb::io::readGrid(file_path, -1, true);
        int _totalGrids = _gridCountHandle.gridCount();
        std::string out_meta_string = "";
        for (size_t i = 0; i < (size_t)_totalGrids; i++)
        {
            out_meta_string = out_meta_string + std::to_string(i) + 
                "\t| " + _gridCountHandle.gridData(i)->gridName();
        }

        if(grid_n >= _totalGrids)
            Log(Error, "\"%d\": is not within the total #grids %d! \nCurrent data: \n%s", 
            grid_n, _totalGrids, out_meta_string);

        m_nanoHandle = nanovdb::io::readGrid(file_path, grid_n, true);
        if(m_nanoHandle){
            m_bboxVdb = m_nanoHandle.gridMetaData()->worldBBox();
            uint8_t *pGridData = m_nanoHandle.data();
            assert (pGridData != 0); 
            m_pGridData32 = (uint32_t*) pGridData;
        } else {
            Throw("GridHandle issues opening grid# %d", grid_n);
        }
        
        // Check for conforming with the gridType
        nanovdb::FloatGrid  *float_nanoGrid     = nullptr;
        nanovdb::DoubleGrid *double_nanoGrid    = nullptr;
        nanovdb::Int32Grid  *int32_nanoGrid     = nullptr;
        {
            switch (index_found)
            {
            case 0: // "unknown"
                Throw("GridHandle does not contain a grid with \"%s\" value type", grid_type_str);
                break;

            case 1:  // "float"
                float_nanoGrid = m_nanoHandle.grid<float>();
                m_pnanoGridType = PNANOVDB_GRID_TYPE_FLOAT;
                break;

            case 2:
                double_nanoGrid = m_nanoHandle.grid<double>();
                m_pnanoGridType = PNANOVDB_GRID_TYPE_DOUBLE;
                break;

            case 3:
                int32_nanoGrid = m_nanoHandle.grid<int32_t>();
                m_pnanoGridType = PNANOVDB_GRID_TYPE_INT32;
                break;
            // ... Expand past 

            default:
                break;
            }

            if (!float_nanoGrid 
                && !double_nanoGrid 
                && !int32_nanoGrid)
                Throw("GridHandle does not contain a grid with \"%s\" value type", grid_type_str);
        
            if(m_nanoHandle.gridMetaData()->hasMinMax()){
                float_nanoGrid->tree().extrema(m_min, m_max);
                // Throw("MIN MAX %f %f\" values", m_min, m_max);
            } else {
                Throw("Min max does not exist");
                m_max = 1.0f;
            }
        }

        // Copy the data from the file to drJIT
        m_byteSize = m_nanoHandle.size();
        m_floatSize = m_byteSize / 4;
        m_dataCopy = drjit::empty<Float>(m_floatSize);
        if constexpr (!IsJIT){
            // memcpy(m_dataCopy.data(), m_pGridData32, m_byteSize);
        }else if constexpr (IsLLVM){
            jit_memcpy(JitBackend::LLVM, m_dataCopy.data(), m_pGridData32, m_byteSize); 
        }else{
            jit_memcpy(JitBackend::CUDA, m_dataCopy.data(), m_pGridData32, m_byteSize); 
        }

        if (props.has_property("max_value")) {
            m_fixed_max = true;
            m_max = props.get<ScalarFloat>("max_value");
        }
        
        // Create a PNanoVDB handle, since it is easier to get the offset this way for the coords.
        
        uint32_t byteSize = (uint64_t) m_nanoHandle.size();
        uint32_t data32Size = byteSize / 4;

        // UInt32Jit jitDataCopy = drjit::empty<UInt32Jit>(data32Size);
        // uint32_t* jitDataPointer = (uint32_t*) jitDataCopy.data();
        // memcpy(jitDataCopy.data(), m_pGridData32, byteSize);
        // UInt64Jit sizeInWords = drjit::full<UInt64Jit>(m_nanoHandle.size(), 1);
        
        m_pnanoBuf = drjit_make_buf(m_pGridData32, m_nanoHandle.size());
            
            // drjit_make_buf(m_pGridData32, (uint64_t) m_nanoHandle.size());
        m_pnanoGridHandle = drjit_grid_handle_t();

        m_pnanoTreeHandle = drjit_grid_get_tree(m_pnanoBuf, m_pnanoGridHandle);
        m_pnanoRootHandle = drjit_tree_get_root(m_pnanoBuf, m_pnanoTreeHandle);

        // nanovdb::DefaultReadAccessor <float> readAccessor = float_nanoGrid->getAccessor();
        if constexpr (IsJIT){
                drjit_coord_t jitCoordData = { 
                    drjit::full<Int32Jit>(1, 10), 
                    drjit::full<Int32Jit>(1, 10), 
                    drjit::full<Int32Jit>(1, 10) };
                
                char *str = strdup(jit_var_graphviz());
                FILE *f_out = fopen("./graphviz.txt", "wb");
                fputs(str, f_out);
                fclose(f_out);


                drjit_address_t pnanoAddress = drjit_root_get_value_address(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &jitCoordData);            
                Float result = drjit_read_float(m_pnanoBuf, pnanoAddress);
        }
    }

    UnpolarizedSpectrum eval(const Interaction3f &it,
                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        return interpolate_1(it, active);
    }

    Float eval_1(const Interaction3f &it, Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        
        return interpolate_1(it, active);
    }


    MI_INLINE Float interpolate_1(const Interaction3f &it, Mask active) const {
        MI_MASK_ARGUMENT(active);
        drjit_coord_t pnanoCoordinateTest;
        
        Point3f p = m_to_local * it.p;
        Float result;

        // dr::width(p);
        
            // TODO: attempt at extracting the data from Point3f p;
        
        // Point3f detachedPoint3f = drjit::detach(p);
        // Float t_examine = detachedPoint3f.data()[0];
        
        // float t2 = drjit::slice(drjit::detach(t_examine), 0);
        // Point t_data = drjit::slice(detached, 0);

        // FloatArrayL xDiffData = drjit::detach(p.x());

        auto pointerWidth = dr::width(p);
        auto itData = it.p.data(); //TODO: it.p seems to do something, but not sure what>?
        auto pData = p.data();
        auto dataWidth = dr::width(itData);
        auto testWidth = dr::width(itData[0]);
        
        Float tempx = itData[0];
        Float tempy = itData[1];
        Float tempz = itData[2];
        
        // Test writing directly back, 
        // scalar_rgb - OK
        // p.data()[0] += 5;
        // auto data2 = p.data();

// #if defined(MI_ENABLE_LLVM) || defined(MI_ENABLE_CUDA)
//         if(jit_has_backend(JitBackend::LLVM)){
//             drjit_coord_t pnanoCoordinateTest;
//             pnanoCoordinateTest.x = (p.x());
//             pnanoCoordinateTest.y = (p.y());
//             pnanoCoordinateTest.z = (p.z());
            
//             Float result;
//             drjit_address_t pnanoAddress = drjit_root_get_value_address(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &pnanoCoordinateTest);
            
//             result = drjit_read_float(m_pnanoBuf, pnanoAddress);

//             return result;
//         }
//         else if (jit_has_backend(JitBackend::CUDA))
//         {
//             // TODO::
//             auto i = 0;
//         }
//         else
//         {
//             float x = *pData;
//             float y = *(pData+1);
//             float z = pData[2];

//             // Fix this by using the bbox values?
//             pnanoCoordinateTest.x = int32_t(x * 100.f);
//             pnanoCoordinateTest.y = int32_t(y * 100.f);
//             pnanoCoordinateTest.z = int32_t(z * 100.f);

//             drjit_address_t pnanoAddress = drjit_root_get_value_address(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &pnanoCoordinateTest);

//             Float tempL = drjit_read_float(m_pnanoBuf, pnanoAddress);

//             result = tempL.data()[0];
//         }
// #else
        // ISSUE #1 - Cannot copy memory from the handle to drjit, since no backend is initialized. 
        // I could initialize it, but it really doesn't make sense to?

        if constexpr (IsJIT){
            Float x = p.x();
            Float y = p.y();
            Float z = p.z();

            auto min = m_bboxVdb.min();
            auto res = resolution(); // Order is 2-1-0 in resolution() -> need to swap x,z
            // pnanoCoordinateTest.x = z;//(z * res[2] + (int)min[0]);
            // pnanoCoordinateTest.y = y;//(y * res[1] + (int)min[1]);
            // pnanoCoordinateTest.z = x;//(x * res[0] + (int)min[2]);
            // drjit_coord_t jitCoordData = { 
            //     drjit::full<Int32Jit>(1, 10), 
            //     drjit::full<Int32Jit>(1, 10), 
            //     drjit::full<Int32Jit>(1, 10) };
            
            // char *str = strdup(jit_var_graphviz());
            // FILE *f_out = fopen("./graphviz.txt", "wb");
            // fputs(str, f_out);
            // fclose(f_out);


            // drjit_address_t pnanoAddress = drjit_root_get_value_address(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &jitCoordData);            
            // result = drjit_read_float(m_pnanoBuf, pnanoAddress);
            
            // drjit_root_tile_handle_t tile = drjit_root_find_tile(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &jitCoordData);
            // result = tile.address.byte_offset;
            // drjit_upper_handle_t child = drjit_root_get_child(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, tile);
            // result = child.address.byte_offset;
            
            // drjit_address_t testAddress = {drjit::full<UInt64Jit>(44531060)};
            // result = drjit_read_uint64(m_pnanoBuf, testAddress);
        } else {
            result = 0;
        }
        

// #endif
        // drjit::LLVMArray<float> asd1temp = drjit::detach(p.x());
        // float temp = (drjit::slice(asd1temp, 0));

        // pnanovdb_address_t pnanoAddress = pnanovdb_root_get_value_address(m_pnanoGridType, m_pnanoBuf, m_pnanoRootHandle, &pnanoCoordinateTest);
        // // uint64_t offset = pnanoAddress.byte_offset;

        // float pnanoValue = pnanovdb_read_float(m_pnanoBuf, pnanoAddress);

        // UInt64ArrayC jitOffsets = drjit::zeros<UInt64ArrayC>(p.x()[0]);
        // for (auto i = 0; i < p.shape()[0]; ++i) 
        // {
        //     
        //     pnanovdb_address_t pnanoAddress = pnanovdb_root_get_value_address(pnanoGridType, pnanoBuf, pnanoRootHandle, &pnanoCoordinateTest);
        //     uint64_t offset = pnanoAddress.byte_offset;

        //     // Record the offset both to the int + UInt32Jit
        //     recordedOffsets[i] = offset;
        //     jitOffsets.data()[i] = offset >> 2u;
        // }

        // // Then run a gather by indices
        // FloatArrayL dataGather = drjit::gather<FloatJit>(m_dataCopy, jitOffsets);

        return result;
}

    ScalarFloat max() const override { 
        return m_max;
    }

    // void max_per_channel(ScalarFloat *out) const override {
    //     for (size_t i=0; i<m_max_per_channel.size(); ++i)
    //         out[i] = m_max_per_channel[i];
    // }

    ScalarVector3i resolution() const override {
        auto minBboxVdb = m_bboxVdb.min();
        auto maxBboxVdb = m_bboxVdb.max();
        return 
            { (int) (maxBboxVdb[2] - minBboxVdb[2]), 
              (int) (maxBboxVdb[1] - minBboxVdb[1]), 
              (int) (maxBboxVdb[0] - minBboxVdb[0]) };
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "GridTest[" << std::endl
            << "  to_local = " << string::indent(m_to_local, 13) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  bboxVdb = " << string::indent(m_bboxVdb) << "," << std::endl
            << "  dimensions = " << resolution() << "," << std::endl
            << "  min = " << m_min << "," << std::endl
            << "  max = " << m_max << "," << std::endl
            // << "  channels = " << m_texture.shape()[3] << std::endl
            << "]";
        return oss.str();
    }

    void traverse(TraversalCallback *callback) override {   
        // callback->put_parameter("data", m_texture.tensor(), +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    MI_DECLARE_CLASS()

protected:
    // /**
    //  * \brief Returns the number of channels in the grid
    //  *
    //  * For object instances that perform spectral upsampling, the channel that
    //  * holds all scaling coefficients is omitted.
    //  */
    // MI_INLINE size_t nchannels() const {
    //     const size_t channels = m_texture.shape()[3];
    //     // When spectral upsampling is requested, a fourth channel is added to
    //     // the internal texture data to handle scaling coefficients.
    //     if (is_spectral_v<Spectrum> && channels == 4 && !m_raw)
    //         return 3;

    //     return channels;
    // }

protected:
    nanovdb::GridHandle<nanovdb::HostBuffer> m_nanoHandle;
    uint32_t *m_pGridData32 = nullptr;
    uint32_t m_byteSize;
    uint32_t m_floatSize;

// DrJit implementation -> LLVM, CUDA support
    drjit_buf_t m_pnanoBuf;
    drjit_grid_type_t m_pnanoGridType;
    drjit_grid_handle_t m_pnanoGridHandle;
    drjit_tree_handle_t m_pnanoTreeHandle;
    drjit_root_handle_t m_pnanoRootHandle;

// Scalar support
    
    
    // DrJIT memcopy for the buffer size
    Float m_dataCopy;
    // Texture3f m_texture;

    nanovdb::BBoxR m_bboxVdb;
    
    bool m_accel;

    bool m_raw;

    bool m_fixed_max = false;
    ScalarFloat m_min;
    ScalarFloat m_max;

    std::vector<ScalarFloat> m_max_per_channel;
};

MI_IMPLEMENT_CLASS_VARIANT(GridPnano, Volume)
MI_EXPORT_PLUGIN(GridPnano, "GridTest texture")

NAMESPACE_END(mitsuba)
