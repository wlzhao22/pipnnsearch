#ifndef PARANN_H_
#define PARANN_H_

#include <cuda_runtime.h>
#include "phms.h"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include "phms_v3_options.h"

__global__ void neighbor_sort_warp_v3(unsigned* neighbors, unsigned* counts, float* distances,
                                      bool* next_iter, unsigned degree, unsigned queries);
#include <vector>

#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

typedef enum _DataTypeEnum
{
    ENUM_UNSIGNED_INT8 = 0,
    ENUM_UNSIGNED_INT32,
    ENUM_FLOAT

} DataTypeEnum;

typedef struct __attribute__((__packed__)) _GraphMedatadata
{
    unsigned long long ullMedoid;
    unsigned long long ulluIndexEntryLen;
    int uDatatype;
    unsigned uDim;
    unsigned uDegree;
    unsigned uDatasetSize;
} GraphMedataData;

typedef struct _IndexLoad
{
    unsigned long long MEDOID;
    unsigned uDataType;
    unsigned D;
    unsigned R;
    unsigned N;
    unsigned int uChunks;

    off_t size_indexfile;
    unsigned long long ullIndex_Entry_LEN;

    uint8_t* d_compressedVectors;
    float* d_pqTable;
    unsigned* d_chunksOffset;
    float* d_centroid;

    unsigned uShardStart;
    unsigned uShardEnd;

    float* h_pqTable_hostcopy;
    float* h_centroid_hostcopy;
    unsigned* h_chunksOffset_hostcopy;

    uint8_t* pIndex;
} IndexLoad;

typedef struct _GPUInstance
{
    int physical_device = -1;
    unsigned* d_query_seeds = nullptr;
    unsigned* d_resident_query_ids = nullptr;
    unsigned query_capacity = 0;
    unsigned resident_selection_count = 0;
    unsigned numThreads_K4;
    unsigned numThreads_K1;
    unsigned numThreads_K2;
    unsigned numThreads_K3_merge;
    unsigned numThreads_K3;
    unsigned K4_blockSize;
    unsigned numThreads_K5;
    void* d_queriesFP;
    result_ann_t* d_nearestNeighbours = NULL;
    float* d_pqDistTables;
    float* d_BestLSetsDist;
    unsigned* d_BestLSets_count;
    unsigned* d_BestLSets;
    bool* d_BestLSets_visited;
    unsigned* d_parents;
    unsigned* d_ngpu_parents;
    float* d_neighborsDist_query;
    float* d_neighborsDist_query_aux;
    unsigned* d_neighbors;
    unsigned* d_neighbors_aux;
    unsigned* d_numNeighbors_query;
    unsigned* d_neighbors_temp;
    unsigned* d_numNeighbors_query_temp;
    unsigned* d_neighbors_temp_alt;
    unsigned* d_numNeighbors_query_temp_alt;
    unsigned* d_iter;
    unsigned* d_mark;
    unsigned* d_mark_alt;
    unsigned* d_expansion_upload;
    unsigned* d_expansion_upload_alt;
    bool* d_nextIter;
    bool* d_processed_bit_vec;

    void* d_FPSetCoordsList;
    unsigned* d_FPSetCoordsList_Counts;
    float* d_L2distances;
    unsigned* d_L2ParentIds;
    float* d_L2distances_aux;
    unsigned* d_L2ParentIds_aux;

    unsigned uShardStart;
    unsigned uShardEnd;

    cudaStream_t streamFPTransfers;
    cudaStream_t streamParent;
    cudaStream_t streamChildren;
    cudaStream_t streamKernels;
    cudaEvent_t eventChildrenReady;
    cudaEvent_t eventChildrenReadyAlt;
    cudaEvent_t eventKernelsReady;
    cudaEvent_t eventParentCopyDone = nullptr;
    cudaGraphExec_t ngpuExpansionGraphs[2] = {nullptr, nullptr};

} GPUInstance;

typedef struct _HostInstance
{
    unsigned numCPUthreads;
    unsigned* parents = NULL;
    unsigned* neighbors = NULL;
    unsigned* numNeighbors_query = NULL;
    unsigned* expansion_upload = NULL;
    unsigned* expansion_upload_alt = NULL;
    void* FPSetCoordsList;
    unsigned long long FPSetCoords_size_bytes;
    unsigned long long FPSetCoords_rowsize_bytes;
    unsigned long long FPSetCoords_size;
    unsigned long long FPSetCoords_rowsize;

} HostInstance;

typedef struct _SearchParams
{
    int recall;
    int worklist_length;
    unsigned uDistFunc;
} SearchParams;

template <typename T>
__global__ void populate_pqDist_par(float* d_pqTable, float* d_pqDistTables, T* d_queriesFP,
                                    unsigned* d_chunksOffset, float* d_centroid, unsigned n_chunks,
                                    unsigned long long D, unsigned n_DimAdjust = 0);

template <typename T>
__global__ void compute_L2Dist(T* d_FPSetCoordsList, unsigned* d_FPSetCoordsList_Counts,
                               T* d_queriesFP, unsigned* d_L2ParentIds, float* d_L2distances,
                               unsigned d_numQueries, unsigned long long D,
                               unsigned n_DimAdjust = 0);

__global__ void compute_neighborDist_par(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                         uint8_t* d_compressedVectors, float* d_pqDistTables,
                                         float* d_neighborsDist_query, unsigned n_chunks,
                                         unsigned R, unsigned uShardStart);

__global__ void neighbor_filtering_distance_fused(
    unsigned* d_neighbors, unsigned* d_neighbors_temp, unsigned* d_numNeighbors_query,
    unsigned* d_numNeighbors_query_temp, bool* d_processed_bit_vec, uint8_t* d_compressedVectors,
    float* d_pqDistTables, float* d_neighborsDist_query, unsigned n_chunks, unsigned R,
    unsigned uShardStart, unsigned uShardEnd, const unsigned* input_offsets = nullptr,
    const unsigned* input_masks = nullptr);

__global__ void ngpu_expansion_fused(unsigned* d_neighbors, unsigned* d_neighbors_input,
                                     unsigned* d_numNeighbors_query, unsigned* d_counts_input,
                                     float* d_neighborsDist_query, unsigned* d_BestLSets,
                                     float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                     unsigned* d_BestLSets_count, unsigned* d_mark, unsigned iter,
                                     unsigned uWLLen, unsigned long long MEDOID,
                                     bool* d_processed_bit_vec, uint8_t* d_compressedVectors,
                                     float* d_pqDistTables, unsigned n_chunks, unsigned R,
                                     unsigned uShardStart, unsigned uShardEnd,
                                     const unsigned* query_seeds = nullptr);

__global__ void compute_parent1(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                float* d_neighborsDist_query, unsigned* d_BestLSets,
                                float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
                                unsigned* d_mark, unsigned* d_iter, unsigned* d_L2ParentIds,
                                unsigned* d_FPSetCoordsList_Counts, unsigned d_numQueries,
                                unsigned long long MEDOID, unsigned R);

__global__ void compute_parent1_ngpu2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                      float* d_neighborsDist_query, unsigned* d_ngpu_parents,
                                      bool* d_nextIter, unsigned d_numQueries,
                                      unsigned long long MEDOID, unsigned R,
                                      const unsigned* query_seeds = nullptr);

__global__ void compute_parent2_ngpu2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                      float* d_neighborsDist_query, unsigned* d_BestLSets,
                                      float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                      unsigned* d_BestLSets_count, unsigned* d_ngpu_parents,
                                      bool* d_nextIter, unsigned d_numQueries, unsigned uWLLen,
                                      unsigned long long MEDOID, unsigned R,
                                      const unsigned* query_seeds = nullptr);

__global__ void compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                float* d_neighborsDist_query, unsigned* d_BestLSets,
                                float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
                                unsigned* d_mark, unsigned* d_iter, unsigned* d_L2ParentIds,
                                unsigned* d_FPSetCoordsList_Counts, unsigned d_numQueries,
                                unsigned uWLLen, unsigned long long MEDOID, unsigned R);

__global__ void compute_BestLSets_par_sort_msort(unsigned* d_neighbors, unsigned* d_neighbors_aux,
                                                 unsigned* d_neighbors_offset,
                                                 float* d_neighborsDist_query,
                                                 float* d_neighborsDist_query_aux, bool* d_nextIter,
                                                 unsigned R);

__global__ void compute_BestLSets_par_merge(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                            float* d_neighborsDist_query, unsigned* d_BestLSets,
                                            float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                            unsigned* d_parents, unsigned iter, bool* d_nextIter,
                                            unsigned* d_BestLSets_count, unsigned* d_mark,
                                            unsigned uWLLen, unsigned long long MEDOID, unsigned R,
                                            const unsigned* query_seeds = nullptr);

__global__ void neighbor_filtering_new(unsigned* d_neighbors, unsigned* d_neighbors_temp,
                                       unsigned* d_numNeighbors_query,
                                       unsigned* d_numNeighbors_query_temp,
                                       bool* d_processed_bit_vec, unsigned R, unsigned uShardStart,
                                       unsigned uShardEnd);

__global__ void compute_NearestNeighbours(unsigned* d_L2ParentIds, unsigned* d_L2ParentIds_aux,
                                          unsigned* d_FPSetCoordsList_Counts, float* d_L2distances,
                                          float* d_L2distances_aux,
                                          result_ann_t* d_nearestNeighbours, unsigned d_numQueries,
                                          unsigned d_recall);

__global__ void compute_neighborDist_par_cachewarmup(unsigned* d_neighbors,
                                                     uint8_t* d_compressedVectors,
                                                     unsigned n_chunks, unsigned R);

__device__ unsigned upper_bound_d(float arr[], unsigned lo, unsigned hi, float target);

__device__ unsigned lower_bound_d(float arr[], unsigned lo, unsigned hi, float target);

__device__ unsigned lower_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,
                                     unsigned row_size, unsigned queryID);

__device__ unsigned upper_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,
                                     unsigned row_size, unsigned queryID);

__device__ unsigned hashFn1_d(unsigned x);
__device__ unsigned hashFn2_d(unsigned x);

template <typename T> inline std::string getValues(T* data, size_t num)
{
    std::stringstream stream;
    stream << "[";
    for (size_t i = 0; i < num; i++)
    {
        stream << std::to_string(data[i]) << ",";
    }
    stream << "]" << std::endl;

    return stream.str();
}

template <typename T>
inline bool load_bin_impl(std::basic_istream<char>& reader, size_t actual_file_size, T*& data,
                          size_t& npts, size_t& dim)
{
    int npts_i32;
    int dim_i32;
    reader.read((char*)&npts_i32, sizeof(int));
    reader.read((char*)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size
               << " while expected size is  " << expected_actual_file_size << " npts = " << npts
               << " dim = " << dim << " size of <T>= " << sizeof(T) << std::endl;
        std::cout << stream.str();
        return false;
    }

    data = new T[npts * dim];
    reader.read((char*)data, npts * dim * sizeof(T));

    std::cout << "Finished reading bin file." << std::endl;
    return true;
}

template <typename T>
inline bool load_bin(const std::string& bin_file, T*& data, size_t& npts, size_t& dim)
{

    std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    if (reader.fail())
    {
        std::cout << "Reading bin file " << bin_file.c_str() << " failed. Error:" << strerror(errno)
                  << std::endl;
        return false;
    }

    uint64_t fsize = reader.tellg();
    reader.seekg(0);

    return load_bin_impl<T>(reader, fsize, data, npts, dim);
}

template <typename T> class PHMSSearchInner
{
    IndexLoad m_objInputData;
    SearchParams m_objSearchParams;
    GPUInstance m_objGPUInst;
    HostInstance m_objHostInst;
    const T* m_resident_host_source = nullptr;
    const unsigned* m_resident_host_ids = nullptr;
    unsigned m_expansion_budget_override = 0;

  public:
    void phms_set_expansion_budget(unsigned budget)
    {
        m_expansion_budget_override = budget;
    }

    bool phms_load(char* indexfile_path_prefix);

    void phms_alloc(int numQueries);

    void phms_prepare_ngpu_graph(int numQueries);

    void phms_select_resident_queries(const T* device_source, unsigned total_queries,
                                      const unsigned* host_ids, unsigned count,
                                      const T* host_source = nullptr);

    void phms_init(int numQueries);

    void phms_set_searchparams(int recall, int worklist_length, DistFunc nDistFunc = ENUM_DIST_L2);

    void phms_query(T* query_array, int num_queries, result_ann_t* nearestNeighbours,
                    float* nearestNeighbours_dist);

    void phms_free();

    void phms_unload();

    bool phms_load_sharded(char* indexfile_path_prefix, unsigned shard_start, unsigned shard_end);

    static void ngpu_phms_query(std::vector<PHMSSearchInner<T>*>& gpu_impls, T* query_array,
                                int num_queries, result_ann_t* nearestNeighbours,
                                float* nearestNeighbours_dist,
                                const std::vector<unsigned>& shard_ends,
                                const unsigned* query_seeds = nullptr,
                                std::vector<std::vector<unsigned>>* debug_tracks = nullptr);
};

#endif
