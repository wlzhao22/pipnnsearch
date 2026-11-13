#include "phms_search.cuh"
#include "utils/utils.h"
#include "phms_shard_adjacency.h"
#include "phms_v3_options.h"
#include "phms_compact_pack.h"
#include "phms_graphcut_batches.h"
#include "phms_timeline.h"
#include <memory>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <sys/mman.h>
#include <chrono>
#include <cerrno>

#include <cub/cub.cuh>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#ifdef PHMS_INDEX_LAYOUT_EXPERIMENT
static uint8_t* g_thp_index = nullptr;
static size_t g_thp_index_bytes = 0;
static uint8_t* phms_thp_index_alloc(size_t bytes)
{
    const size_t alignment = 2ull * 1024 * 1024;
    const size_t length = (bytes + alignment - 1) & ~(alignment - 1);
    uint8_t* reserve = static_cast<uint8_t*>(mmap(
        nullptr, length + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (reserve == MAP_FAILED)
        throw std::runtime_error("THP index mmap failed");
    uintptr_t raw = reinterpret_cast<uintptr_t>(reserve),
              aligned = (raw + alignment - 1) & ~uintptr_t(alignment - 1);
    size_t prefix = aligned - raw;
    size_t suffix = alignment - prefix;
    if (prefix)
        munmap(reserve, prefix);
    if (suffix)
        munmap(reinterpret_cast<void*>(aligned + length), suffix);
    g_thp_index = reinterpret_cast<uint8_t*>(aligned);
    g_thp_index_bytes = length;
    int result = madvise(g_thp_index, length, MADV_HUGEPAGE);
    printf(
        "[INDEX THP] requested_bytes=%zu mapped_bytes=%zu madvise_result=%d errno=%d address=%p\n",
        bytes, length, result, result ? errno : 0, g_thp_index);
    return g_thp_index;
}
#endif
#include <atomic>
#include <vector>
#include <mutex>
using namespace std::chrono;
using namespace std;
static std::mutex g_phms_index_mtx;
static std::mutex g_ngpu_profile_print_mtx;
static std::atomic<unsigned> g_ngpu_profile_sequence{0};
static std::atomic<unsigned> g_ngpu_submit_sequence{0};
static IndexLoad g_phms_index_cache;
static bool g_phms_index_loaded = false;
static int g_phms_index_refcnt = 0;
static std::shared_ptr<PhmsShardAdjacency> g_shard_adjacency;

static bool phms_compact_upload_enabled()
{
    const char* value = std::getenv("PHMS_COMPACT_SHARD_UPLOAD");
    if (!value || std::atoi(value) == 0)
        return false;
    const char* pack = std::getenv("PHMS_PACK_EXPANSION_UPLOAD");
    const char* filter = std::getenv("PHMS_FUSE_FILTER_DISTANCE");
    const char* expansion = std::getenv("PHMS_FUSE_EXPANSION");
    if ((pack && !std::atoi(pack)) || (filter && !std::atoi(filter)) ||
        (expansion && std::atoi(expansion)))
        throw std::runtime_error(
            "compact upload requires packed upload, fused filter, and PHMS_FUSE_EXPANSION=0");
    return true;
}

__device__ __constant__ const uint64_t* phms_gc_bits = nullptr;
__device__ __constant__ const unsigned* phms_gc_prefix = nullptr;
__device__ __constant__ unsigned phms_gc_nodes = 0;
__device__ __forceinline__ bool phms_gc_member(unsigned id, unsigned begin, unsigned end)
{
    if (phms_gc_bits)
        return id < phms_gc_nodes && (phms_gc_bits[id / 64] & (1ull << (id % 64)));
    return begin == end || (id >= begin && id < end);
}
__device__ __forceinline__ unsigned long long phms_gc_row(unsigned id, unsigned begin)
{
    if (phms_gc_bits)
        return (unsigned long long)phms_gc_prefix[id / 64] +
               __popcll(phms_gc_bits[id / 64] & ((1ull << (id % 64)) - 1));
    return (unsigned long long)(id - begin);
}
struct PerGpuCache
{
    uint8_t* d_compressedVectors = nullptr;
    float* d_pqTable = nullptr;
    unsigned* d_chunksOffset = nullptr;
    float* d_centroid = nullptr;
    uint64_t* gc_bits = nullptr;
    unsigned* gc_prefix = nullptr;
    int refcnt = 0;
};
static std::map<int, PerGpuCache> g_per_gpu_cache;
static std::mutex g_per_gpu_mtx;

static unsigned phms_ngpu_launch_threads(const char* name, unsigned default_value, unsigned minimum,
                                         unsigned maximum, unsigned multiple)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (*end != '\0' || parsed < minimum || parsed > maximum || (parsed % multiple) != 0)
    {
        fprintf(stderr, "[PHMS NGPU] ignoring invalid %s=%s; using %u\n", name, value,
                default_value);
        return default_value;
    }
    return static_cast<unsigned>(parsed);
}

static bool phms_compact_tile_enabled()
{
    const char* value = std::getenv("PHMS_COMPACT_TILE_PACK");
    return phms_compact_upload_enabled() && value && std::atoi(value) != 0;
}

static size_t phms_compact_fixed_payload(int nq, unsigned r, unsigned gpus)
{
    const unsigned rows =
        std::min<unsigned>(nq, phms_ngpu_launch_threads("PHMS_CPU_QBLK", 64, 8, 64, 8));
    const unsigned quota = (r + gpus - 1) / gpus;
    return ((nq + rows - 1) / rows) * (size_t)rows * quota;
}

#define MAX_R 64

#define PQ_PIVOTS_FILE_SUFFIX "_pq_pivots.bin"
#define PQ_COMPRESSEDVECTORS_FILE_SUFFIX "_pq_compressed.bin"
#define PQ_PIVOTS_NUM_SECTIONS (3)
#define GRAPH_INDEX_FILE_SUFFIX "_disk.bin"
#define GRAPH_INDEX_METADATA_FILE_SUFFIX "_disk_metadata.bin"

#ifndef PHMS_V3_BF_ENTRIES
#define PHMS_V3_BF_ENTRIES 399887U
#endif
static_assert(PHMS_V3_BF_ENTRIES >= 1024U && PHMS_V3_BF_ENTRIES <= 399887U,
              "V3 Bloom entries must be between 1024 and the original 399887");
#define BF_ENTRIES PHMS_V3_BF_ENTRIES
const unsigned BF_MEMORY = ((BF_ENTRIES + 31u) / 32u) * sizeof(unsigned);
__device__ __forceinline__ bool phms_bloom_test(bool* storage, unsigned hash)
{
    return (atomicOr(reinterpret_cast<unsigned*>(storage) + (hash >> 5), 0u) &
            (1u << (hash & 31))) != 0;
}
__device__ __forceinline__ void phms_bloom_set(bool* storage, unsigned hash)
{
    atomicOr(reinterpret_cast<unsigned*>(storage) + (hash >> 5), 1u << (hash & 31));
}

#define NAX_EXTRA_ITERATION (50)
#define MAX_PARENTS_PERQUERY (MAX_L + NAX_EXTRA_ITERATION)

#define SIZEPARENTLIST (1 + 1 + 1)
#ifndef PHMS_V3_PARENTS_PER_ROUND
#define PHMS_V3_PARENTS_PER_ROUND 2
#endif
static_assert(PHMS_V3_PARENTS_PER_ROUND == 2 || PHMS_V3_PARENTS_PER_ROUND == 4,
              "supported V3 global beams: 2,4");
#define NGPU_PARENT_RECORD_SIZE (1 + 2 * PHMS_V3_PARENTS_PER_ROUND)
#define INVALID_NODE_ID 0xFFFFFFFFu

enum NgpuExpansionGraphVariant
{
    NGPU_EXPANSION_GRAPH_BUFFER0_WITH_SORT = 0,
    NGPU_EXPANSION_GRAPH_BUFFER1_NO_SORT = 1,
    NGPU_EXPANSION_GRAPH_COUNT = 2
};

#define NGPU_PARENTS_PER_ROUND PHMS_V3_PARENTS_PER_ROUND

#define _HUGEPAGES

using namespace std;
using Clock = std::chrono::high_resolution_clock;

template <typename T> PHMSSearch<T>::PHMSSearch()
{
    m_pImpl = new PHMSSearchInner<T>();
}

template <typename T> PHMSSearch<T>::~PHMSSearch()
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    delete pobjPhmsInner;
}

template <typename T> bool PHMSSearch<T>::phms_load(char* indexfile_path_prefix)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    return pobjPhmsInner->phms_load(indexfile_path_prefix);
}

template <typename T> void PHMSSearch<T>::phms_alloc(int numQueries)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_alloc(numQueries);
}

template <typename T> void PHMSSearch<T>::phms_prepare_ngpu_graph(int numQueries)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_prepare_ngpu_graph(numQueries);
}

template <typename T> void PHMSSearch<T>::phms_init(int numQueries)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_init(numQueries);
}

template <typename T>
void PHMSSearch<T>::phms_set_searchparams(int recall, int worklist_length, DistFunc nDistFunc)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    return pobjPhmsInner->phms_set_searchparams(recall, worklist_length, nDistFunc);
}

template <typename T> void PHMSSearch<T>::phms_set_expansion_budget(unsigned budget)
{
    static_cast<PHMSSearchInner<T>*>(m_pImpl)->phms_set_expansion_budget(budget);
}

template <typename T>
void PHMSSearch<T>::phms_query(T* query_array, int num_queries, result_ann_t* nearestNeighbours,
                               float* nearestNeighbours_dist)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_query(query_array, num_queries, nearestNeighbours, nearestNeighbours_dist);
}

template <typename T> void PHMSSearch<T>::phms_free()
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_free();
}

template <typename T>
void PHMSSearch<T>::phms_select_resident_queries(const T* source, unsigned total,
                                                 const unsigned* ids, unsigned count,
                                                 const T* host_source)
{
    static_cast<PHMSSearchInner<T>*>(m_pImpl)->phms_select_resident_queries(source, total, ids,
                                                                            count, host_source);
}

template <typename T>
__global__ void phms_v4_gather_queries(const T* source, const unsigned* ids, T* destination,
                                       unsigned count, unsigned dimension)
{
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < (size_t)count * dimension)
        destination[k] = source[(size_t)ids[k / dimension] * dimension + k % dimension];
}

template <typename T>
void PHMSSearchInner<T>::phms_select_resident_queries(const T* source, unsigned total,
                                                      const unsigned* ids, unsigned count,
                                                      const T* host_source)
{
    GPUInstance& gi = m_objGPUInst;
    if (!source || !ids || !count || count > gi.query_capacity)
        throw std::runtime_error("invalid V4 resident selection capacity");
    for (unsigned i = 0; i < count; ++i)
        if (ids[i] >= total)
            throw std::runtime_error("V4 resident query ID out of range");
    gpuErrchk(cudaSetDevice(gi.physical_device));
    gpuErrchk(cudaMemcpyAsync(gi.d_resident_query_ids, ids, count * sizeof(unsigned),
                              cudaMemcpyHostToDevice, gi.streamKernels));
    const size_t elements = (size_t)count * m_objInputData.D;
    phms_v4_gather_queries<<<(elements + 255) / 256, 256, 0, gi.streamKernels>>>(
        source, gi.d_resident_query_ids, (T*)gi.d_queriesFP, count, m_objInputData.D);
    gpuErrchk(cudaGetLastError());
    gi.resident_selection_count = count;
    m_resident_host_source = host_source;
    m_resident_host_ids = ids;
}

template <typename T> void PHMSSearch<T>::phms_unload()
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    pobjPhmsInner->phms_unload();
}

template <typename T>
bool PHMSSearch<T>::phms_load_sharded(char* indexfile_path_prefix, unsigned shard_start,
                                      unsigned shard_end)
{
    PHMSSearchInner<T>* pobjPhmsInner = static_cast<PHMSSearchInner<T>*>(m_pImpl);
    try
    {
        return pobjPhmsInner->phms_load_sharded(indexfile_path_prefix, shard_start, shard_end);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "[PHMS load error] %s\n", e.what());
        return false;
    }
}

template <typename T>
void PHMSSearch<T>::ngpu_phms_query(std::vector<PHMSSearch<T>*>& gpu_objs, T* query_array,
                                    int num_queries, result_ann_t* nearestNeighbours,
                                    float* nearestNeighbours_dist,
                                    const std::vector<unsigned>& shard_ends,
                                    const unsigned* query_seeds,
                                    std::vector<std::vector<unsigned>>* debug_tracks)
{
    std::vector<PHMSSearchInner<T>*> impls;
    impls.reserve(gpu_objs.size());
    for (auto* obj : gpu_objs)
        impls.push_back(static_cast<PHMSSearchInner<T>*>(obj->m_pImpl));
    try
    {
        PHMSSearchInner<T>::ngpu_phms_query(impls, query_array, num_queries, nearestNeighbours,
                                            nearestNeighbours_dist, shard_ends, query_seeds,
                                            debug_tracks);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "[PHMS search error] %s\n", e.what());
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> bool PHMSSearchInner<T>::phms_load(char* indexfile_path_prefix)
{

    std::lock_guard<std::mutex> lk(g_phms_index_mtx);

    if (g_phms_index_loaded)
    {
        m_objInputData = g_phms_index_cache;
        g_phms_index_refcnt++;
        return true;
    }

    string diskann_Generatedfiles_path = string(indexfile_path_prefix);
    string pqPivots_file = diskann_Generatedfiles_path + PQ_PIVOTS_FILE_SUFFIX;
    string compressedVector_file = diskann_Generatedfiles_path + PQ_COMPRESSEDVECTORS_FILE_SUFFIX;
    string graphAdjListAndFP_file = diskann_Generatedfiles_path + GRAPH_INDEX_FILE_SUFFIX;
    string graphMetadata_file = diskann_Generatedfiles_path + GRAPH_INDEX_METADATA_FILE_SUFFIX;
    bool bRet = true;

    ifstream in1(pqPivots_file, std::ios::binary);
    if (!in1.is_open())
    {
        printf("Error.. Could not open the PQ Pivots File: %s", pqPivots_file.c_str());
        return false;
    }
    ifstream in2(compressedVector_file, std::ios::binary);
    if (!in2.is_open())
    {
        printf("Error.. Could not open the PQ Compressed Vectors File: %s",
               compressedVector_file.c_str());
        return false;
    }
    ifstream in3(graphAdjListAndFP_file, std::ios::binary);
    if (!in3.is_open())
    {
        printf("Error.. Could not open the Graph Index File: %s", graphAdjListAndFP_file.c_str());
        return false;
    }

    GraphMedataData objGrapMetaData;
    ifstream in4(graphMetadata_file, std::ios::binary);
    if (!in4.is_open())
    {
        printf("Error.. Could not open the Metadata File: %s\n", graphMetadata_file.c_str());
        return false;
    }
    in4.read((char*)&objGrapMetaData, sizeof(GraphMedataData));
    cout << "Metadata : " << objGrapMetaData.ullMedoid << ", " << objGrapMetaData.ulluIndexEntryLen
         << ", " << objGrapMetaData.uDatatype << ", " << objGrapMetaData.uDim << ", "
         << objGrapMetaData.uDegree << ", " << objGrapMetaData.uDatasetSize << endl;

    m_objInputData.MEDOID = objGrapMetaData.ullMedoid;
    m_objInputData.D = objGrapMetaData.uDim;
    m_objInputData.R = objGrapMetaData.uDegree;
    m_objInputData.N = objGrapMetaData.uDatasetSize;
    m_objInputData.ullIndex_Entry_LEN = objGrapMetaData.ulluIndexEntryLen;

    assert(m_objInputData.R == MAX_R);

    float* pqTable = NULL;
    float* pqTable_T = NULL;
    unsigned* chunksOffset = NULL;
    float* centroid = NULL;
    uint8_t* compressedVectors = NULL;
    unsigned int N = 0;
    unsigned int uChunks = 0;
    unsigned *puNeighbour = NULL, *puNeighbour1 = NULL, *puNumNeighbours = NULL,
             *puNumNeighbours1 = NULL;
    unsigned uNumPQSectionOffsets = 0;
    struct PQMetadata
    {
        unsigned long long pqPivots_FileOffset, pqCentroid_FileOffset, pqChunkOffsets_FileOffset,
            pqPivots_FileSize;
    } pqFileOffsets = {0, 0, 0, 0};

    in2.read((char*)&N, sizeof(int));
    cout << "Reading:" << compressedVector_file << endl;
    cout << "No of points in Dataset = " << N << endl;
    in2.read((char*)&uChunks, sizeof(int));
    cout << "# PQ Chunks = " << uChunks << endl;
    m_objInputData.uChunks = uChunks;

    compressedVectors = (uint8_t*)malloc(sizeof(uint8_t) * uChunks * N);
    if (NULL == compressedVectors)
    {
        printf("Error.. Malloc failed for Compressed Vectors\n");
        bRet = false;
        goto cleanup;
    }
    in2.read((char*)compressedVectors, sizeof(uint8_t) * N * uChunks);
    in2.close();
    cout << "Finished reading bin file." << endl;

    printf("Transferring Compressed Vectors to GPU ...\n");
    gpuErrchk(cudaMalloc(&m_objInputData.d_compressedVectors, sizeof(uint8_t) * N * uChunks));
    gpuErrchk(cudaMemcpy(m_objInputData.d_compressedVectors, compressedVectors,
                         (unsigned long long)(sizeof(uint8_t) * (unsigned long long)(uChunks)*N),
                         cudaMemcpyHostToDevice));
    free(compressedVectors);
    compressedVectors = NULL;

    in1.read((char*)&uNumPQSectionOffsets, sizeof(unsigned));
    if (uNumPQSectionOffsets != PQ_PIVOTS_NUM_SECTIONS + 1)
    {
        printf("Error.. PQ Pivots File format error\n");
        return false;
    }
    in1.seekg(8, std::ios::beg);
    in1.read((char*)&pqFileOffsets, sizeof(PQMetadata));

    in1.seekg(pqFileOffsets.pqPivots_FileOffset + 8, std::ios::beg);
    pqTable = (float*)malloc(sizeof(float) * (256 * m_objInputData.D));
    if (NULL == pqTable)
    {
        printf("Error.. Malloc failed PQ Table\n");
        return false;
    }
    in1.read((char*)pqTable, sizeof(float) * 256 * m_objInputData.D);

    pqTable_T = (float*)malloc(sizeof(float) * (256 * m_objInputData.D));
    if (NULL == pqTable_T)
    {
        bRet = false;
        goto cleanup;
    }
    for (unsigned row = 0; row < 256; ++row)
        for (unsigned col = 0; col < m_objInputData.D; ++col)
            pqTable_T[col * 256 + row] = pqTable[row * m_objInputData.D + col];

    in1.seekg(pqFileOffsets.pqCentroid_FileOffset + 8, std::ios::beg);
    centroid = (float*)malloc(sizeof(float) * m_objInputData.D);
    in1.read((char*)centroid, sizeof(float) * m_objInputData.D);

    in1.seekg(pqFileOffsets.pqChunkOffsets_FileOffset + 8, std::ios::beg);
    chunksOffset = (unsigned*)malloc(sizeof(unsigned) * (m_objInputData.uChunks + 1));
    in1.read((char*)chunksOffset, sizeof(unsigned) * (m_objInputData.uChunks + 1));
    in1.close();
    cout << "Finished reading :" << pqPivots_file << endl;

    gpuErrchk(cudaMalloc(&m_objInputData.d_pqTable, sizeof(float) * (256 * m_objInputData.D)));
    gpuErrchk(cudaMalloc(&m_objInputData.d_chunksOffset,
                         sizeof(unsigned) * (m_objInputData.uChunks + 1)));
    gpuErrchk(cudaMalloc(&m_objInputData.d_centroid, sizeof(float) * (m_objInputData.D)));
    gpuErrchk(cudaMemcpy(m_objInputData.d_pqTable, pqTable_T,
                         sizeof(float) * (256 * m_objInputData.D), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_objInputData.d_chunksOffset, chunksOffset,
                         sizeof(unsigned) * (m_objInputData.uChunks + 1), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_objInputData.d_centroid, centroid, sizeof(float) * (m_objInputData.D),
                         cudaMemcpyHostToDevice));

    m_objInputData.size_indexfile = caclulate_filesize(graphAdjListAndFP_file.c_str());

#ifdef _HUGEPAGES
    m_objInputData.pIndex =
        (uint8_t*)mmap(NULL, m_objInputData.size_indexfile, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (m_objInputData.pIndex == MAP_FAILED)
    {
        printf("[Warning] Huge Pages allocation failed. Falling back to malloc.\n");
        m_objInputData.pIndex = (uint8_t*)malloc(m_objInputData.size_indexfile);
    }
#else
    m_objInputData.pIndex = (uint8_t*)malloc(m_objInputData.size_indexfile);
#endif
    if (NULL == m_objInputData.pIndex)
    {
        printf("Error.. Malloc failed for Graph Index.\n");
        gpuErrchk(cudaFree(m_objInputData.d_compressedVectors));
        bRet = false;
        goto cleanup;
    }
    log_message("Index Load Started");
    in3.read((char*)m_objInputData.pIndex, m_objInputData.size_indexfile);
    log_message("Index Load Done");
    in3.close();

    puNumNeighbours = (unsigned*)(m_objInputData.pIndex + ((m_objInputData.ullIndex_Entry_LEN * 0) +
                                                           (sizeof(T) * m_objInputData.D)));
    puNeighbour = puNumNeighbours + 1;
    puNumNeighbours1 = (unsigned*)(m_objInputData.pIndex +
                                   ((m_objInputData.ullIndex_Entry_LEN * (m_objInputData.N - 1)) +
                                    (sizeof(T) * m_objInputData.D)));
    puNeighbour1 = puNumNeighbours1 + (*puNumNeighbours1);
    printf("First Neighbour : %u \t Last Neighbour: %u\n", *puNeighbour, *puNeighbour1);
    assert(*puNeighbour <= m_objInputData.N);
    assert(*puNeighbour1 <= m_objInputData.N);
    cout << "Graph Medoid is : " << m_objInputData.MEDOID << "\t" << endl;
    cout << "BF Size : " << BF_ENTRIES << "\t" << endl;

cleanup:
    free(pqTable);
    pqTable = NULL;
    free(pqTable_T);
    pqTable_T = NULL;
    free(chunksOffset);
    chunksOffset = NULL;
    free(centroid);
    centroid = NULL;

    if (bRet)
    {
        g_phms_index_cache = m_objInputData;
        g_phms_index_loaded = true;
        g_phms_index_refcnt = 1;
        g_phms_index_cache.h_pqTable_hostcopy =
            (float*)malloc(sizeof(float) * 256 * m_objInputData.D);
        memcpy(g_phms_index_cache.h_pqTable_hostcopy, pqTable_T,
               sizeof(float) * 256 * m_objInputData.D);
        g_phms_index_cache.h_centroid_hostcopy = (float*)malloc(sizeof(float) * m_objInputData.D);
        memcpy(g_phms_index_cache.h_centroid_hostcopy, centroid, sizeof(float) * m_objInputData.D);
        g_phms_index_cache.h_chunksOffset_hostcopy =
            (unsigned*)malloc(sizeof(unsigned) * (m_objInputData.uChunks + 1));
        memcpy(g_phms_index_cache.h_chunksOffset_hostcopy, chunksOffset,
               sizeof(unsigned) * (m_objInputData.uChunks + 1));
    }
    return bRet;
}

template <typename T>
bool PHMSSearchInner<T>::phms_load_sharded(char* indexfile_path_prefix, unsigned shard_start,
                                           unsigned shard_end)
{
    std::lock_guard<std::mutex> lk(g_phms_index_mtx);

    if (!g_phms_index_loaded)
    {
        string diskann_Generatedfiles_path = string(indexfile_path_prefix);
        string pqPivots_file = diskann_Generatedfiles_path + PQ_PIVOTS_FILE_SUFFIX;
        string compressedVector_file =
            diskann_Generatedfiles_path + PQ_COMPRESSEDVECTORS_FILE_SUFFIX;
        string graphAdjListAndFP_file = diskann_Generatedfiles_path + GRAPH_INDEX_FILE_SUFFIX;
        string graphMetadata_file = diskann_Generatedfiles_path + GRAPH_INDEX_METADATA_FILE_SUFFIX;

        GraphMedataData objGrapMetaData;
        ifstream in4(graphMetadata_file, std::ios::binary);
        if (!in4.is_open())
        {
            printf("Error.. Could not open metadata file: %s\n", graphMetadata_file.c_str());
            return false;
        }
        in4.read((char*)&objGrapMetaData, sizeof(GraphMedataData));
        m_objInputData.MEDOID = objGrapMetaData.ullMedoid;
        m_objInputData.D = objGrapMetaData.uDim;
        m_objInputData.R = objGrapMetaData.uDegree;
        m_objInputData.N = objGrapMetaData.uDatasetSize;
        m_objInputData.ullIndex_Entry_LEN = objGrapMetaData.ulluIndexEntryLen;
        in4.close();

        ifstream in2(compressedVector_file, std::ios::binary);
        if (!in2.is_open())
        {
            printf("Error.. Could not open compressed vectors file: %s\n",
                   compressedVector_file.c_str());
            return false;
        }
        unsigned int N_file = 0;
        unsigned int uChunks = 0;
        in2.read((char*)&N_file, sizeof(int));
        in2.read((char*)&uChunks, sizeof(int));
        m_objInputData.uChunks = uChunks;
        in2.close();

        ifstream in1(pqPivots_file, std::ios::binary);
        if (!in1.is_open())
        {
            printf("Error.. Could not open PQ Pivots file: %s\n", pqPivots_file.c_str());
            return false;
        }
        unsigned uNumPQSectionOffsets = 0;
        float* pqTable = NULL;
        float* pqTable_T = NULL;
        unsigned* chunksOffset = NULL;
        float* centroid = NULL;

        struct PQMetadata
        {
            unsigned long long a, b, c, d;
        } pqFO = {0, 0, 0, 0};
        in1.read((char*)&uNumPQSectionOffsets, sizeof(unsigned));
        in1.seekg(8, std::ios::beg);
        in1.read((char*)&pqFO, sizeof(PQMetadata));

        in1.seekg(pqFO.a + 8, std::ios::beg);
        pqTable = (float*)malloc(sizeof(float) * 256 * m_objInputData.D);
        in1.read((char*)pqTable, sizeof(float) * 256 * m_objInputData.D);
        pqTable_T = (float*)malloc(sizeof(float) * 256 * m_objInputData.D);
        for (unsigned r = 0; r < 256; ++r)
            for (unsigned c = 0; c < m_objInputData.D; ++c)
                pqTable_T[c * 256 + r] = pqTable[r * m_objInputData.D + c];

        in1.seekg(pqFO.b + 8, std::ios::beg);
        centroid = (float*)malloc(sizeof(float) * m_objInputData.D);
        in1.read((char*)centroid, sizeof(float) * m_objInputData.D);

        in1.seekg(pqFO.c + 8, std::ios::beg);
        chunksOffset = (unsigned*)malloc(sizeof(unsigned) * (m_objInputData.uChunks + 1));
        in1.read((char*)chunksOffset, sizeof(unsigned) * (m_objInputData.uChunks + 1));
        in1.close();

        ifstream in3(graphAdjListAndFP_file, std::ios::binary);
        if (!in3.is_open())
        {
            printf("Error.. Could not open graph index file: %s\n", graphAdjListAndFP_file.c_str());
            return false;
        }
        m_objInputData.size_indexfile = caclulate_filesize(graphAdjListAndFP_file.c_str());
#ifdef PHMS_INDEX_LAYOUT_EXPERIMENT
        const bool use_thp =
            std::getenv("PHMS_INDEX_THP") && std::atoi(std::getenv("PHMS_INDEX_THP"));
        if (use_thp)
            m_objInputData.pIndex = phms_thp_index_alloc(m_objInputData.size_indexfile);
        else
        {
#endif
#ifdef _HUGEPAGES
            m_objInputData.pIndex =
                (uint8_t*)mmap(NULL, m_objInputData.size_indexfile, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
            if (m_objInputData.pIndex == MAP_FAILED)
            {
                printf("[Warning] Huge Pages failed in sharded load, use malloc.\n");
                m_objInputData.pIndex = (uint8_t*)malloc(m_objInputData.size_indexfile);
            }
#else
        m_objInputData.pIndex = (uint8_t*)malloc(m_objInputData.size_indexfile);
#endif
#ifdef PHMS_INDEX_LAYOUT_EXPERIMENT
        }
#endif
        in3.read((char*)m_objInputData.pIndex, m_objInputData.size_indexfile);
        in3.close();

        if (phms_compact_upload_enabled())
        {
            const char* path = std::getenv("PHMS_SHARD_ADJ_FILE");
            if (!path || !*path)
                throw std::runtime_error("PHMS_SHARD_ADJ_FILE is required");
            printf("[PHMS OFFLINE ADJ] loading %s (outside query timing)\n", path);
            g_shard_adjacency = std::make_shared<PhmsShardAdjacency>(path, graphAdjListAndFP_file);
            const auto& h = g_shard_adjacency->header;
            if (h.nodes != m_objInputData.N || h.degree != m_objInputData.R ||
                h.dimension != m_objInputData.D || h.element_bytes != sizeof(T) ||
                h.entry_bytes != m_objInputData.ullIndex_Entry_LEN)
                throw std::runtime_error("offline adjacency index metadata mismatch");
            printf("[PHMS OFFLINE ADJ] resident shared CPU mapping: %llu bytes\n",
                   (unsigned long long)h.file_bytes);
        }

        unsigned* puNumNeighbours =
            (unsigned*)(m_objInputData.pIndex +
                        ((m_objInputData.ullIndex_Entry_LEN * 0) + (sizeof(T) * m_objInputData.D)));
        unsigned* puNeighbour = puNumNeighbours + 1;
        assert(*puNeighbour <= m_objInputData.N);

        m_objInputData.h_pqTable_hostcopy = (float*)malloc(sizeof(float) * 256 * m_objInputData.D);
        memcpy(m_objInputData.h_pqTable_hostcopy, pqTable_T,
               sizeof(float) * 256 * m_objInputData.D);
        m_objInputData.h_centroid_hostcopy = (float*)malloc(sizeof(float) * m_objInputData.D);
        memcpy(m_objInputData.h_centroid_hostcopy, centroid, sizeof(float) * m_objInputData.D);
        m_objInputData.h_chunksOffset_hostcopy =
            (unsigned*)malloc(sizeof(unsigned) * (m_objInputData.uChunks + 1));
        memcpy(m_objInputData.h_chunksOffset_hostcopy, chunksOffset,
               sizeof(unsigned) * (m_objInputData.uChunks + 1));

        m_objInputData.d_pqTable = nullptr;
        m_objInputData.d_chunksOffset = nullptr;
        m_objInputData.d_centroid = nullptr;
        m_objInputData.d_compressedVectors = nullptr;
        g_phms_index_cache = m_objInputData;
        g_phms_index_loaded = true;
        g_phms_index_refcnt = 1;
        free(pqTable);
        free(pqTable_T);
        free(chunksOffset);
        free(centroid);
    }
    else
    {
        m_objInputData = g_phms_index_cache;
        g_phms_index_refcnt++;
    }

    int dev_id;
    cudaGetDevice(&dev_id);
    {
        std::lock_guard<std::mutex> lock(g_per_gpu_mtx);
        auto& cache = g_per_gpu_cache[dev_id];
        if (cache.refcnt == 0)
        {
            gpuErrchk(cudaMalloc(&cache.d_pqTable, sizeof(float) * 256 * m_objInputData.D));
            gpuErrchk(
                cudaMalloc(&cache.d_chunksOffset, sizeof(unsigned) * (m_objInputData.uChunks + 1)));
            gpuErrchk(cudaMalloc(&cache.d_centroid, sizeof(float) * m_objInputData.D));
            gpuErrchk(cudaMemcpy(cache.d_pqTable, g_phms_index_cache.h_pqTable_hostcopy,
                                 sizeof(float) * 256 * m_objInputData.D, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(cache.d_chunksOffset, g_phms_index_cache.h_chunksOffset_hostcopy,
                                 sizeof(unsigned) * (m_objInputData.uChunks + 1),
                                 cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(cache.d_centroid, g_phms_index_cache.h_centroid_hostcopy,
                                 sizeof(float) * m_objInputData.D, cudaMemcpyHostToDevice));

            const char* layout = std::getenv("PHMS_GRAPHCUT_LAYOUT");
            if (layout && *layout)
            {
                if (phms_compact_upload_enabled())
                    throw std::runtime_error("GraphCut cannot use ID-shard compact adjacency");
                const std::string file =
                    std::string(layout) + "/pq" + std::to_string(dev_id) + ".bin";
                std::ifstream f(file, std::ios::binary | std::ios::ate);
                if (!f)
                    throw std::runtime_error("cannot open " + file);
                const auto bytes = uint64_t(f.tellg());
                f.seekg(0);
                PhmsGcHeader h{};
                f.read((char*)&h, 40);
                phms_gc_validate(h, bytes, dev_id, m_objInputData.N, m_objInputData.uChunks);
                std::vector<uint64_t> bits(h.words);
                std::vector<unsigned> prefix(h.words);
                f.read((char*)bits.data(), h.words * 8);
                f.read((char*)prefix.data(), h.words * 4);
                uint64_t count = 0;
                for (uint64_t w = 0; w < h.words; ++w)
                {
                    if (prefix[w] != count)
                        throw std::runtime_error("bad GraphCut prefix");
                    count += __builtin_popcountll(bits[w]);
                }
                if (count != h.local_nodes)
                    throw std::runtime_error("bad GraphCut cardinality");
                if (h.nodes % 64 && (bits.back() >> (h.nodes % 64)))
                    throw std::runtime_error("GraphCut padding bits set");
                std::vector<uint8_t> codes(h.local_nodes * h.chunks);
                f.read((char*)codes.data(), codes.size());
                if (!f)
                    throw std::runtime_error("truncated GraphCut PQ");
                gpuErrchk(cudaMalloc(&cache.gc_bits, h.words * 8));
                gpuErrchk(cudaMalloc(&cache.gc_prefix, h.words * 4));
                gpuErrchk(cudaMalloc(&cache.d_compressedVectors, codes.size()));
                gpuErrchk(
                    cudaMemcpy(cache.gc_bits, bits.data(), h.words * 8, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(cache.gc_prefix, prefix.data(), h.words * 4,
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(cache.d_compressedVectors, codes.data(), codes.size(),
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpyToSymbol(phms_gc_bits, &cache.gc_bits, sizeof(cache.gc_bits)));
                gpuErrchk(
                    cudaMemcpyToSymbol(phms_gc_prefix, &cache.gc_prefix, sizeof(cache.gc_prefix)));
                gpuErrchk(cudaMemcpyToSymbol(phms_gc_nodes, &h.nodes, sizeof(h.nodes)));
                printf("[V4 GraphCut] GPU %d nodes=%llu PQ_bytes=%llu rank_bytes=%llu\n", dev_id,
                       (unsigned long long)h.local_nodes, (unsigned long long)codes.size(),
                       (unsigned long long)h.words * 12);
            }
            else
            {
                string cv_file = string(indexfile_path_prefix) + PQ_COMPRESSEDVECTORS_FILE_SUFFIX;
                ifstream in2(cv_file, std::ios::binary);
                if (!in2.is_open())
                {
                    printf("Error open %s\n", cv_file.c_str());
                    return false;
                }
                unsigned int Nf = 0;
                unsigned int uCh = 0;
                in2.read((char*)&Nf, sizeof(int));
                in2.read((char*)&uCh, sizeof(int));
                unsigned sn = shard_end - shard_start;
                uint64_t sb = (uint64_t)sn * (uint64_t)uCh;
                uint64_t so = (uint64_t)shard_start * (uint64_t)uCh;
                uint8_t* shard = (uint8_t*)malloc(sb);
                in2.seekg(8 + so, std::ios::beg);
                in2.read((char*)shard, sb);
                in2.close();
                gpuErrchk(cudaMalloc(&cache.d_compressedVectors, sb));
                gpuErrchk(cudaMemcpy(cache.d_compressedVectors, shard, sb, cudaMemcpyHostToDevice));
                free(shard);
                printf("[NGPU] GPU %d cache: shard [%u,%u) (%llu bytes)\n", dev_id, shard_start,
                       shard_end, (unsigned long long)sb);
            }
        }

        m_objInputData.d_pqTable = cache.d_pqTable;
        m_objInputData.d_chunksOffset = cache.d_chunksOffset;
        m_objInputData.d_centroid = cache.d_centroid;
        m_objInputData.d_compressedVectors = cache.d_compressedVectors;
        cache.refcnt++;
    }

    m_objInputData.uShardStart = shard_start;
    m_objInputData.uShardEnd = shard_end;
    m_objGPUInst.physical_device = dev_id;
    m_objGPUInst.uShardStart = shard_start;
    m_objGPUInst.uShardEnd = shard_end;

    if (phms_compact_upload_enabled())
    {
        bool match = false;
        const auto& h = g_shard_adjacency->header;
        for (unsigned g = 0; g < h.shards; ++g)
            match |= shard_start == (g ? h.ends[g - 1] : 0) && shard_end == h.ends[g];
        if (!match)
            throw std::runtime_error("offline adjacency PQ shard boundary mismatch");
    }

    return true;
}

template <typename T>
void PHMSSearchInner<T>::ngpu_phms_query(std::vector<PHMSSearchInner<T>*>& gpu_impls, T* queriesFP,
                                         int numQueries, result_ann_t* nearestNeighbours,
                                         float* nearestNeighbours_dist,
                                         const std::vector<unsigned>& shard_ends,
                                         const unsigned* query_seeds,
                                         std::vector<std::vector<unsigned>>* debug_tracks)
{
    const int num_gpus = (int)gpu_impls.size();
    if (num_gpus < 1)
        return;
    auto wait_event = [&](cudaEvent_t event) { gpuErrchk(cudaEventSynchronize(event)); };
    auto wait_stream = [&](cudaStream_t stream) { gpuErrchk(cudaStreamSynchronize(stream)); };
    const int submit_shift =
        phms_v3::rotate_submit()
            ? (int)(g_ngpu_submit_sequence.fetch_add(1, std::memory_order_relaxed) %
                    (unsigned)num_gpus)
            : 0;
    bool profile_phases = false;
    if (const char* profile_env = std::getenv("PHMS_PROFILE_PHASES"))
        profile_phases = (std::atoi(profile_env) != 0);
    const unsigned profile_id =
        profile_phases ? g_ngpu_profile_sequence.fetch_add(1, std::memory_order_relaxed) : 0;
    bool defer_round_sync = true;
    if (const char* defer_env = std::getenv("PHMS_DEFER_ROUND_SYNC"))
        defer_round_sync = (std::atoi(defer_env) != 0);
    bool fuse_filter_distance = true;
    if (const char* fuse_env = std::getenv("PHMS_FUSE_FILTER_DISTANCE"))
        fuse_filter_distance = (std::atoi(fuse_env) != 0);
    bool fuse_expansion = false;
    if (const char* expansion_env = std::getenv("PHMS_FUSE_EXPANSION"))
        fuse_expansion = (std::atoi(expansion_env) != 0);
    fuse_expansion = fuse_expansion && fuse_filter_distance;
    const unsigned expansion_threads =
        phms_ngpu_launch_threads("PHMS_NGPU_EXPANSION_THREADS", 128, 32, 512, 32);
    bool graph_parent_node = true;
    if (const char* parent_env = std::getenv("PHMS_GRAPH_PARENT_NODE"))
        graph_parent_node = (std::atoi(parent_env) != 0);
    bool pack_expansion_upload = true;
    if (const char* pack_env = std::getenv("PHMS_PACK_EXPANSION_UPLOAD"))
        pack_expansion_upload = (std::atoi(pack_env) != 0);
    const unsigned cpu_query_block = phms_ngpu_launch_threads("PHMS_CPU_QBLK", 64, 8, 64, 8);
    const bool compact_upload = phms_compact_upload_enabled();
    const bool tile_pack = compact_upload && phms_compact_tile_enabled();
    const bool warp_sort = phms_v3::warp_sort_enabled();
    const unsigned parent_pipeline = phms_v3::parent_pipeline();
    if (parent_pipeline && (!defer_round_sync || NGPU_PARENTS_PER_ROUND != 2))
        throw std::runtime_error(
            "v3 parent handoff requires deferred sync and the original two parents");
    bool pipeline_parent_prep = false;
    if (const char* value = std::getenv("PHMS_PIPELINE_PARENT_PREP"))
        pipeline_parent_prep = std::atoi(value) != 0;
    if (pipeline_parent_prep && compact_upload)
        throw std::runtime_error(
            "parent preparation pipeline requires the original broadcast format");
    bool overlap_cpu_rerank = false;
    if (const char* value = std::getenv("PHMS_OVERLAP_CPU_RERANK"))
        overlap_cpu_rerank = std::atoi(value) != 0;
    bool return_tracks_only = false;
#ifdef PHMS_INDEPENDENT_EXPERIMENT
    return_tracks_only =
        std::getenv("PHMS_INDEPENDENT_SEARCH") && std::atoi(std::getenv("PHMS_INDEPENDENT_SEARCH"));
    if (return_tracks_only)
    {
        if (num_gpus != 1 || !debug_tracks)
            throw std::runtime_error("independent search requires one GPU and returned tracks");
        overlap_cpu_rerank = false;
    }
#endif
    const unsigned parent_prefetch_lines =
        phms_ngpu_launch_threads("PHMS_PARENT_PREFETCH_LINES", 0, 0, 5, 1);
    const unsigned graph_lookahead = phms_ngpu_launch_threads("PHMS_GRAPH_LOOKAHEAD", 2, 1, 64, 1);
    bool device_local_submit = false;
    if (const char* submit_env = std::getenv("PHMS_DEVICE_LOCAL_SUBMIT"))
        device_local_submit = (std::atoi(submit_env) != 0);
    const unsigned parallel_submit_min =
        phms_ngpu_launch_threads("PHMS_PARALLEL_GPU_MIN_CARDS", 3, 2, 8, 1);
    bool parallel_gpu_submit = std::getenv("PHMS_PARALLEL_GPU_SUBMIT") &&
                               std::atoi(std::getenv("PHMS_PARALLEL_GPU_SUBMIT")) &&
                               num_gpus >= parallel_submit_min;
    const bool combined_round_graph = std::getenv("PHMS_COMBINED_ROUND_GRAPH") &&
                                      std::atoi(std::getenv("PHMS_COMBINED_ROUND_GRAPH"));
    if (combined_round_graph && (pipeline_parent_prep || parallel_gpu_submit || compact_upload ||
                                 NGPU_PARENTS_PER_ROUND != 2))
        throw std::runtime_error(
            "combined round graph requires ordinary two-parent broadcast, serial submission");
    if (parallel_gpu_submit &&
        (pipeline_parent_prep || compact_upload || NGPU_PARENTS_PER_ROUND != 2))
        throw std::runtime_error("parallel GPU submission requires ordinary two-parent broadcast, "
                                 "no parent preparation pipeline");
    const auto profile_total_start = steady_clock::now();
    PhmsTraceRange core_trace("Init");
    double profile_init_ms = 0.0;
    double profile_parent_ms = 0.0;
    double profile_cpu_ms = 0.0;
    double profile_gpu_ms = 0.0;
    double profile_rerank_ms = 0.0;
    double profile_compact_pack_ms = 0.0;
    double profile_early_submit_ms = 0.0;
    double profile_overlap_rerank_ms = 0.0;
    unsigned pipelined_rounds = 0;
    uint64_t upload_bytes = 0;
    uint64_t legacy_upload_bytes = 0;
    bool report_upload = profile_phases;
    if (const char* value = std::getenv("PHMS_REPORT_UPLOAD"))
        report_upload |= std::atoi(value) != 0;
    bool profile_detail = false;
    if (const char* detail_env = std::getenv("PHMS_PROFILE_DETAIL"))
        profile_detail = profile_phases && std::atoi(detail_env) != 0;
    double profile_parent_submit_ms = 0.0;
    double profile_parent_wait_ms = 0.0;
    double profile_select_work_ms = 0.0;
    double profile_fetch_work_ms = 0.0;
    double profile_select_max_ms = 0.0;
    double profile_fetch_max_ms = 0.0;
    std::vector<double> profile_wait_by_gpu(num_gpus, 0.0);
    auto profile_elapsed_ms = [](steady_clock::time_point begin, steady_clock::time_point end)
    { return duration<double, std::milli>(end - begin).count(); };
    if ((int)shard_ends.size() != num_gpus)
    {
        fprintf(stderr, "[PHMS NGPU] shard_ends size (%zu) does not match GPU count (%d)\n",
                shard_ends.size(), num_gpus);
        return;
    }
    for (int g = 0; g < num_gpus; ++g)
    {
        if (!std::getenv("PHMS_GRAPHCUT_LAYOUT") &&
            (shard_ends[g] == 0 || (g > 0 && shard_ends[g] <= shard_ends[g - 1])))
        {
            fprintf(stderr, "[PHMS NGPU] invalid shard boundary at GPU %d\n", g);
            return;
        }
    }

    auto start1 = high_resolution_clock::now();

    IndexLoad& idx0 = gpu_impls[0]->m_objInputData;
    SearchParams& params = gpu_impls[0]->m_objSearchParams;
    HostInstance& h0 = gpu_impls[0]->m_objHostInst;
    std::vector<double> detail_select(profile_detail ? h0.numCPUthreads : 0, 0.0);
    std::vector<double> detail_fetch(profile_detail ? h0.numCPUthreads : 0, 0.0);

    const unsigned R = idx0.R;
    const unsigned D = (unsigned)idx0.D;
    const auto* resident_impl = gpu_impls[0];
    const bool resident_host =
        resident_impl->m_objGPUInst.resident_selection_count == (unsigned)numQueries &&
        resident_impl->m_resident_host_source && resident_impl->m_resident_host_ids;
    auto query_vector = [&](unsigned q) -> const T*
    {
        return resident_host ? resident_impl->m_resident_host_source +
                                   (size_t)resident_impl->m_resident_host_ids[q] * D
                             : queriesFP + (size_t)q * D;
    };
    const unsigned uChunks = idx0.uChunks;
    const unsigned uWLLen = params.worklist_length;
    if (compact_upload)
    {
        if (!g_shard_adjacency || g_shard_adjacency->header.shards != (unsigned)num_gpus)
            throw std::runtime_error("compact upload GPU count does not match offline adjacency");
        for (int g = 0; g < num_gpus; ++g)
            if (shard_ends[g] != g_shard_adjacency->header.ends[g])
                throw std::runtime_error("compact upload has mismatched PQ boundaries");
    }
    const unsigned uMaxNeighbors = R + 1;
    const unsigned long long MEDOID = idx0.MEDOID;

    const unsigned uMaxParentsPerQuery =
        gpu_impls[0]->m_expansion_budget_override
            ? phms_v3::checked_budget(gpu_impls[0]->m_expansion_budget_override, uWLLen)
            : phms_v3::expansion_budget(uWLLen);
    for (auto* impl : gpu_impls)
    {
        unsigned budget = impl->m_expansion_budget_override
                              ? phms_v3::checked_budget(impl->m_expansion_budget_override, uWLLen)
                              : phms_v3::expansion_budget(uWLLen);
        if (budget != uMaxParentsPerQuery)
            throw std::runtime_error("inconsistent per-GPU budget within cooperative job");
    }
    const unsigned final_candidates = phms_v3::final_candidates();
    const bool skip_unused_tail =
        std::getenv("PHMS_SKIP_UNUSED_TAIL") && std::atoi(std::getenv("PHMS_SKIP_UNUSED_TAIL"));
    if (skip_unused_tail &&
        (final_candidates || !defer_round_sync || compact_upload || pipeline_parent_prep ||
         combined_round_graph || parallel_gpu_submit || NGPU_PARENTS_PER_ROUND != 2))
        throw std::runtime_error("unused tail skip requires ordinary deferred two-parent search, "
                                 "no tail candidate recovery");
    const unsigned track_capacity = uMaxParentsPerQuery + final_candidates * num_gpus;
    const int parents_per_round = NGPU_PARENTS_PER_ROUND;
    printf("[PHMS V3 BEAM] parents_per_round=%d BF_entries=%u\n", parents_per_round,
           (unsigned)BF_ENTRIES);
    if (parents_per_round > 2 && compact_upload)
        throw std::runtime_error("experimental beam4 currently requires broadcast upload");

    const size_t track_words = (size_t)track_capacity * numQueries;
    const size_t count_words = (size_t)numQueries;
    std::vector<float> cached_fp_distances(overlap_cpu_rerank ? track_words : 0);
    std::vector<unsigned> cached_fp_counts(overlap_cpu_rerank ? numQueries : 0, 0);
    const size_t mark_words = (size_t)parents_per_round * numQueries;
    const size_t required_bytes = (track_words + count_words + mark_words) * sizeof(unsigned) +
                                  (size_t)num_gpus * sizeof(bool);
    const size_t pinned_workspace_bytes =
        (size_t)uMaxParentsPerQuery * h0.FPSetCoords_rowsize_bytes;
    if (required_bytes > pinned_workspace_bytes)
    {
        fprintf(stderr,
                "[PHMS NGPU] pinned workspace is too small "
                "(%zu required, %zu available)\n",
                required_bytes, pinned_workspace_bytes);
        return;
    }

    unsigned* workspace = static_cast<unsigned*>(h0.FPSetCoordsList);
    unsigned* h_global_parent_track = workspace;
    unsigned* h_global_parent_counts = h_global_parent_track + track_words;
    unsigned* marks_base = h_global_parent_counts + count_words;
    bool* nextIter_local = reinterpret_cast<bool*>(marks_base + mark_words);
    memset(h_global_parent_track, 0, track_words * sizeof(unsigned));
    memset(h_global_parent_counts, 0, count_words * sizeof(unsigned));

    std::vector<unsigned*> h_global_marks(parents_per_round, nullptr);
    for (int p = 0; p < parents_per_round; ++p)
        h_global_marks[p] = marks_base + (size_t)p * numQueries;

    for (int q = 0; q < numQueries; q++)
    {
        h_global_parent_track[0 * numQueries + q] = query_seeds ? query_seeds[q] : MEDOID;
        h_global_parent_counts[q] = 1;
    }

    std::vector<unsigned*> h_parents(num_gpus, NULL);
    for (int g = 0; g < num_gpus; ++g)
    {
        h_parents[g] = gpu_impls[g]->m_objHostInst.parents;
    }
    auto queue_parent_handoff = [&](int g)
    {
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        gpuErrchk(cudaStreamWaitEvent(gi.streamParent, gi.eventKernelsReady, 0));
        gpuErrchk(cudaMemcpyAsync(h_parents[g], gi.d_ngpu_parents,
                                  (size_t)numQueries * NGPU_PARENT_RECORD_SIZE * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost, gi.streamParent));
    };
    auto queue_next_sort = [&](int g)
    {
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        if (warp_sort)
            neighbor_sort_warp_v3<<<(numQueries + 3) / 4, 128, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query, gi.d_nextIter, R,
                numQueries);
        else
            compute_BestLSets_par_sort_msort<<<numQueries, gi.numThreads_K3, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_neighbors_aux, gi.d_numNeighbors_query,
                gi.d_neighborsDist_query, gi.d_neighborsDist_query_aux, gi.d_nextIter, R);
        gpuErrchk(cudaPeekAtLastError());
    };
    auto queue_direct_parent_tail = [&](int g)
    {
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        if (parent_pipeline == 5)
        {
            gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
            queue_parent_handoff(g);
            gpuErrchk(cudaEventRecord(gi.eventParentCopyDone, gi.streamParent));
            queue_next_sort(g);
            gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, gi.eventParentCopyDone, 0));
            return;
        }
        gpuErrchk(cudaMemcpyAsync(h_parents[g], gi.d_ngpu_parents,
                                  (size_t)numQueries * NGPU_PARENT_RECORD_SIZE * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost, gi.streamKernels));
        queue_next_sort(g);
        if (parent_pipeline == 4)
            gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
    };

    std::vector<HostInstance*> expansion_hosts(parents_per_round, &h0);
    for (int p = 1; p < parents_per_round; ++p)
        if (p < num_gpus)
            expansion_hosts[p] = &gpu_impls[p]->m_objHostInst;
    std::vector<unsigned*> h_expansion_counts(parents_per_round, nullptr);
    std::vector<unsigned*> h_expansion_neighbors(parents_per_round, nullptr);
    for (int p = 0; p < parents_per_round; ++p)
    {
        HostInstance& hp = *expansion_hosts[p];
        if (p >= num_gpus)
        {
            if (p != 1 || !hp.expansion_upload_alt)
                throw std::runtime_error("missing independent Parent host buffer");
            if (pack_expansion_upload && !compact_upload)
                h_global_marks[p] = hp.expansion_upload_alt;
            h_expansion_counts[p] = hp.expansion_upload_alt + numQueries;
            h_expansion_neighbors[p] = hp.expansion_upload_alt + 2 * (size_t)numQueries;
            continue;
        }
        if (pack_expansion_upload && !compact_upload)
        {
            h_global_marks[p] = hp.expansion_upload;
            h_expansion_counts[p] = hp.expansion_upload + numQueries;
            h_expansion_neighbors[p] = hp.expansion_upload + 2 * (size_t)numQueries;
        }
        else
        {
            h_expansion_counts[p] = hp.numNeighbors_query;
            h_expansion_neighbors[p] = hp.neighbors;
        }
    }
    std::vector<const uint32_t*> compact_records(
        compact_upload ? (size_t)parents_per_round * numQueries : 0);
    std::vector<uint8_t*> deferred_parent_rows(pipeline_parent_prep ? numQueries : 0, nullptr);
    std::vector<unsigned*> compact_buffers(compact_upload ? parents_per_round * num_gpus : 0);
    std::vector<size_t> compact_words(compact_buffers.size());
    std::vector<unsigned> compact_counts(
        compact_upload ? parents_per_round * num_gpus * h0.numCPUthreads : 0);
    std::vector<unsigned> compact_bases(compact_counts.size());
    std::vector<unsigned> compact_overflow(compact_buffers.size());
    const unsigned compact_tile_rows = std::min<unsigned>(cpu_query_block, numQueries);
    const unsigned compact_tile_quota = (R + num_gpus - 1) / num_gpus;
    const size_t compact_fixed_payload =
        tile_pack ? phms_compact_fixed_payload(numQueries, R, num_gpus) : 0;
    uint64_t compact_overflow_bytes = 0;
    if (compact_upload)
        for (int p = 0; p < parents_per_round; ++p)
            for (int g = 0; g < num_gpus; ++g)
            {
                auto& host = gpu_impls[g]->m_objHostInst;
                compact_buffers[p * num_gpus + g] =
                    p ? host.expansion_upload_alt : host.expansion_upload;
            }

    for (int g = 0; g < num_gpus; ++g)
    {
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        gi.numThreads_K4 = 128;
        gi.numThreads_K1 = 256;
        gi.numThreads_K2 = phms_ngpu_launch_threads("PHMS_NGPU_DIST_THREADS", 512, 8, 512, 8);
        gi.numThreads_K3 = R + 1;
        gi.K4_blockSize = 256;
        gi.numThreads_K5 = phms_ngpu_launch_threads("PHMS_NGPU_FILTER_THREADS", 256, 32, 1024, 32);
        gi.numThreads_K3_merge = 2 * uWLLen;
        assert(gi.numThreads_K3_merge <= 1024);
    }

#ifdef _TIMERS
    double time_K1 = 0.0;
    double time_B1 = 0.0;
    double time_B2 = 0.0;
    double time_neighbor_filtering = 0.0;
    double time_prfetch = 0.0;
    double time_reranking = 0.0;
    double time_transfer = 0.0;
    double seek_neighbours_time = 0.0;
    double time_convergence = 0.0;
    double time_cuda = 0.0;
    double time_1 = 0.0;
    double time_2 = 0.0;
    double time_3 = 0.0;
    double time_4 = 0.0;
    double time_5 = 0.0;
    high_resolution_clock::time_point t_start, t_end;
    high_resolution_clock::time_point cuda_start, cuda_end;
#endif

#ifdef _TIMERS
#endif
    for (int visit = 0; visit < num_gpus; ++visit)
    {
        const int g = (visit + submit_shift) % num_gpus;
        cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        if (gi.resident_selection_count)
        {
            if (gi.resident_selection_count != (unsigned)numQueries)
                throw std::runtime_error("V4 resident selection size mismatch");
            gi.resident_selection_count = 0;
        }
        else
            gpuErrchk(cudaMemcpyAsync(gi.d_queriesFP, queriesFP, sizeof(T) * D * numQueries,
                                      cudaMemcpyHostToDevice, gi.streamKernels));
        gpuErrchk(cudaMemsetAsync(gi.d_pqDistTables, 0, sizeof(float) * uChunks * 256 * numQueries,
                                  gi.streamKernels));
        gpuErrchk(cudaMemsetAsync(gi.d_processed_bit_vec, 0, sizeof(bool) * BF_MEMORY * numQueries,
                                  gi.streamKernels));
        gpuErrchk(cudaMemsetAsync(gi.d_BestLSets_count, 0, sizeof(unsigned) * numQueries,
                                  gi.streamKernels));
    }

    unsigned iter = 1;
    for (unsigned q = 0; q < (unsigned)numQueries; ++q)
    {
        const unsigned seed = query_seeds ? query_seeds[q] : (unsigned)MEDOID;
        if (seed >= idx0.N)
            throw std::runtime_error("invalid query entry");
        const unsigned* row =
            (const unsigned*)(idx0.pIndex + idx0.ullIndex_Entry_LEN * seed + D * sizeof(T));
        if (row[0] > R)
            throw std::runtime_error("invalid graph degree");
        unsigned* dst = h0.neighbors + (size_t)q * uMaxNeighbors;
        dst[0] = seed;
        memcpy(dst + 1, row + 1, row[0] * sizeof(unsigned));
        h0.numNeighbors_query[q] = row[0] + 1;
    }
    for (int g = 0; g < num_gpus; ++g)
    {
        auto& gi = gpu_impls[g]->m_objGPUInst;
        gpuErrchk(cudaSetDevice(gi.physical_device));
        gpuErrchk(cudaMemcpyAsync(gi.d_query_seeds, h_global_parent_track,
                                  (size_t)numQueries * sizeof(unsigned), cudaMemcpyHostToDevice,
                                  gi.streamKernels));
    }

    for (int visit = 0; visit < num_gpus; ++visit)
    {
        const int g = (visit + submit_shift) % num_gpus;
        cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        gpuErrchk(cudaMemcpyAsync(gi.d_neighbors_temp, h0.neighbors,
                                  sizeof(unsigned) * numQueries * uMaxNeighbors,
                                  cudaMemcpyHostToDevice, gi.streamChildren));
        gpuErrchk(cudaMemcpyAsync(gi.d_numNeighbors_query_temp, h0.numNeighbors_query,
                                  sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice,
                                  gi.streamChildren));
        gpuErrchk(cudaEventRecord(gi.eventChildrenReady, gi.streamChildren));
    }

    auto end1 = high_resolution_clock::now();
    auto dur_ms1 = duration_cast<milliseconds>(end1 - start1);

    for (int visit = 0; visit < num_gpus; ++visit)
    {
        const int g = (visit + submit_shift) % num_gpus;
        cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
        GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
        IndexLoad& idx = gpu_impls[g]->m_objInputData;
        gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, gi.eventChildrenReady, 0));

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        populate_pqDist_par<<<numQueries, gi.numThreads_K1, (D * (sizeof(float) + sizeof(T))),
                              gi.streamKernels>>>(
            idx.d_pqTable, gi.d_pqDistTables, (T*)gi.d_queriesFP, idx.d_chunksOffset,
            idx.d_centroid, uChunks, D, params.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0);
#ifdef _TIMERS
        cudaStreamSynchronize(gi.streamKernels);
        t_end = high_resolution_clock::now();
        time_K1 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        if (!fuse_filter_distance)
            gpuErrchk(cudaMemsetAsync(gi.d_numNeighbors_query, 0, sizeof(unsigned) * numQueries,
                                      gi.streamKernels));
        if (fuse_filter_distance)
        {
            neighbor_filtering_distance_fused<<<numQueries, gi.numThreads_K2, 0,
                                                gi.streamKernels>>>(
                gi.d_neighbors, gi.d_neighbors_temp, gi.d_numNeighbors_query,
                gi.d_numNeighbors_query_temp, gi.d_processed_bit_vec, idx.d_compressedVectors,
                gi.d_pqDistTables, gi.d_neighborsDist_query, uChunks, R, gi.uShardStart,
                gi.uShardEnd);
        }
        else
        {
            neighbor_filtering_new<<<numQueries, gi.numThreads_K5, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_neighbors_temp, gi.d_numNeighbors_query,
                gi.d_numNeighbors_query_temp, gi.d_processed_bit_vec, R, gi.uShardStart,
                gi.uShardEnd);
        }
#ifdef _TIMERS
        cudaStreamSynchronize(gi.streamKernels);
        t_end = high_resolution_clock::now();
        time_neighbor_filtering += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        if (!fuse_filter_distance)
            compute_neighborDist_par<<<numQueries, gi.numThreads_K2, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_numNeighbors_query, idx.d_compressedVectors, gi.d_pqDistTables,
                gi.d_neighborsDist_query, uChunks, R, gi.uShardStart);
#ifdef _TIMERS
        cudaStreamSynchronize(gi.streamKernels);
        t_end = high_resolution_clock::now();
        time_B1 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        compute_parent1_ngpu2<<<(numQueries + gi.numThreads_K4 - 1) / gi.numThreads_K4,
                                gi.numThreads_K4, 0, gi.streamKernels>>>(
            gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query, gi.d_ngpu_parents,
            gi.d_nextIter, numQueries, MEDOID, R, gi.d_query_seeds);
        if (defer_round_sync && parent_pipeline < 3)
            gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
        if (parent_pipeline == 1 || parent_pipeline == 2)
            queue_parent_handoff(g);
        if (parent_pipeline == 2)
            queue_next_sort(g);
        if (parent_pipeline >= 3)
            queue_direct_parent_tail(g);
#ifdef _TIMERS
        cudaStreamSynchronize(gi.streamKernels);
        t_end = high_resolution_clock::now();
        time_prfetch += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif
    }

    if (!defer_round_sync)
    {
        for (int g = 0; g < num_gpus; ++g)
        {
            cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
            cudaStreamSynchronize(gpu_impls[g]->m_objGPUInst.streamKernels);
        }
    }

    bool nextIter = true;
    unsigned coordination_rounds = 0;

    bool use_cuda_graph = true;
    if (const char* graph_env = std::getenv("PHMS_USE_CUDA_GRAPH"))
        use_cuda_graph = (std::atoi(graph_env) != 0);

    if (use_cuda_graph)
    {
        for (int g = 0; g < num_gpus; ++g)
        {
            GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
            if (gi.ngpuExpansionGraphs[NGPU_EXPANSION_GRAPH_BUFFER0_WITH_SORT] == nullptr ||
                gi.ngpuExpansionGraphs[NGPU_EXPANSION_GRAPH_BUFFER1_NO_SORT] == nullptr)
            {
                use_cuda_graph = false;
                break;
            }
        }
    }
    if (use_cuda_graph)
    {
        printf("[PHMS CUDA Graph] enabled: %d GPU x 2 steady expansion graphs; "
               "prepared outside timed search\n",
               num_gpus);
    }
    else
    {
        printf("[PHMS CUDA Graph] disabled or graphs were not prepared\n");
    }

    end1 = high_resolution_clock::now();
    dur_ms1 = duration_cast<milliseconds>(end1 - start1);
    if (profile_phases)
        profile_init_ms = profile_elapsed_ms(profile_total_start, steady_clock::now());
#ifdef _TIMERS
    cout << "dataPre " << dur_ms1.count() << " 毫秒\n";
#endif

    do
    {
        core_trace.set("Parent_ready_wait");
        ++coordination_rounds;
        const int expansions_this_round =
            std::min(parents_per_round, (int)(uMaxParentsPerQuery - iter));
        const bool skip_tail_round =
            skip_unused_tail && iter + expansions_this_round >= uMaxParentsPerQuery;
        const bool pipeline_round = pipeline_parent_prep && expansions_this_round == 2;
        if (pipeline_round)
            ++pipelined_rounds;
        steady_clock::time_point profile_phase_start;
        if (profile_phases)
            profile_phase_start = steady_clock::now();
#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        if (!parent_pipeline || parent_pipeline == 1)
            for (int visit = 0; visit < num_gpus; ++visit)
            {
                const int g = (visit + submit_shift) % num_gpus;
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
                if (!parent_pipeline)
                {
                    if (defer_round_sync)
                        gpuErrchk(cudaStreamWaitEvent(gi.streamParent, gi.eventKernelsReady, 0));

                    gpuErrchk(
                        cudaMemcpyAsync(h_parents[g], gi.d_ngpu_parents,
                                        sizeof(unsigned) * numQueries * NGPU_PARENT_RECORD_SIZE,
                                        cudaMemcpyDeviceToHost, gi.streamParent));
                }

                if (warp_sort)
                    neighbor_sort_warp_v3<<<(numQueries + 3) / 4, 128, 0, gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                        gi.d_nextIter, R, numQueries);
                else
                    compute_BestLSets_par_sort_msort<<<numQueries, gi.numThreads_K3, 0,
                                                       gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_neighbors_aux, gi.d_numNeighbors_query,
                        gi.d_neighborsDist_query, gi.d_neighborsDist_query_aux, gi.d_nextIter, R);
            }
        steady_clock::time_point profile_wait_start;
        if (profile_detail)
        {
            profile_wait_start = steady_clock::now();
            profile_parent_submit_ms += profile_elapsed_ms(profile_phase_start, profile_wait_start);
        }
        for (int g = 0; g < num_gpus; ++g)
        {
            steady_clock::time_point wait_begin;
            if (profile_detail)
                wait_begin = steady_clock::now();
            if (parent_pipeline == 4)
            {
                wait_event(gpu_impls[g]->m_objGPUInst.eventKernelsReady);
            }
            else
            {
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                gpuErrchk(cudaStreamSynchronize(parent_pipeline >= 3
                                                    ? gpu_impls[g]->m_objGPUInst.streamKernels
                                                    : gpu_impls[g]->m_objGPUInst.streamParent));
            }
            if (profile_detail)
                profile_wait_by_gpu[g] += profile_elapsed_ms(wait_begin, steady_clock::now());
            if (parent_prefetch_lines && g + 1 < num_gpus)
            {
                gpuErrchk(cudaSetDevice(gpu_impls[g + 1]->m_objGPUInst.physical_device));
                const cudaError_t status =
                    cudaStreamQuery(gpu_impls[g + 1]->m_objGPUInst.streamParent);
                if (status == cudaErrorNotReady)
                {
                    for (int q = 0; q < numQueries; ++q)
                    {
                        const size_t base = (size_t)q * NGPU_PARENT_RECORD_SIZE;
                        const unsigned count =
                            std::min(h_parents[g][base], (unsigned)NGPU_PARENTS_PER_ROUND);
                        for (unsigned p = 0; p < count; ++p)
                        {
                            const unsigned id = h_parents[g][base + 1 + 2 * p];
                            if (id >= shard_ends.back())
                                continue;
                            const uint8_t* adjacency = idx0.pIndex +
                                                       (uint64_t)id * idx0.ullIndex_Entry_LEN +
                                                       (size_t)D * sizeof(T);
                            for (unsigned line = 0; line < parent_prefetch_lines; ++line)
                                __builtin_prefetch(adjacency + line * 64, 0, 3);
                        }
                    }
                }
                else if (status != cudaSuccess)
                {
                    gpuErrchk(status);
                }
            }
        }
        if (profile_detail)
            profile_parent_wait_ms += profile_elapsed_ms(profile_wait_start, steady_clock::now());
        if (defer_round_sync)
        {
            nextIter = false;
            for (int g = 0; g < num_gpus && !nextIter; ++g)
            {
                for (int q = 0; q < numQueries; ++q)
                {
                    if (h_parents[g][(size_t)q * NGPU_PARENT_RECORD_SIZE] != 0)
                    {
                        nextIter = true;
                        break;
                    }
                }
            }
            if (!nextIter)
            {
                for (int g = 0; g < num_gpus; ++g)
                {
                    cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                    wait_stream(gpu_impls[g]->m_objGPUInst.streamKernels);
                }
                break;
            }
        }
        if (profile_phases)
        {
            const auto now = steady_clock::now();
            profile_parent_ms += profile_elapsed_ms(profile_phase_start, now);
            profile_phase_start = now;
        }
#ifdef _TIMERS
        t_end = high_resolution_clock::now();
        time_1 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        core_trace.set("CPU_arbitrate_graph");
        bool parent_launched_in_graph = false;
        std::atomic<bool> parallel_parent_in_graph{false};
        const unsigned round_iter = iter;
        auto upload_expansion = [&](int p, int g, bool select_device)
        {
            if (skip_tail_round)
                return;
            if (select_device)
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
            GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
            const bool use_alt = (p % 2 != 0);
            cudaEvent_t ready_event = use_alt ? gi.eventChildrenReadyAlt : gi.eventChildrenReady;

            if (pack_expansion_upload)
            {
                unsigned* d_upload = use_alt ? gi.d_expansion_upload_alt : gi.d_expansion_upload;
                const size_t upload_words =
                    compact_upload ? compact_words[p * num_gpus + g] : (size_t)numQueries * (R + 3);
                if (parallel_gpu_submit)
                {
#pragma omp atomic update
                    upload_bytes += upload_words * sizeof(unsigned);
#pragma omp atomic update
                    legacy_upload_bytes += (size_t)numQueries * (R + 3) * sizeof(unsigned);
                }
                else
                {
                    upload_bytes += upload_words * sizeof(unsigned);
                    legacy_upload_bytes += (size_t)numQueries * (R + 3) * sizeof(unsigned);
                }
                gpuErrchk(cudaMemcpyAsync(
                    d_upload,
                    compact_upload ? compact_buffers[p * num_gpus + g] : h_global_marks[p],
                    upload_words * sizeof(unsigned), cudaMemcpyHostToDevice, gi.streamChildren));
            }
            else
            {
                unsigned* d_mark_input = use_alt ? gi.d_mark_alt : gi.d_mark;
                unsigned* d_neighbors_input =
                    use_alt ? gi.d_neighbors_temp_alt : gi.d_neighbors_temp;
                unsigned* d_counts_input =
                    use_alt ? gi.d_numNeighbors_query_temp_alt : gi.d_numNeighbors_query_temp;
                gpuErrchk(cudaMemcpyAsync(d_mark_input, h_global_marks[p],
                                          sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice,
                                          gi.streamChildren));
                gpuErrchk(cudaMemcpyAsync(d_neighbors_input, h_expansion_neighbors[p],
                                          sizeof(unsigned) * numQueries * uMaxNeighbors,
                                          cudaMemcpyHostToDevice, gi.streamChildren));
                gpuErrchk(cudaMemcpyAsync(d_counts_input, h_expansion_counts[p],
                                          sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice,
                                          gi.streamChildren));
            }
            gpuErrchk(cudaEventRecord(ready_event, gi.streamChildren));
        };
        auto launch_expansion = [&](int p, int g, bool select_device)
        {
            if (skip_tail_round)
                return;
            const unsigned iter = round_iter + p;
            const bool append_sort = (p + 1 < expansions_this_round);
            const bool use_alt = (p % 2 != 0);
            const bool graph_expansion = use_cuda_graph && (iter > 1) &&
                                         ((!use_alt && append_sort) || (use_alt && !append_sort));
            if (select_device)
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
            GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
            IndexLoad& idx = gpu_impls[g]->m_objInputData;
            unsigned* d_upload = use_alt ? gi.d_expansion_upload_alt : gi.d_expansion_upload;
            unsigned* d_mark_input =
                pack_expansion_upload ? d_upload : (use_alt ? gi.d_mark_alt : gi.d_mark);
            unsigned* d_counts_input =
                pack_expansion_upload
                    ? d_upload + numQueries
                    : (use_alt ? gi.d_numNeighbors_query_temp_alt : gi.d_numNeighbors_query_temp);
            unsigned* d_neighbors_input =
                pack_expansion_upload ? d_upload + (compact_upload ? 5 : 2) * (size_t)numQueries
                                      : (use_alt ? gi.d_neighbors_temp_alt : gi.d_neighbors_temp);
            cudaEvent_t ready_event = use_alt ? gi.eventChildrenReadyAlt : gi.eventChildrenReady;
            gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, ready_event, 0));

            if (graph_expansion)
            {
                const int variant = use_alt ? NGPU_EXPANSION_GRAPH_BUFFER1_NO_SORT
                                            : NGPU_EXPANSION_GRAPH_BUFFER0_WITH_SORT;
                gpuErrchk(cudaGraphLaunch(gi.ngpuExpansionGraphs[variant], gi.streamKernels));
                if (graph_parent_node && use_alt && !append_sort)
                {
                    if (parallel_gpu_submit)
                        parallel_parent_in_graph.store(true, std::memory_order_relaxed);
                    else
                        parent_launched_in_graph = true;
                }
            }
            else
            {
                if (fuse_expansion)
                {
                    ngpu_expansion_fused<<<numQueries, expansion_threads, 0, gi.streamKernels>>>(
                        gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                        gi.d_neighborsDist_query, gi.d_BestLSets, gi.d_BestLSetsDist,
                        gi.d_BestLSets_visited, gi.d_BestLSets_count, d_mark_input, iter, uWLLen,
                        MEDOID, gi.d_processed_bit_vec, idx.d_compressedVectors, gi.d_pqDistTables,
                        uChunks, R, gi.uShardStart, gi.uShardEnd, gi.d_query_seeds);
                }
                else
                {
                    compute_BestLSets_par_merge<<<numQueries, gi.numThreads_K3_merge,
                                                  uMaxNeighbors * sizeof(float),
                                                  gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                        gi.d_BestLSets, gi.d_BestLSetsDist, gi.d_BestLSets_visited, gi.d_parents,
                        iter, gi.d_nextIter, gi.d_BestLSets_count, d_mark_input, uWLLen, MEDOID, R,
                        gi.d_query_seeds);
                }
                if (!fuse_expansion && fuse_filter_distance)
                {
                    neighbor_filtering_distance_fused<<<numQueries, gi.numThreads_K2, 0,
                                                        gi.streamKernels>>>(
                        gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                        gi.d_processed_bit_vec, idx.d_compressedVectors, gi.d_pqDistTables,
                        gi.d_neighborsDist_query, uChunks, R, gi.uShardStart, gi.uShardEnd,
                        compact_upload ? d_upload + 2 * (size_t)numQueries : nullptr,
                        compact_upload ? d_upload + 3 * (size_t)numQueries : nullptr);
                }
                else if (!fuse_expansion)
                {
                    gpuErrchk(cudaMemsetAsync(gi.d_numNeighbors_query, 0,
                                              sizeof(unsigned) * numQueries, gi.streamKernels));
                    neighbor_filtering_new<<<numQueries, gi.numThreads_K5, 0, gi.streamKernels>>>(
                        gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                        gi.d_processed_bit_vec, R, gi.uShardStart, gi.uShardEnd);
                    compute_neighborDist_par<<<numQueries, gi.numThreads_K2, 0, gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_numNeighbors_query, idx.d_compressedVectors,
                        gi.d_pqDistTables, gi.d_neighborsDist_query, uChunks, R, gi.uShardStart);
                }
                if (append_sort)
                {
                    if (warp_sort)
                        neighbor_sort_warp_v3<<<(numQueries + 3) / 4, 128, 0, gi.streamKernels>>>(
                            gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                            gi.d_nextIter, R, numQueries);
                    else
                        compute_BestLSets_par_sort_msort<<<numQueries, gi.numThreads_K3, 0,
                                                           gi.streamKernels>>>(
                            gi.d_neighbors, gi.d_neighbors_aux, gi.d_numNeighbors_query,
                            gi.d_neighborsDist_query, gi.d_neighborsDist_query_aux, gi.d_nextIter,
                            R);
                }
            }
        };
        for (int p = 0; p < expansions_this_round; ++p)
        {
            memset(h_expansion_counts[p], 0, sizeof(unsigned) * numQueries);
            std::fill_n(h_global_marks[p], numQueries, INVALID_NODE_ID);
        }

        omp_set_num_threads(h0.numCPUthreads);
        if (profile_detail)
        {
            std::fill(detail_select.begin(), detail_select.end(), 0.0);
            std::fill(detail_fetch.begin(), detail_fetch.end(), 0.0);
        }
        unsigned compact_errors = 0;
        unsigned invalid_parent_records = 0;
        if (tile_pack)
            std::fill(compact_overflow.begin(), compact_overflow.end(), 0u);
        steady_clock::time_point pack_begin;
        auto graph_work = [&](int tid, int nth, int first, int limit)
        {
            double select_work = 0.0;
            double fetch_work = 0.0;

            const uint64_t entryLen = idx0.ullIndex_Entry_LEN;
            uint8_t* const pIndex = idx0.pIndex;
            const unsigned nDim = (unsigned)idx0.D;
            const size_t fpBytes = (size_t)nDim * sizeof(T);

            constexpr int MAX_QBLK = 64;
            const int QBLK = static_cast<int>(cpu_query_block);
            const int PF_LINES = compact_upload ? 6 : 5;
            const size_t adjacency_offset = compact_upload ? 0 : fpBytes;
            const int LOOKAHEAD = static_cast<int>(graph_lookahead);

            uint8_t* src_list[MAX_QBLK * NGPU_PARENTS_PER_ROUND];
            int q_list[MAX_QBLK * NGPU_PARENTS_PER_ROUND];
            uint8_t expansion_list[MAX_QBLK * NGPU_PARENTS_PER_ROUND];

            for (int base = first + tid * QBLK; base < limit; base += nth * QBLK)
            {
                int cnt = 0;
                int end = base + QBLK;
                if (end > limit)
                    end = limit;

                steady_clock::time_point work_begin;
                if (profile_detail)
                    work_begin = steady_clock::now();
                for (int q = base; q < end; ++q)
                {
                    unsigned global_parent[NGPU_PARENTS_PER_ROUND];
                    float best_dist[NGPU_PARENTS_PER_ROUND];
                    for (int p = 0; p < NGPU_PARENTS_PER_ROUND; ++p)
                    {
                        best_dist[p] = 1e30f;
                        global_parent[p] = INVALID_NODE_ID;
                    }

                    for (int g = 0; g < num_gpus; ++g)
                    {
                        const size_t parent_base = (size_t)q * NGPU_PARENT_RECORD_SIZE;
                        const unsigned count =
                            std::min(h_parents[g][parent_base], (unsigned)NGPU_PARENTS_PER_ROUND);
                        for (unsigned c = 0; c < count; ++c)
                        {
                            const unsigned pid = h_parents[g][parent_base + 1 + 2 * c];
                            const unsigned bits = h_parents[g][parent_base + 2 + 2 * c];
                            if (pid >= idx0.N)
                            {
#pragma omp atomic update
                                ++invalid_parent_records;
                                continue;
                            }
                            float dist;
                            memcpy(&dist, &bits, sizeof(float));

                            phms_v3::offer_parent(pid, dist, global_parent, best_dist);
                        }
                    }

                    for (int p = 0; p < expansions_this_round; ++p)
                    {
                        h_global_marks[p][q] = global_parent[p];
                        if (compact_upload)
                            compact_records[(size_t)p * numQueries + q] = nullptr;
                        if (pipeline_round && p == 1)
                            deferred_parent_rows[q] = nullptr;
                        if (global_parent[p] == INVALID_NODE_ID)
                            continue;

                        unsigned track_idx = h_global_parent_counts[q];
                        if (track_idx < uMaxParentsPerQuery)
                        {
                            h_global_parent_track[(size_t)track_idx * numQueries + q] =
                                global_parent[p];
                            h_global_parent_counts[q] = track_idx + 1;
                        }

                        if (skip_tail_round)
                            continue;
                        uint8_t* const parent_row =
                            compact_upload ? (uint8_t*)g_shard_adjacency->record(global_parent[p])
                                           : pIndex + entryLen * (uint64_t)global_parent[p];
                        if (pipeline_round && p == 1)
                        {
                            deferred_parent_rows[q] = parent_row;
                            continue;
                        }
                        src_list[cnt] = parent_row;
                        q_list[cnt] = q;
                        expansion_list[cnt] = (uint8_t)p;
                        ++cnt;
                    }
                }

                if (profile_detail)
                {
                    const auto now = steady_clock::now();
                    select_work += profile_elapsed_ms(work_begin, now);
                    work_begin = now;
                }
                for (int i = 0; i < min(LOOKAHEAD, cnt); ++i)
                {
                    uint8_t* pf_src = src_list[i] + adjacency_offset;
#pragma unroll
                    for (int l = 0; l < PF_LINES; ++l)
                    {
                        __builtin_prefetch(pf_src + (size_t)l * 64, 0, 3);
                    }
                }

                const unsigned offset = (idx0.R + 1);

                for (int i = 0; i < cnt; ++i)
                {
                    if (i + LOOKAHEAD < cnt)
                    {
                        uint8_t* pf_src = src_list[i + LOOKAHEAD] + adjacency_offset;
#pragma unroll
                        for (int l = 0; l < PF_LINES; ++l)
                        {
                            __builtin_prefetch(pf_src + (size_t)l * 64, 0, 3);
                        }
                    }

                    const int q = q_list[i];
                    const int p = expansion_list[i];
                    uint8_t* src = src_list[i];
                    if (compact_upload)
                    {
                        compact_records[(size_t)p * numQueries + q] =
                            reinterpret_cast<const uint32_t*>(src);
                        continue;
                    }
                    const size_t offset_neighbors = (size_t)q * offset;

                    unsigned* puNumNeighbours = (unsigned*)(src + fpBytes);
                    unsigned deg = *puNumNeighbours;

                    if (deg > idx0.R)
                        deg = idx0.R;
                    memcpy(h_expansion_neighbors[p] + offset_neighbors, puNumNeighbours + 1,
                           (size_t)deg * sizeof(unsigned));
                    h_expansion_counts[p][q] = deg;
                }
                if (tile_pack)
                    phms_pack_shard_tile(compact_records.data(), h_global_marks.data(),
                                         compact_buffers.data(), numQueries, expansions_this_round,
                                         num_gpus, R, base, end, compact_tile_rows,
                                         compact_tile_quota, compact_overflow.data(),
                                         compact_errors);
                if (profile_detail)
                    fetch_work += profile_elapsed_ms(work_begin, steady_clock::now());
            }
            if (pipeline_round)
            {
#pragma omp barrier
#pragma omp master
                {
                    steady_clock::time_point submit_begin;
                    if (profile_phases)
                        submit_begin = steady_clock::now();
                    for (int g = 0; g < num_gpus; ++g)
                        upload_expansion(0, g, true);
                    for (int g = 0; g < num_gpus; ++g)
                        launch_expansion(0, g, true);
                    if (profile_phases)
                        profile_early_submit_ms +=
                            profile_elapsed_ms(submit_begin, steady_clock::now());
                }
                steady_clock::time_point deferred_begin;
                if (profile_detail)
                    deferred_begin = steady_clock::now();
                for (int base = tid * QBLK; base < numQueries; base += nth * QBLK)
                {
                    int cnt = 0;
                    const int end = std::min(base + QBLK, numQueries);
                    for (int q = base; q < end; ++q)
                        if (deferred_parent_rows[q])
                        {
                            src_list[cnt] = deferred_parent_rows[q];
                            q_list[cnt++] = q;
                        }
                    for (int i = 0; i < std::min(LOOKAHEAD, cnt); ++i)
                        for (int l = 0; l < PF_LINES; ++l)
                            __builtin_prefetch(src_list[i] + fpBytes + (size_t)l * 64, 0, 3);
                    for (int i = 0; i < cnt; ++i)
                    {
                        if (i + LOOKAHEAD < cnt)
                            for (int l = 0; l < PF_LINES; ++l)
                                __builtin_prefetch(
                                    src_list[i + LOOKAHEAD] + fpBytes + (size_t)l * 64, 0, 3);
                        const unsigned* row =
                            reinterpret_cast<const unsigned*>(src_list[i] + fpBytes);
                        const unsigned degree = std::min(*row, R);
                        const unsigned q = q_list[i];
                        memcpy(h_expansion_neighbors[1] + (size_t)q * (R + 1), row + 1,
                               (size_t)degree * 4);
                        h_expansion_counts[1][q] = degree;
                    }
                }
                if (profile_detail)
                    fetch_work += profile_elapsed_ms(deferred_begin, steady_clock::now());
            }
            if (profile_detail)
            {
                detail_select[tid] = select_work;
                detail_fetch[tid] = fetch_work;
            }
            if (compact_upload && !tile_pack)
            {
#pragma omp barrier
                if (profile_phases)
                {
#pragma omp single
                    {
                        pack_begin = steady_clock::now();
                    }
                }
                phms_pack_shard_rows(compact_records.data(), h_global_marks.data(),
                                     compact_buffers.data(), numQueries, expansions_this_round,
                                     num_gpus, R, h0.numCPUthreads, compact_counts.data(),
                                     compact_bases.data(), compact_words.data(), compact_errors);
                if (profile_phases)
                {
#pragma omp barrier
#pragma omp master
                    {
                        profile_compact_pack_ms +=
                            profile_elapsed_ms(pack_begin, steady_clock::now());
                    }
                }
            }
        };
#pragma omp parallel
        {
            graph_work(omp_get_thread_num(), omp_get_num_threads(), 0, numQueries);
        }
        if (invalid_parent_records)
            throw std::runtime_error(
                "out-of-range GPU Parent ID; search aborted before graph access");
        if (compact_errors)
            throw std::runtime_error("invalid offline row or compact packing overflow");
        if (tile_pack)
            for (int p = 0; p < expansions_this_round; ++p)
                for (int g = 0; g < num_gpus; ++g)
                {
                    const size_t i = p * num_gpus + g;
                    compact_words[i] =
                        5 * (size_t)numQueries + compact_fixed_payload + compact_overflow[i];
                    compact_overflow_bytes += (uint64_t)compact_overflow[i] * 4;
                }
        if (profile_detail)
        {
            double max_select = 0.0;
            double max_fetch = 0.0;
            for (size_t t = 0; t < detail_select.size(); ++t)
            {
                profile_select_work_ms += detail_select[t];
                profile_fetch_work_ms += detail_fetch[t];
                max_select = std::max(max_select, detail_select[t]);
                max_fetch = std::max(max_fetch, detail_fetch[t]);
            }
            profile_select_max_ms += max_select;
            profile_fetch_max_ms += max_fetch;
        }
        if (profile_phases)
        {
            const auto now = steady_clock::now();
            profile_cpu_ms += profile_elapsed_ms(profile_phase_start, now);
            profile_phase_start = now;
        }

#ifdef _TIMERS
        t_end = high_resolution_clock::now();
        time_2 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
        t_start = high_resolution_clock::now();
#endif
        core_trace.set("GPU_submit_CPU_overlap");
        const int submissions = expansions_this_round * num_gpus;
        if (parallel_gpu_submit)
        {
#pragma omp parallel for num_threads(h0.numCPUthreads) schedule(static)
            for (int g = 0; g < num_gpus; ++g)
            {
                gpuErrchk(cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device));
                for (int p = 0; p < expansions_this_round; ++p)
                    upload_expansion(p, g, false);
                for (int p = 0; p < expansions_this_round; ++p)
                    launch_expansion(p, g, false);
            }
            parent_launched_in_graph = parallel_parent_in_graph.load(std::memory_order_relaxed);
        }
        else if (combined_round_graph && use_cuda_graph && expansions_this_round == 2 &&
                 round_iter > 1)
        {
            for (int p = 0; p < 2; ++p)
                for (int g = 0; g < num_gpus; ++g)
                    upload_expansion(p, g, true);
            for (int g = 0; g < num_gpus; ++g)
            {
                auto& gi = gpu_impls[g]->m_objGPUInst;
                gpuErrchk(cudaSetDevice(gi.physical_device));
                gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, gi.eventChildrenReady, 0));
                gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, gi.eventChildrenReadyAlt, 0));
                gpuErrchk(cudaGraphLaunch(gi.ngpuExpansionGraphs[0], gi.streamKernels));
            }
            parent_launched_in_graph = graph_parent_node;
        }
        else if (parents_per_round > 2)
        {
            for (int p = 0; p < expansions_this_round; ++p)
                for (int g = 0; g < num_gpus; ++g)
                {
                    gpuErrchk(cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device));
                    GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
                    if (p >= 2)
                        gpuErrchk(cudaStreamWaitEvent(gi.streamChildren, gi.eventKernelsReady, 0));
                    upload_expansion(p, g, false);
                    launch_expansion(p, g, false);
                    gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
                }
        }
        else
        {
            for (int submit = 0; submit < submissions; ++submit)
            {
                const int p =
                    device_local_submit ? submit % expansions_this_round : submit / num_gpus;
                const int g0 =
                    device_local_submit ? submit / expansions_this_round : submit % num_gpus;
                const int g = (g0 + submit_shift) % num_gpus;
                if (pipeline_round && p == 0)
                    continue;
                upload_expansion(p, g, !device_local_submit || p == 0 || pipeline_round);
            }
            for (int submit = 0; submit < submissions; ++submit)
            {
                const int p =
                    device_local_submit ? submit % expansions_this_round : submit / num_gpus;
                const int g0 =
                    device_local_submit ? submit / expansions_this_round : submit % num_gpus;
                const int g = (g0 + submit_shift) % num_gpus;
                if (pipeline_round && p == 0)
                    continue;
                launch_expansion(p, g, !device_local_submit || p == 0 || pipeline_round);
            }
        }
        iter += expansions_this_round;

        if (!skip_tail_round && !(parent_pipeline >= 3 && parent_launched_in_graph))
            for (int g = 0; g < num_gpus; ++g)
            {
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                GPUInstance& gi = gpu_impls[g]->m_objGPUInst;
                if (!parent_launched_in_graph)
                {
                    compute_parent2_ngpu2<<<(numQueries + gi.numThreads_K4 - 1) / gi.numThreads_K4,
                                            gi.numThreads_K4, 0, gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                        gi.d_BestLSets, gi.d_BestLSetsDist, gi.d_BestLSets_visited,
                        gi.d_BestLSets_count, gi.d_ngpu_parents, gi.d_nextIter, numQueries, uWLLen,
                        MEDOID, R, gi.d_query_seeds);
                }
                if (defer_round_sync && parent_pipeline < 3)
                {
                    gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
                }
                else if (!defer_round_sync)
                {
                    gpuErrchk(cudaMemcpyAsync(&nextIter_local[g], gi.d_nextIter, sizeof(bool),
                                              cudaMemcpyDeviceToHost, gi.streamKernels));
                }
                if ((parent_pipeline == 1 || parent_pipeline == 2) && iter < uMaxParentsPerQuery)
                    queue_parent_handoff(g);
                if (parent_pipeline == 2 && iter < uMaxParentsPerQuery)
                    queue_next_sort(g);
                if (parent_pipeline >= 3 && !parent_launched_in_graph && iter < uMaxParentsPerQuery)
                    queue_direct_parent_tail(g);
            }

        if (overlap_cpu_rerank)
        {
            PhmsTraceRange fp_trace("CPU_incremental_FP");
            steady_clock::time_point fp_begin;
            if (profile_phases)
                fp_begin = steady_clock::now();
            auto fp_query = [&](int q)
            {
                const unsigned end = std::min(h_global_parent_counts[q], uMaxParentsPerQuery);
                const T* query_fp = query_vector(q);
                const unsigned dim_adj = params.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0;
                for (unsigned i = cached_fp_counts[q]; i < end; ++i)
                {
                    const unsigned pid = h_global_parent_track[(size_t)i * numQueries + q];
                    const T* fp = reinterpret_cast<const T*>(idx0.pIndex + idx0.ullIndex_Entry_LEN *
                                                                               (uint64_t)pid);
                    float dist = 0.0f;
#pragma omp simd reduction(+ : dist)
                    for (unsigned d = 0; d < D - dim_adj; ++d)
                    {
                        float diff = (float)fp[d] - (float)query_fp[d];
                        dist += diff * diff;
                    }
                    cached_fp_distances[(size_t)i * numQueries + q] = dist;
                }
                cached_fp_counts[q] = end;
            };
#pragma omp parallel for num_threads(h0.numCPUthreads) schedule(static)
            for (int q = 0; q < numQueries; ++q)
                fp_query(q);
            if (profile_phases)
                profile_overlap_rerank_ms += profile_elapsed_ms(fp_begin, steady_clock::now());
        }
        if (!defer_round_sync)
        {
            nextIter = false;
            for (int g = 0; g < num_gpus; ++g)
            {
                cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                cudaStreamSynchronize(gpu_impls[g]->m_objGPUInst.streamKernels);
                if (nextIter_local[g])
                    nextIter = true;
            }
        }
        if (profile_phases)
            profile_gpu_ms += profile_elapsed_ms(profile_phase_start, steady_clock::now());

        if (iter >= uMaxParentsPerQuery)
        {
            core_trace.set("Final_budget_GPU_wait");
            if (defer_round_sync)
            {
                for (int g = 0; g < num_gpus; ++g)
                {
                    cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device);
                    wait_stream(gpu_impls[g]->m_objGPUInst.streamKernels);
                }
            }
            printf("[bzhang NGPU] Warning: expansion budget reached "
                   "(expanded=%u, coordination_rounds=%u)\n",
                   iter, coordination_rounds);
            break;
        }

#ifdef _TIMERS
        t_end = high_resolution_clock::now();
        time_3 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

    } while (nextIter);

    if (final_candidates)
    {
        for (int g = 0; g < num_gpus; ++g)
        {
            gpuErrchk(cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device));
            auto& gi = gpu_impls[g]->m_objGPUInst;
            gpuErrchk(
                cudaMemcpyAsync(h_parents[g], gi.d_ngpu_parents,
                                (size_t)numQueries * NGPU_PARENT_RECORD_SIZE * sizeof(unsigned),
                                cudaMemcpyDeviceToHost, gi.streamParent));
        }
        for (int g = 0; g < num_gpus; ++g)
        {
            gpuErrchk(cudaSetDevice(gpu_impls[g]->m_objGPUInst.physical_device));
            gpuErrchk(cudaStreamSynchronize(gpu_impls[g]->m_objGPUInst.streamParent));
        }
        unsigned errors = 0;
#pragma omp parallel for num_threads(h0.numCPUthreads) schedule(static)
        for (int q = 0; q < numQueries; ++q)
        {
            unsigned n = h_global_parent_counts[q];
            for (int g = 0; g < num_gpus; ++g)
            {
                const unsigned* row = h_parents[g] + (size_t)q * NGPU_PARENT_RECORD_SIZE;
                for (unsigned i = 0; i < std::min(row[0], final_candidates); ++i)
                {
                    if (phms_v3::append_candidate(h_global_parent_track, numQueries, q,
                                                  track_capacity, n, row[1 + 2 * i],
                                                  shard_ends.back()) < 0)
                    {
#pragma omp atomic update
                        ++errors;
                    }
                }
            }
            h_global_parent_counts[q] = n;
        }
        if (errors)
            throw std::runtime_error("invalid v3 final candidate or track overflow");
    }
    core_trace.set("Tail_rerank");
    steady_clock::time_point profile_rerank_start;
    if (profile_phases)
        profile_rerank_start = steady_clock::now();

#ifdef _TIMERS
    t_start = high_resolution_clock::now();
#endif
    struct Cand
    {
        unsigned id;
        float dist;
    };

    if (!return_tracks_only)
    {
        memset(nearestNeighbours, 0, sizeof(result_ann_t) * numQueries * params.recall);
        memset(nearestNeighbours_dist, 0, sizeof(float) * numQueries * params.recall);

        const unsigned max_cands = track_capacity;

        auto rerank_range = [&](unsigned first, unsigned end)
        {
            std::vector<Cand> thread_cand_buf(max_cands);

            for (unsigned q = first; q < end; q++)
            {
                unsigned n = h_global_parent_counts[q];
                const T* query_fp = query_vector(q);
                if (n == 0)
                    continue;

                uint8_t* pIndex = idx0.pIndex;
                unsigned long long entryLen = idx0.ullIndex_Entry_LEN;
                unsigned nDim = (unsigned)idx0.D;
                unsigned dimAdj = (params.uDistFunc == ENUM_DIST_MIPS) ? MIPS_EXTRA_DIM : 0;

                unsigned valid_count = 0;
                for (unsigned i = 0; i < n && i < max_cands; i++)
                {
                    unsigned pid = h_global_parent_track[(size_t)i * numQueries + q];
                    T* fp_vec = (T*)(pIndex + entryLen * (uint64_t)pid);
                    float dist = 0.0f;

                    if (overlap_cpu_rerank && i < cached_fp_counts[q])
                    {
                        dist = cached_fp_distances[(size_t)i * numQueries + q];
                    }
                    else
                    {
#pragma omp simd reduction(+ : dist)
                        for (unsigned d = 0; d < nDim - dimAdj; d++)
                        {
                            float diff = (float)fp_vec[d] - (float)query_fp[d];
                            dist += diff * diff;
                        }
                    }

                    thread_cand_buf[valid_count] = Cand{pid, dist};
                    valid_count++;
                }

                if (valid_count == 0)
                    continue;

                unsigned topk = std::min((unsigned)params.recall, valid_count);
                std::partial_sort(thread_cand_buf.begin(), thread_cand_buf.begin() + topk,
                                  thread_cand_buf.begin() + valid_count,
                                  [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

                for (unsigned j = 0; j < topk; j++)
                {
                    nearestNeighbours[(size_t)q * params.recall + j] =
                        (result_ann_t)thread_cand_buf[j].id;
                    nearestNeighbours_dist[(size_t)q * params.recall + j] = thread_cand_buf[j].dist;
                }
            }
        };
#pragma omp parallel num_threads(h0.numCPUthreads)
        {
            unsigned tid = omp_get_thread_num();
            unsigned nth = omp_get_num_threads();
            rerank_range(uint64_t(numQueries) * tid / nth, uint64_t(numQueries) * (tid + 1) / nth);
        }
    }
    if (profile_phases)
        profile_rerank_ms = profile_elapsed_ms(profile_rerank_start, steady_clock::now());
#ifdef _TIMERS
    t_end = high_resolution_clock::now();
    time_4 += duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
#endif

#ifdef _TIMERS
    printf("[bzhang NGPU] Search done. %d GPU(s), total iterations = %u\n", num_gpus, iter);
#endif

#ifdef _TIMERS

    double totalTime = time_K1 + time_B1 + time_neighbor_filtering + time_transfer +
                       time_convergence + seek_neighbours_time + time_prfetch + time_reranking +
                       time_B2;
    cout << "Total time from timers = (1) + (2) + (4) + (5) + (6) + (8) = " << totalTime << " ms"
         << endl;
    cout << " time_1: " << time_1 << " time_2: " << time_2 << " time_3: " << time_3
         << " time_4: " << time_4 << std::endl;
#endif

    if (profile_phases)
    {
        const double total_ms = profile_elapsed_ms(profile_total_start, steady_clock::now());
        const double accounted_ms = profile_init_ms + profile_parent_ms + profile_cpu_ms +
                                    profile_gpu_ms + profile_rerank_ms;
        std::lock_guard<std::mutex> profile_lock(g_ngpu_profile_print_mtx);
        printf("[PHMS PROFILE %u] queries=%d rounds=%u total=%.3f ms "
               "init=%.3f parent_d2h=%.3f cpu_arbitrate_graph=%.3f "
               "h2d_gpu_round=%.3f rerank=%.3f other=%.3f\n",
               profile_id, numQueries, coordination_rounds, total_ms, profile_init_ms,
               profile_parent_ms, profile_cpu_ms, profile_gpu_ms, profile_rerank_ms,
               total_ms - accounted_ms);
        if (profile_detail)
        {
            printf("[PHMS DETAIL %u] gpus=%d parent_submit=%.3f parent_wait=%.3f "
                   "select_worker_sum=%.3f graph_worker_sum=%.3f "
                   "select_round_max_sum=%.3f graph_round_max_sum=%.3f wait_by_gpu=",
                   profile_id, num_gpus, profile_parent_submit_ms, profile_parent_wait_ms,
                   profile_select_work_ms, profile_fetch_work_ms, profile_select_max_ms,
                   profile_fetch_max_ms);
            for (int g = 0; g < num_gpus; ++g)
                printf("%s%.3f", g ? "," : "", profile_wait_by_gpu[g]);
            printf("\n");
        }
        printf("[PHMS PIPELINE %u] enabled=%d early_rounds=%u early_submit=%.3f ms "
               "(submission included in cpu_arbitrate_graph)\n",
               profile_id, pipeline_parent_prep, pipelined_rounds, profile_early_submit_ms);
        printf("[PHMS RERANK OVERLAP %u] enabled=%d incremental_cpu=%.3f ms "
               "(included in h2d_gpu_round; final sort remains in rerank)\n",
               profile_id, overlap_cpu_rerank, profile_overlap_rerank_ms);
    }
    if (report_upload)
    {
        std::lock_guard<std::mutex> lock(g_ngpu_profile_print_mtx);
        printf(
            "[PHMS UPLOAD] compact=%d gpus=%d queries=%d bytes=%llu legacy_bytes=%llu pack_ms=%.3f "
            "tile=%d overflow_bytes=%llu\n",
            compact_upload, num_gpus, numQueries, (unsigned long long)upload_bytes,
            (unsigned long long)legacy_upload_bytes,
            (profile_phases && !tile_pack) ? profile_compact_pack_ms : -1.0, tile_pack,
            (unsigned long long)compact_overflow_bytes);
    }

    if (debug_tracks)
    {
        debug_tracks->assign(numQueries, {});
        for (unsigned q = 0; q < (unsigned)numQueries; ++q)
            for (unsigned p = 0; p < h_global_parent_counts[q]; ++p)
                (*debug_tracks)[q].push_back(h_global_parent_track[(size_t)p * numQueries + q]);
    }
}

template <typename T> void PHMSSearchInner<T>::phms_alloc(int numQueries)
{
    const bool compact_upload = phms_compact_upload_enabled();
    gpuErrchk(cudaGetDevice(&m_objGPUInst.physical_device));
    const unsigned uMAX_PARENTS_PERQUERY =
        (m_objSearchParams.worklist_length + NAX_EXTRA_ITERATION);
    m_objHostInst.FPSetCoords_size = m_objInputData.D;
    m_objHostInst.FPSetCoords_size_bytes = m_objHostInst.FPSetCoords_size * sizeof(T);
    m_objHostInst.FPSetCoords_rowsize = m_objHostInst.FPSetCoords_size * numQueries;
    m_objHostInst.FPSetCoords_rowsize_bytes = m_objHostInst.FPSetCoords_size_bytes * numQueries;

    gpuErrchk(cudaMalloc(&m_objGPUInst.d_query_seeds, sizeof(unsigned) * numQueries));
    m_objGPUInst.query_capacity = numQueries;
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_resident_query_ids, sizeof(unsigned) * numQueries));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_queriesFP, sizeof(T) * (numQueries * m_objInputData.D)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_nearestNeighbours,
                         (m_objSearchParams.recall * numQueries) * sizeof(result_ann_t)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_pqDistTables,
                         sizeof(float) * (256 * m_objInputData.uChunks * numQueries)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_BestLSetsDist,
                         sizeof(float) * (numQueries * (m_objSearchParams.worklist_length))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_BestLSets_count, sizeof(unsigned) * (numQueries)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_BestLSets_visited,
                         sizeof(bool) * (numQueries * (m_objSearchParams.worklist_length))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_BestLSets,
                         sizeof(unsigned) * (numQueries * (m_objSearchParams.worklist_length))));
    gpuErrchk(
        cudaMalloc(&m_objGPUInst.d_parents, sizeof(unsigned) * (numQueries * (SIZEPARENTLIST))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_ngpu_parents,
                         sizeof(unsigned) * numQueries * NGPU_PARENT_RECORD_SIZE));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighbors,
                         sizeof(unsigned) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighbors_temp,
                         sizeof(unsigned) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighbors_temp_alt,
                         sizeof(unsigned) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighbors_aux,
                         sizeof(unsigned) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_numNeighbors_query, sizeof(unsigned) * (numQueries)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_numNeighbors_query_temp, sizeof(unsigned) * numQueries));
    gpuErrchk(
        cudaMalloc(&m_objGPUInst.d_numNeighbors_query_temp_alt, sizeof(unsigned) * numQueries));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighborsDist_query,
                         sizeof(float) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_neighborsDist_query_aux,
                         sizeof(float) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_processed_bit_vec, sizeof(bool) * BF_MEMORY * numQueries));
    printf("[BLOOM STORAGE] entries=%u bytes_per_query=%u\n", (unsigned)BF_ENTRIES, BF_MEMORY);
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_nextIter, sizeof(bool)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_iter, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_mark, sizeof(unsigned) * (numQueries)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_mark_alt, sizeof(unsigned) * (numQueries)));
    const size_t expansion_upload_words =
        phms_compact_tile_enabled()
            ? (size_t)numQueries * (m_objInputData.R + 5) +
                  phms_compact_fixed_payload(numQueries, m_objInputData.R,
                                             g_shard_adjacency->header.shards)
            : (size_t)numQueries * (m_objInputData.R + (compact_upload ? 5 : 3));
    gpuErrchk(
        cudaMalloc(&m_objGPUInst.d_expansion_upload, expansion_upload_words * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_expansion_upload_alt,
                         expansion_upload_words * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_FPSetCoordsList_Counts, numQueries * sizeof(unsigned)));
    gpuErrchk(
        cudaMalloc(&m_objGPUInst.d_FPSetCoordsList,
                   (uMAX_PARENTS_PERQUERY * numQueries) * m_objHostInst.FPSetCoords_size_bytes));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_L2distances,
                         (uMAX_PARENTS_PERQUERY * numQueries) * sizeof(float)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_L2ParentIds,
                         (uMAX_PARENTS_PERQUERY * numQueries) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_L2distances_aux,
                         (uMAX_PARENTS_PERQUERY * numQueries) * sizeof(float)));
    gpuErrchk(cudaMalloc(&m_objGPUInst.d_L2ParentIds_aux,
                         (uMAX_PARENTS_PERQUERY * numQueries) * sizeof(unsigned)));

    gpuErrchk(cudaStreamCreate(&m_objGPUInst.streamFPTransfers));
    gpuErrchk(cudaStreamCreate(&m_objGPUInst.streamParent));
    gpuErrchk(cudaStreamCreate(&m_objGPUInst.streamChildren));
    gpuErrchk(cudaStreamCreate(&m_objGPUInst.streamKernels));
    gpuErrchk(cudaEventCreateWithFlags(&m_objGPUInst.eventChildrenReady, cudaEventDisableTiming));
    gpuErrchk(
        cudaEventCreateWithFlags(&m_objGPUInst.eventChildrenReadyAlt, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&m_objGPUInst.eventKernelsReady, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&m_objGPUInst.eventParentCopyDone, cudaEventDisableTiming));

    unsigned threads = 64;
    const char* env = getenv("PHMS_CPU_THREADS");
    if (env)
    {
        int v = atoi(env);
        if (v > 0)
            threads = (unsigned)v;
    }
    m_objHostInst.numCPUthreads = threads;
    std::cout << "m_objHostInst.numCPUthreads : " << m_objHostInst.numCPUthreads << std::endl;

    gpuErrchk(cudaMallocHost(&m_objHostInst.neighbors,
                             sizeof(unsigned) * (numQueries * (m_objInputData.R + 1))));
    gpuErrchk(cudaMallocHost(&m_objHostInst.numNeighbors_query, sizeof(unsigned) * (numQueries)));
    gpuErrchk(
        cudaMallocHost(&m_objHostInst.expansion_upload, expansion_upload_words * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&m_objHostInst.expansion_upload_alt,
                             expansion_upload_words * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&m_objHostInst.parents,
                             sizeof(unsigned) * numQueries * NGPU_PARENT_RECORD_SIZE));
    gpuErrchk(
        cudaMallocHost(&m_objHostInst.FPSetCoordsList, (uMAX_PARENTS_PERQUERY * numQueries) *
                                                           m_objHostInst.FPSetCoords_size_bytes));
}

template <typename T> void PHMSSearchInner<T>::phms_prepare_ngpu_graph(int numQueries)
{
    const bool warp_sort = phms_v3::warp_sort_enabled();
    const unsigned parent_pipeline = phms_v3::parent_pipeline();
    const bool compact_upload = phms_compact_upload_enabled();
    const char* graph_env = std::getenv("PHMS_USE_CUDA_GRAPH");
    if (graph_env != nullptr && std::atoi(graph_env) == 0)
        return;
    bool fuse_filter_distance = true;
    if (const char* fuse_env = std::getenv("PHMS_FUSE_FILTER_DISTANCE"))
        fuse_filter_distance = (std::atoi(fuse_env) != 0);
    bool fuse_expansion = false;
    if (const char* expansion_env = std::getenv("PHMS_FUSE_EXPANSION"))
        fuse_expansion = (std::atoi(expansion_env) != 0);
    fuse_expansion = fuse_expansion && fuse_filter_distance;
    const unsigned expansion_threads =
        phms_ngpu_launch_threads("PHMS_NGPU_EXPANSION_THREADS", 128, 32, 512, 32);
    bool graph_parent_node = true;
    if (const char* parent_env = std::getenv("PHMS_GRAPH_PARENT_NODE"))
        graph_parent_node = (std::atoi(parent_env) != 0);
    bool pack_expansion_upload = true;
    if (const char* pack_env = std::getenv("PHMS_PACK_EXPANSION_UPLOAD"))
        pack_expansion_upload = (std::atoi(pack_env) != 0);

    GPUInstance& gi = m_objGPUInst;
    IndexLoad& idx = m_objInputData;
    const unsigned uWLLen = m_objSearchParams.worklist_length;
    const unsigned uMaxNeighbors = idx.R + 1;

    gi.numThreads_K4 = 128;
    gi.numThreads_K1 = 256;
    gi.numThreads_K2 = phms_ngpu_launch_threads("PHMS_NGPU_DIST_THREADS", 512, 8, 512, 8);
    gi.numThreads_K3 = idx.R + 1;
    gi.K4_blockSize = 256;
    gi.numThreads_K5 = phms_ngpu_launch_threads("PHMS_NGPU_FILTER_THREADS", 256, 32, 1024, 32);
    gi.numThreads_K3_merge = 2 * uWLLen;
    assert(gi.numThreads_K3_merge <= 1024);
    const bool combined_round = std::getenv("PHMS_COMBINED_ROUND_GRAPH") &&
                                std::atoi(std::getenv("PHMS_COMBINED_ROUND_GRAPH"));
    cudaGraph_t retained_graphs[2] = {nullptr, nullptr};
    auto capture_expansion = [&](int variant, bool use_alt_buffer, bool append_sort)
    {
        unsigned* d_upload = use_alt_buffer ? gi.d_expansion_upload_alt : gi.d_expansion_upload;
        unsigned* d_mark_input =
            pack_expansion_upload ? d_upload : (use_alt_buffer ? gi.d_mark_alt : gi.d_mark);
        unsigned* d_counts_input = pack_expansion_upload
                                       ? d_upload + numQueries
                                       : (use_alt_buffer ? gi.d_numNeighbors_query_temp_alt
                                                         : gi.d_numNeighbors_query_temp);
        unsigned* d_neighbors_input =
            pack_expansion_upload
                ? d_upload + (compact_upload ? 5 : 2) * (size_t)numQueries
                : (use_alt_buffer ? gi.d_neighbors_temp_alt : gi.d_neighbors_temp);
        cudaGraph_t graph = nullptr;
        gpuErrchk(cudaStreamBeginCapture(gi.streamKernels, cudaStreamCaptureModeThreadLocal));

        if (fuse_expansion)
        {
            ngpu_expansion_fused<<<numQueries, expansion_threads, 0, gi.streamKernels>>>(
                gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                gi.d_neighborsDist_query, gi.d_BestLSets, gi.d_BestLSetsDist,
                gi.d_BestLSets_visited, gi.d_BestLSets_count, d_mark_input, 2u, uWLLen, idx.MEDOID,
                gi.d_processed_bit_vec, idx.d_compressedVectors, gi.d_pqDistTables, idx.uChunks,
                idx.R, gi.uShardStart, gi.uShardEnd, gi.d_query_seeds);
        }
        else
        {
            compute_BestLSets_par_merge<<<numQueries, gi.numThreads_K3_merge,
                                          uMaxNeighbors * sizeof(float), gi.streamKernels>>>(
                gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query, gi.d_BestLSets,
                gi.d_BestLSetsDist, gi.d_BestLSets_visited, gi.d_parents, 2u, gi.d_nextIter,
                gi.d_BestLSets_count, d_mark_input, uWLLen, idx.MEDOID, idx.R, gi.d_query_seeds);
        }
        if (!fuse_expansion && fuse_filter_distance)
        {
            neighbor_filtering_distance_fused<<<numQueries, gi.numThreads_K2, 0,
                                                gi.streamKernels>>>(
                gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                gi.d_processed_bit_vec, idx.d_compressedVectors, gi.d_pqDistTables,
                gi.d_neighborsDist_query, idx.uChunks, idx.R, gi.uShardStart, gi.uShardEnd,
                compact_upload ? d_upload + 2 * (size_t)numQueries : nullptr,
                compact_upload ? d_upload + 3 * (size_t)numQueries : nullptr);
        }
        else if (!fuse_expansion)
        {
            gpuErrchk(cudaMemsetAsync(gi.d_numNeighbors_query, 0, sizeof(unsigned) * numQueries,
                                      gi.streamKernels));
            neighbor_filtering_new<<<numQueries, gi.numThreads_K5, 0, gi.streamKernels>>>(
                gi.d_neighbors, d_neighbors_input, gi.d_numNeighbors_query, d_counts_input,
                gi.d_processed_bit_vec, idx.R, gi.uShardStart, gi.uShardEnd);
            compute_neighborDist_par<<<numQueries, gi.numThreads_K2, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_numNeighbors_query, idx.d_compressedVectors, gi.d_pqDistTables,
                gi.d_neighborsDist_query, idx.uChunks, idx.R, gi.uShardStart);
        }

        if (append_sort)
        {
            if (warp_sort)
                neighbor_sort_warp_v3<<<(numQueries + 3) / 4, 128, 0, gi.streamKernels>>>(
                    gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                    gi.d_nextIter, idx.R, numQueries);
            else
                compute_BestLSets_par_sort_msort<<<numQueries, gi.numThreads_K3, 0,
                                                   gi.streamKernels>>>(
                    gi.d_neighbors, gi.d_neighbors_aux, gi.d_numNeighbors_query,
                    gi.d_neighborsDist_query, gi.d_neighborsDist_query_aux, gi.d_nextIter, idx.R);
        }
        if (graph_parent_node && use_alt_buffer && !append_sort)
        {
            compute_parent2_ngpu2<<<(numQueries + gi.numThreads_K4 - 1) / gi.numThreads_K4,
                                    gi.numThreads_K4, 0, gi.streamKernels>>>(
                gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query, gi.d_BestLSets,
                gi.d_BestLSetsDist, gi.d_BestLSets_visited, gi.d_BestLSets_count, gi.d_ngpu_parents,
                gi.d_nextIter, numQueries, uWLLen, idx.MEDOID, idx.R, gi.d_query_seeds);
            if (parent_pipeline >= 3)
            {
                cudaStream_t copy_stream = gi.streamKernels;
                if (parent_pipeline == 5)
                {
                    gpuErrchk(cudaEventRecord(gi.eventKernelsReady, gi.streamKernels));
                    gpuErrchk(cudaStreamWaitEvent(gi.streamParent, gi.eventKernelsReady, 0));
                    copy_stream = gi.streamParent;
                }
                gpuErrchk(
                    cudaMemcpyAsync(m_objHostInst.parents, gi.d_ngpu_parents,
                                    (size_t)numQueries * NGPU_PARENT_RECORD_SIZE * sizeof(unsigned),
                                    cudaMemcpyDeviceToHost, copy_stream));
                if (parent_pipeline == 5)
                    gpuErrchk(cudaEventRecord(gi.eventParentCopyDone, gi.streamParent));
                if (warp_sort)
                    neighbor_sort_warp_v3<<<(numQueries + 3) / 4, 128, 0, gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_numNeighbors_query, gi.d_neighborsDist_query,
                        gi.d_nextIter, idx.R, numQueries);
                else
                    compute_BestLSets_par_sort_msort<<<numQueries, gi.numThreads_K3, 0,
                                                       gi.streamKernels>>>(
                        gi.d_neighbors, gi.d_neighbors_aux, gi.d_numNeighbors_query,
                        gi.d_neighborsDist_query, gi.d_neighborsDist_query_aux, gi.d_nextIter,
                        idx.R);
                if (parent_pipeline == 4)
                    gpuErrchk(cudaEventRecordWithFlags(gi.eventKernelsReady, gi.streamKernels,
                                                       cudaEventRecordExternal));
                if (parent_pipeline == 5)
                    gpuErrchk(cudaStreamWaitEvent(gi.streamKernels, gi.eventParentCopyDone, 0));
            }
        }

        gpuErrchk(cudaStreamEndCapture(gi.streamKernels, &graph));
        gpuErrchk(
            cudaGraphInstantiate(&gi.ngpuExpansionGraphs[variant], graph, nullptr, nullptr, 0));
        if (combined_round)
            retained_graphs[variant] = graph;
        else
            gpuErrchk(cudaGraphDestroy(graph));
    };

    capture_expansion(NGPU_EXPANSION_GRAPH_BUFFER0_WITH_SORT, false, true);
    capture_expansion(NGPU_EXPANSION_GRAPH_BUFFER1_NO_SORT, true, false);
    if (combined_round)
    {
        cudaGraph_t round = nullptr;
        cudaGraphNode_t first = nullptr, second = nullptr;
        cudaGraphExec_t exec = nullptr;
        gpuErrchk(cudaGraphCreate(&round, 0));
        gpuErrchk(cudaGraphAddChildGraphNode(&first, round, nullptr, 0, retained_graphs[0]));
        gpuErrchk(cudaGraphAddChildGraphNode(&second, round, &first, 1, retained_graphs[1]));
        gpuErrchk(cudaGraphInstantiate(&exec, round, nullptr, nullptr, 0));
        gpuErrchk(cudaGraphExecDestroy(gi.ngpuExpansionGraphs[0]));
        gi.ngpuExpansionGraphs[0] = exec;
        gpuErrchk(cudaGraphDestroy(round));
        for (auto graph : retained_graphs)
            gpuErrchk(cudaGraphDestroy(graph));
    }
}

template <typename T> void PHMSSearchInner<T>::phms_init(int numQueries)
{
    m_objGPUInst.numThreads_K4 = 1;
    m_objGPUInst.numThreads_K1 = 256;
    m_objGPUInst.numThreads_K2 = 512;
    m_objGPUInst.numThreads_K3 = m_objInputData.R + 1;
    m_objGPUInst.K4_blockSize = 256;
    m_objGPUInst.numThreads_K5 = 256;
    m_objGPUInst.numThreads_K3_merge = 2 * m_objSearchParams.worklist_length;
    assert(m_objGPUInst.numThreads_K3_merge <= 1024);

    gpuErrchk(cudaMemset(m_objGPUInst.d_iter, 0, sizeof(unsigned)));
    gpuErrchk(cudaMemset(m_objGPUInst.d_pqDistTables, 0,
                         sizeof(float) * (m_objInputData.uChunks * 256 * numQueries)));
    gpuErrchk(
        cudaMemset(m_objGPUInst.d_processed_bit_vec, 0, sizeof(bool) * BF_MEMORY * numQueries));
    gpuErrchk(
        cudaMemset(m_objGPUInst.d_parents, 0, sizeof(unsigned) * (numQueries * (SIZEPARENTLIST))));
    gpuErrchk(cudaMemset(m_objGPUInst.d_BestLSets_count, 0, sizeof(unsigned) * (numQueries)));
    gpuErrchk(cudaMemset(m_objGPUInst.d_mark, 1, sizeof(unsigned) * (numQueries)));
    gpuErrchk(cudaMemset(m_objGPUInst.d_neighborsDist_query, 0,
                         sizeof(float) * (numQueries * (m_objInputData.R + 1))));

    T* FPSetCoordsList = (T*)m_objHostInst.FPSetCoordsList;
    T* d_FPSetCoordsList = (T*)m_objGPUInst.d_FPSetCoordsList;
    unsigned* L2ParentIds = (unsigned*)malloc(sizeof(unsigned) * numQueries);
    unsigned* FPSetCoordsList_Counts = (unsigned*)malloc(sizeof(unsigned) * numQueries);
    for (int i = 0; i < numQueries; i++)
    {
        L2ParentIds[i] = m_objInputData.MEDOID;
        FPSetCoordsList_Counts[i] = 1;
    }
    gpuErrchk(cudaMemcpy(m_objGPUInst.d_L2ParentIds, L2ParentIds, sizeof(unsigned) * numQueries,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_objGPUInst.d_FPSetCoordsList_Counts, FPSetCoordsList_Counts,
                         sizeof(unsigned) * numQueries, cudaMemcpyHostToDevice));
    free(L2ParentIds);
    free(FPSetCoordsList_Counts);

    unsigned* puNumNeighbours =
        (unsigned*)(m_objInputData.pIndex +
                    ((m_objInputData.ullIndex_Entry_LEN * m_objInputData.MEDOID) +
                     (m_objInputData.D * sizeof(T))));
    unsigned medoidDegree = *puNumNeighbours;
    unsigned* puNeighbour = puNumNeighbours + 1;
    vector<unsigned> medoidNeighbors;
    for (unsigned long long ii = 0; ii < medoidDegree; ++ii)
        medoidNeighbors.push_back(puNeighbour[ii]);

    unsigned iter = 1;
    unsigned uMaxNeighbors = m_objInputData.R + 1;
    for (unsigned ii = 0; ii < numQueries; ++ii)
    {
        m_objHostInst.neighbors[ii * uMaxNeighbors] = m_objInputData.MEDOID;
        unsigned numNeighbors = 1;
        for (unsigned i = 0; i < medoidDegree; ++i)
        {
            m_objHostInst.neighbors[ii * uMaxNeighbors + i + 1] = medoidNeighbors[i];
            numNeighbors++;
        }
        m_objHostInst.numNeighbors_query[ii] = numNeighbors;
        memcpy(FPSetCoordsList + ((iter - 1) * m_objHostInst.FPSetCoords_rowsize) +
                   (ii * m_objHostInst.FPSetCoords_size),
               m_objInputData.pIndex + (m_objInputData.ullIndex_Entry_LEN * m_objInputData.MEDOID),
               m_objHostInst.FPSetCoords_size_bytes);
    }
    cudaMemcpy(d_FPSetCoordsList + ((iter - 1) * m_objHostInst.FPSetCoords_rowsize),
               FPSetCoordsList + ((iter - 1) * m_objHostInst.FPSetCoords_rowsize),
               m_objHostInst.FPSetCoords_rowsize_bytes, cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemcpy(m_objGPUInst.d_neighbors_temp, m_objHostInst.neighbors,
                         sizeof(unsigned) * numQueries * (m_objInputData.R + 1),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(m_objGPUInst.d_numNeighbors_query_temp, m_objHostInst.numNeighbors_query,
                         sizeof(unsigned) * (numQueries), cudaMemcpyHostToDevice));
}

template <typename T> void PHMSSearchInner<T>::phms_free()
{
    for (int variant = 0; variant < NGPU_EXPANSION_GRAPH_COUNT; ++variant)
    {
        if (m_objGPUInst.ngpuExpansionGraphs[variant] != nullptr)
        {
            gpuErrchk(cudaGraphExecDestroy(m_objGPUInst.ngpuExpansionGraphs[variant]));
            m_objGPUInst.ngpuExpansionGraphs[variant] = nullptr;
        }
    }
    gpuErrchk(cudaFree(m_objGPUInst.d_query_seeds));
    gpuErrchk(cudaFree(m_objGPUInst.d_resident_query_ids));
    m_objGPUInst.d_resident_query_ids = nullptr;
    m_objGPUInst.resident_selection_count = 0;
    m_objGPUInst.query_capacity = 0;
    m_objGPUInst.d_query_seeds = nullptr;
    gpuErrchk(cudaFree(m_objGPUInst.d_queriesFP));
    gpuErrchk(cudaFree(m_objGPUInst.d_nearestNeighbours));
    gpuErrchk(cudaFree(m_objGPUInst.d_pqDistTables));
    gpuErrchk(cudaFree(m_objGPUInst.d_BestLSetsDist));
    gpuErrchk(cudaFree(m_objGPUInst.d_BestLSets_visited));
    gpuErrchk(cudaFree(m_objGPUInst.d_BestLSets_count));
    gpuErrchk(cudaFree(m_objGPUInst.d_BestLSets));
    gpuErrchk(cudaFree(m_objGPUInst.d_parents));
    gpuErrchk(cudaFree(m_objGPUInst.d_ngpu_parents));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighbors));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighbors_temp));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighbors_temp_alt));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighbors_aux));
    gpuErrchk(cudaFree(m_objGPUInst.d_numNeighbors_query));
    gpuErrchk(cudaFree(m_objGPUInst.d_numNeighbors_query_temp));
    gpuErrchk(cudaFree(m_objGPUInst.d_numNeighbors_query_temp_alt));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighborsDist_query));
    gpuErrchk(cudaFree(m_objGPUInst.d_neighborsDist_query_aux));
    gpuErrchk(cudaFree(m_objGPUInst.d_processed_bit_vec));
    gpuErrchk(cudaFree(m_objGPUInst.d_nextIter));
    gpuErrchk(cudaFree(m_objGPUInst.d_iter));
    gpuErrchk(cudaFree(m_objGPUInst.d_mark));
    gpuErrchk(cudaFree(m_objGPUInst.d_mark_alt));
    gpuErrchk(cudaFree(m_objGPUInst.d_expansion_upload));
    gpuErrchk(cudaFree(m_objGPUInst.d_expansion_upload_alt));
    gpuErrchk(cudaFree(m_objGPUInst.d_FPSetCoordsList_Counts));
    gpuErrchk(cudaFree(m_objGPUInst.d_FPSetCoordsList));
    gpuErrchk(cudaFree(m_objGPUInst.d_L2distances));
    gpuErrchk(cudaFree(m_objGPUInst.d_L2ParentIds));
    gpuErrchk(cudaFree(m_objGPUInst.d_L2distances_aux));
    gpuErrchk(cudaFree(m_objGPUInst.d_L2ParentIds_aux));

    cudaEventDestroy(m_objGPUInst.eventChildrenReady);
    cudaEventDestroy(m_objGPUInst.eventChildrenReadyAlt);
    cudaEventDestroy(m_objGPUInst.eventKernelsReady);
    if (m_objGPUInst.eventParentCopyDone)
    {
        gpuErrchk(cudaEventDestroy(m_objGPUInst.eventParentCopyDone));
        m_objGPUInst.eventParentCopyDone = nullptr;
    }
    cudaStreamDestroy(m_objGPUInst.streamFPTransfers);
    cudaStreamDestroy(m_objGPUInst.streamParent);
    cudaStreamDestroy(m_objGPUInst.streamChildren);
    cudaStreamDestroy(m_objGPUInst.streamKernels);

    gpuErrchk(cudaFreeHost(m_objHostInst.neighbors));
    gpuErrchk(cudaFreeHost(m_objHostInst.numNeighbors_query));
    gpuErrchk(cudaFreeHost(m_objHostInst.expansion_upload));
    if (m_objHostInst.expansion_upload_alt)
    {
        gpuErrchk(cudaFreeHost(m_objHostInst.expansion_upload_alt));
        m_objHostInst.expansion_upload_alt = nullptr;
    }
    gpuErrchk(cudaFreeHost(m_objHostInst.parents));
    gpuErrchk(cudaFreeHost(m_objHostInst.FPSetCoordsList));
}

template <typename T> void PHMSSearchInner<T>::phms_unload()
{
    int dev_id;
    cudaGetDevice(&dev_id);
    bool free_per_gpu = false;
    {
        std::lock_guard<std::mutex> lock(g_per_gpu_mtx);
        auto& cache = g_per_gpu_cache[dev_id];
        cache.refcnt--;
        if (cache.refcnt == 0)
        {
            free_per_gpu = true;
        }
    }

    if (free_per_gpu)
    {
        cudaSetDevice(dev_id);
        if (g_per_gpu_cache[dev_id].d_compressedVectors)
        {
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].gc_bits));
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].gc_prefix));
            g_per_gpu_cache[dev_id].gc_bits = nullptr;
            g_per_gpu_cache[dev_id].gc_prefix = nullptr;
            const uint64_t* no_bits = nullptr;
            const unsigned* no_prefix = nullptr;
            gpuErrchk(cudaMemcpyToSymbol(phms_gc_bits, &no_bits, sizeof(no_bits)));
            gpuErrchk(cudaMemcpyToSymbol(phms_gc_prefix, &no_prefix, sizeof(no_prefix)));
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].d_compressedVectors));
            g_per_gpu_cache[dev_id].d_compressedVectors = nullptr;
        }
        if (g_per_gpu_cache[dev_id].d_pqTable)
        {
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].d_pqTable));
            g_per_gpu_cache[dev_id].d_pqTable = nullptr;
        }
        if (g_per_gpu_cache[dev_id].d_chunksOffset)
        {
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].d_chunksOffset));
            g_per_gpu_cache[dev_id].d_chunksOffset = nullptr;
        }
        if (g_per_gpu_cache[dev_id].d_centroid)
        {
            gpuErrchk(cudaFree(g_per_gpu_cache[dev_id].d_centroid));
            g_per_gpu_cache[dev_id].d_centroid = nullptr;
        }
        printf("[NGPU] GPU %d cache freed\n", dev_id);
    }

    std::lock_guard<std::mutex> lk(g_phms_index_mtx);
    if (!g_phms_index_loaded)
        return;
    g_phms_index_refcnt--;
    if (g_phms_index_refcnt > 0)
        return;

    printf("Phms Unload (shared)\n");
#ifdef PHMS_INDEX_LAYOUT_EXPERIMENT
    if (g_phms_index_cache.pIndex == g_thp_index)
    {
        munmap(g_thp_index, g_thp_index_bytes);
        g_thp_index = nullptr;
        g_thp_index_bytes = 0;
    }
    else
    {
#endif
#ifdef _HUGEPAGES
        munmap(g_phms_index_cache.pIndex, g_phms_index_cache.size_indexfile);
#else
    free(g_phms_index_cache.pIndex);
#endif
#ifdef PHMS_INDEX_LAYOUT_EXPERIMENT
    }
#endif

    if (g_phms_index_cache.h_pqTable_hostcopy)
    {
        free(g_phms_index_cache.h_pqTable_hostcopy);
        g_phms_index_cache.h_pqTable_hostcopy = NULL;
    }
    if (g_phms_index_cache.h_centroid_hostcopy)
    {
        free(g_phms_index_cache.h_centroid_hostcopy);
        g_phms_index_cache.h_centroid_hostcopy = NULL;
    }
    if (g_phms_index_cache.h_chunksOffset_hostcopy)
    {
        free(g_phms_index_cache.h_chunksOffset_hostcopy);
        g_phms_index_cache.h_chunksOffset_hostcopy = NULL;
    }

    memset(&g_phms_index_cache, 0, sizeof(IndexLoad));
    g_phms_index_loaded = false;
    g_phms_index_refcnt = 0;
    g_shard_adjacency.reset();
}

template <typename T>
void PHMSSearchInner<T>::phms_set_searchparams(int recall, int worklist_length, DistFunc nDistFunc)
{
    m_objSearchParams.recall = recall;
    m_objSearchParams.worklist_length = worklist_length;
    m_objSearchParams.uDistFunc = nDistFunc;
}

template <typename T>
void PHMSSearchInner<T>::phms_query(T* queriesFP, int numQueries, result_ann_t* nearestNeighbours,
                                    float* nearestNeighbours_dist)
{

    T* FPSetCoordsList = (T*)m_objHostInst.FPSetCoordsList;
    T* d_queriesFP = (T*)m_objGPUInst.d_queriesFP;
    T* d_FPSetCoordsList = (T*)m_objGPUInst.d_FPSetCoordsList;

#ifdef _TIMERS
    double time_transfer = 0.0f;
    double time_K1 = 0.0f;
    double time_B1 = 0.0f;
    double time_B2 = 0.0f;
    double time_neighbor_filtering = 0.0f;
    double time_prfetch = 0.0f;
    double fp_set_time_gpu = 0.0f;
    double seek_neighbours_time = 0.0f;
    vector<double> time_B1_vec;
    vector<double> time_B2_vec;
    time_B2_vec.push_back(0.0);
#endif

    bool nextIter = false;
    unsigned iter = 1;

#ifdef _TIMERS
    GPUTimer gputimer(m_objGPUInst.streamKernels, false);
#endif

    const unsigned uMAX_PARENTS_PERQUERY =
        (m_objSearchParams.worklist_length + NAX_EXTRA_ITERATION);

    omp_set_num_threads(m_objHostInst.numCPUthreads);

#ifdef _TIMERS
    auto milliStart = log_message("SEARCH STARTED");
    auto start = std::chrono::high_resolution_clock::now();
#endif
    gpuErrchk(cudaMemcpy(d_queriesFP, queriesFP, sizeof(T) * (m_objInputData.D * numQueries),
                         cudaMemcpyHostToDevice));

#ifdef _TIMERS
    auto stop = std::chrono::high_resolution_clock::now();
    time_transfer +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
    gputimer.Start();
#endif

    populate_pqDist_par<<<numQueries, m_objGPUInst.numThreads_K1,
                          (m_objInputData.D * (sizeof(float) + sizeof(T))),
                          m_objGPUInst.streamKernels>>>(
        m_objInputData.d_pqTable, m_objGPUInst.d_pqDistTables, d_queriesFP,
        m_objInputData.d_chunksOffset, m_objInputData.d_centroid, m_objInputData.uChunks,
        m_objInputData.D, m_objSearchParams.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0);
#ifdef _TIMERS
    gputimer.Stop();
    time_K1 += gputimer.Elapsed();
    start = std::chrono::high_resolution_clock::now();
#endif

    gpuErrchk(cudaMemset(m_objGPUInst.d_numNeighbors_query, 0, sizeof(unsigned) * numQueries));

#ifdef _TIMERS
    stop = std::chrono::high_resolution_clock::now();
    time_transfer +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
    gputimer.Start();
#endif

    neighbor_filtering_new<<<numQueries, m_objGPUInst.numThreads_K5, 0,
                             m_objGPUInst.streamKernels>>>(
        m_objGPUInst.d_neighbors, m_objGPUInst.d_neighbors_temp, m_objGPUInst.d_numNeighbors_query,
        m_objGPUInst.d_numNeighbors_query_temp, m_objGPUInst.d_processed_bit_vec, m_objInputData.R,
        0, 0);

#ifdef _TIMERS
    gputimer.Stop();
    time_neighbor_filtering += gputimer.Elapsed();

    gputimer.Start();
#endif

    compute_neighborDist_par<<<numQueries, m_objGPUInst.numThreads_K2, 0,
                               m_objGPUInst.streamKernels>>>(
        m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
        m_objInputData.d_compressedVectors, m_objGPUInst.d_pqDistTables,
        m_objGPUInst.d_neighborsDist_query, m_objInputData.uChunks, m_objInputData.R, 0);

    gpuErrchk(cudaMemcpyAsync(m_objGPUInst.d_iter, &iter, sizeof(unsigned), cudaMemcpyHostToDevice,
                              m_objGPUInst.streamKernels));

#ifdef _TIMERS
    gputimer.Stop();
    time_B1_vec.push_back(gputimer.Elapsed());
    gputimer.Start();
#endif

    compute_parent1<<<(numQueries + m_objGPUInst.numThreads_K4 - 1) / m_objGPUInst.numThreads_K4,
                      m_objGPUInst.numThreads_K4, 0, m_objGPUInst.streamKernels>>>(
        m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
        m_objGPUInst.d_neighborsDist_query, m_objGPUInst.d_BestLSets, m_objGPUInst.d_BestLSetsDist,
        m_objGPUInst.d_BestLSets_visited, m_objGPUInst.d_parents, m_objGPUInst.d_nextIter,
        m_objGPUInst.d_BestLSets_count, m_objGPUInst.d_mark, m_objGPUInst.d_iter,
        m_objGPUInst.d_L2ParentIds, m_objGPUInst.d_FPSetCoordsList_Counts, numQueries,
        m_objInputData.MEDOID, m_objInputData.R);
    cudaStreamSynchronize(m_objGPUInst.streamKernels);
#ifdef _TIMERS
    gputimer.Stop();
    time_prfetch += gputimer.Elapsed();
#endif
    do
    {

#ifdef _TIMERS
        start = std::chrono::high_resolution_clock::now();
#endif

        gpuErrchk(cudaMemcpyAsync(m_objHostInst.parents, m_objGPUInst.d_parents,
                                  sizeof(unsigned) * ((SIZEPARENTLIST)*numQueries),
                                  cudaMemcpyDeviceToHost, m_objGPUInst.streamParent));

#ifdef _TIMERS
        stop = std::chrono::high_resolution_clock::now();
        time_transfer +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
        ;
        gputimer.Start();
#endif

#ifdef _NO_PRETECH
        if (iter == 1)
#endif
        {
            compute_BestLSets_par_sort_msort<<<numQueries, m_objGPUInst.numThreads_K3, 0,
                                               m_objGPUInst.streamKernels>>>(
                m_objGPUInst.d_neighbors, m_objGPUInst.d_neighbors_aux,
                m_objGPUInst.d_numNeighbors_query, m_objGPUInst.d_neighborsDist_query,
                m_objGPUInst.d_neighborsDist_query_aux, m_objGPUInst.d_nextIter, m_objInputData.R);

            compute_BestLSets_par_merge<<<numQueries, m_objGPUInst.numThreads_K3_merge,
                                          (m_objInputData.R + 1) * sizeof(float),
                                          m_objGPUInst.streamKernels>>>(
                m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
                m_objGPUInst.d_neighborsDist_query, m_objGPUInst.d_BestLSets,
                m_objGPUInst.d_BestLSetsDist, m_objGPUInst.d_BestLSets_visited,
                m_objGPUInst.d_parents, iter, m_objGPUInst.d_nextIter,
                m_objGPUInst.d_BestLSets_count, m_objGPUInst.d_mark,
                m_objSearchParams.worklist_length, m_objInputData.MEDOID, m_objInputData.R);
        }

#ifdef _TIMERS
        gputimer.Stop();
        time_B2_vec.push_back(gputimer.Elapsed());
        start = std::chrono::high_resolution_clock::now();
#endif
        memset(m_objHostInst.numNeighbors_query, 0, sizeof(unsigned) * numQueries);

        cudaStreamSynchronize(m_objGPUInst.streamParent);

#ifdef _TIMERS
        stop = std::chrono::high_resolution_clock::now();
        time_transfer +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
        start = std::chrono::high_resolution_clock::now();
#endif
        unsigned offset = (m_objInputData.R + 1);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();

            const uint64_t entryLen = m_objInputData.ullIndex_Entry_LEN;
            uint8_t* const pIndex = m_objInputData.pIndex;

            const unsigned D = (unsigned)m_objInputData.D;
            const size_t fpBytes = (size_t)D * sizeof(T);

            const int QBLK = 128;
            const int PF_LINES = 8;
            uint8_t* src_list[QBLK];
            int q_list[QBLK];
            int cnt = 0;

            for (int base = tid * QBLK; base < numQueries; base += nth * QBLK)
            {
                cnt = 0;

                int end = base + QBLK;
                if (end > numQueries)
                    end = numQueries;

                for (int q = base; q < end; ++q)
                {
                    unsigned numParents = m_objHostInst.parents[(size_t)q * SIZEPARENTLIST];
                    if (numParents == 0)
                        continue;

                    unsigned parent = m_objHostInst.parents[(size_t)q * SIZEPARENTLIST + 1];
                    uint8_t* src = pIndex + entryLen * (uint64_t)parent;

                    src_list[cnt] = src;
                    q_list[cnt] = q;
                    cnt++;

#pragma unroll
                    for (int l = 0; l < PF_LINES; ++l)
                    {
                        __builtin_prefetch(src + (size_t)l * 64, 0, 3);
                    }
                }

                const unsigned offset = (m_objInputData.R + 1);

                for (int k = 0; k < cnt; ++k)
                {
                    const int q = q_list[k];
                    uint8_t* src = src_list[k];

                    const unsigned offset_neighbors = (unsigned)q * offset;

                    memcpy(FPSetCoordsList + (iter * m_objHostInst.FPSetCoords_rowsize) +
                               ((size_t)q * m_objHostInst.FPSetCoords_size),
                           src, fpBytes);

                    unsigned* puNumNeighbours = (unsigned*)(src + fpBytes);
                    unsigned deg = *puNumNeighbours;

                    if (deg > m_objInputData.R)
                        deg = m_objInputData.R;
                    memcpy(m_objHostInst.neighbors + offset_neighbors, puNumNeighbours + 1,
                           (size_t)deg * sizeof(unsigned));
                    m_objHostInst.numNeighbors_query[q] = deg;
                }
            }
        }

#ifdef _TIMERS
        auto stop = std::chrono::high_resolution_clock::now();

        seek_neighbours_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;

        start = std::chrono::high_resolution_clock::now();
#endif
        gpuErrchk(cudaMemcpyAsync(m_objGPUInst.d_neighbors_temp, m_objHostInst.neighbors,
                                  sizeof(unsigned) * numQueries * (m_objInputData.R + 1),
                                  cudaMemcpyHostToDevice, m_objGPUInst.streamChildren));

        gpuErrchk(cudaMemcpyAsync(m_objGPUInst.d_numNeighbors_query_temp,
                                  m_objHostInst.numNeighbors_query, sizeof(unsigned) * (numQueries),
                                  cudaMemcpyHostToDevice, m_objGPUInst.streamChildren));

        cudaMemcpyAsync(d_FPSetCoordsList + (iter * m_objHostInst.FPSetCoords_rowsize),
                        FPSetCoordsList + (iter * m_objHostInst.FPSetCoords_rowsize),
                        m_objHostInst.FPSetCoords_rowsize_bytes, cudaMemcpyHostToDevice,
                        m_objGPUInst.streamFPTransfers);

#ifdef _NO_ASYNC_FP
        cudaStreamSynchronize(m_objGPUInst.streamFPTransfers);
#endif

        gpuErrchk(cudaMemsetAsync(m_objGPUInst.d_numNeighbors_query, 0,
                                  sizeof(unsigned) * numQueries, m_objGPUInst.streamChildren));
        cudaStreamSynchronize(m_objGPUInst.streamChildren);
#ifdef _TIMERS
        stop = std::chrono::high_resolution_clock::now();
        time_transfer +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;

        gputimer.Start();
#endif

        neighbor_filtering_new<<<numQueries, m_objGPUInst.numThreads_K5, 0,
                                 m_objGPUInst.streamKernels>>>(
            m_objGPUInst.d_neighbors, m_objGPUInst.d_neighbors_temp,
            m_objGPUInst.d_numNeighbors_query, m_objGPUInst.d_numNeighbors_query_temp,
            m_objGPUInst.d_processed_bit_vec, m_objInputData.R, 0, 0);
#ifdef _TIMERS
        gputimer.Stop();
        time_neighbor_filtering += gputimer.Elapsed();

        gputimer.Start();
#endif
        gpuErrchk(cudaMemsetAsync(m_objGPUInst.d_neighborsDist_query, 0,
                                  sizeof(float) * (numQueries * (m_objInputData.R + 1)),
                                  m_objGPUInst.streamKernels));

        compute_neighborDist_par<<<numQueries, m_objGPUInst.numThreads_K2, 0,
                                   m_objGPUInst.streamKernels>>>(
            m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
            m_objInputData.d_compressedVectors, m_objGPUInst.d_pqDistTables,
            m_objGPUInst.d_neighborsDist_query, m_objInputData.uChunks, m_objInputData.R, 0);

#ifdef _TIMERS
        gputimer.Stop();
        time_B1_vec.push_back(gputimer.Elapsed());
        gputimer.Start();
#endif
        ++iter;

        gpuErrchk(cudaMemcpyAsync(m_objGPUInst.d_iter, &iter, sizeof(unsigned),
                                  cudaMemcpyHostToDevice, m_objGPUInst.streamKernels));

#ifdef _NO_PRETECH
        compute_BestLSets_par_sort_msort<<<numQueries, m_objGPUInst.numThreads_K3, 0,
                                           m_objGPUInst.streamKernels>>>(
            m_objGPUInst.d_neighbors, m_objGPUInst.d_neighbors_aux,
            m_objGPUInst.d_numNeighbors_query, m_objGPUInst.d_neighborsDist_query,
            m_objGPUInst.d_neighborsDist_query_aux, m_objGPUInst.d_nextIter, m_objInputData.R);

        compute_BestLSets_par_merge<<<numQueries, m_objGPUInst.numThreads_K3_merge,
                                      (m_objInputData.R + 1) * sizeof(float),
                                      m_objGPUInst.streamKernels>>>(
            m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
            m_objGPUInst.d_neighborsDist_query, m_objGPUInst.d_BestLSets,
            m_objGPUInst.d_BestLSetsDist, m_objGPUInst.d_BestLSets_visited, m_objGPUInst.d_parents,
            iter, m_objGPUInst.d_nextIter, m_objGPUInst.d_BestLSets_count, m_objGPUInst.d_mark,
            m_objSearchParams.worklist_length, m_objInputData.MEDOID, m_objInputData.R);
#endif

        compute_parent2<<<(numQueries + m_objGPUInst.numThreads_K4 - 1) /
                              m_objGPUInst.numThreads_K4,
                          m_objGPUInst.numThreads_K4, 0, m_objGPUInst.streamKernels>>>(
            m_objGPUInst.d_neighbors, m_objGPUInst.d_numNeighbors_query,
            m_objGPUInst.d_neighborsDist_query, m_objGPUInst.d_BestLSets,
            m_objGPUInst.d_BestLSetsDist, m_objGPUInst.d_BestLSets_visited, m_objGPUInst.d_parents,
            m_objGPUInst.d_nextIter, m_objGPUInst.d_BestLSets_count, m_objGPUInst.d_mark,
            m_objGPUInst.d_iter, m_objGPUInst.d_L2ParentIds, m_objGPUInst.d_FPSetCoordsList_Counts,
            numQueries, m_objSearchParams.worklist_length, m_objInputData.MEDOID, m_objInputData.R);

#ifdef _TIMERS
        gputimer.Stop();
        time_prfetch += gputimer.Elapsed();
        start = std::chrono::high_resolution_clock::now();
#endif
        gpuErrchk(cudaMemcpyAsync(&nextIter, m_objGPUInst.d_nextIter, sizeof(bool),
                                  cudaMemcpyDeviceToHost, m_objGPUInst.streamKernels));
        cudaStreamSynchronize(m_objGPUInst.streamKernels);

#ifdef _TIMERS
        stop = std::chrono::high_resolution_clock::now();
        time_transfer +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
#endif

        if (iter == uMAX_PARENTS_PERQUERY - 1)
        {
#ifdef _DBG_BOUNDS
            printf("Error: Iterations crossed the assumed limit. FPSetCoords size overrun \n");
#endif
            break;
        }
    } while (nextIter);

#ifdef _TIMERS
    gputimer.Start();
#endif
    cudaStreamSynchronize(m_objGPUInst.streamFPTransfers);

    compute_L2Dist<<<numQueries, m_objGPUInst.K4_blockSize, m_objInputData.D * sizeof(T)>>>(
        d_FPSetCoordsList, m_objGPUInst.d_FPSetCoordsList_Counts, d_queriesFP,
        m_objGPUInst.d_L2ParentIds, m_objGPUInst.d_L2distances, numQueries, m_objInputData.D,
        m_objSearchParams.uDistFunc == ENUM_DIST_MIPS ? MIPS_EXTRA_DIM : 0);

    compute_NearestNeighbours<<<numQueries, MAX_PARENTS_PERQUERY>>>(
        m_objGPUInst.d_L2ParentIds, m_objGPUInst.d_L2ParentIds_aux,
        m_objGPUInst.d_FPSetCoordsList_Counts, m_objGPUInst.d_L2distances,
        m_objGPUInst.d_L2distances_aux, m_objGPUInst.d_nearestNeighbours, numQueries,
        m_objSearchParams.recall);

#ifdef _TIMERS
    gputimer.Stop();

    fp_set_time_gpu += gputimer.Elapsed();

    start = std::chrono::high_resolution_clock::now();
#endif
    gpuErrchk(cudaMemcpy(nearestNeighbours, m_objGPUInst.d_nearestNeighbours,
                         sizeof(result_ann_t) * (m_objSearchParams.recall * numQueries),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(nearestNeighbours_dist, m_objGPUInst.d_L2distances,
                         sizeof(float) * (m_objSearchParams.recall * numQueries),
                         cudaMemcpyDeviceToHost));

#ifdef _TIMERS
    stop = std::chrono::high_resolution_clock::now();
    time_transfer +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
#endif
#ifdef _TIMERS
    auto milliEnd = log_message("SEARCH END");
#endif

#ifdef _TIMERS
    assert(time_B1_vec.size() >= 1);
    float time_B1_avg = time_B1_vec[0];
    time_B1 = time_B1_avg;
    for (unsigned idx = 1; idx < time_B1_vec.size(); ++idx)
    {
        time_B1_avg = time_B1_avg + (time_B1_vec[idx] - time_B1_avg) / (idx + 1);
        time_B1 += time_B1_vec[idx];
    }

    assert(time_B2_vec.size() >= 1);
    float time_B2_avg = time_B2_vec[0];
    time_B2 = time_B2_avg;
    for (unsigned idx = 1; idx < time_B2_vec.size(); ++idx)
    {
        time_B2_avg = time_B2_avg + (time_B2_vec[idx] - time_B2_avg) / (idx + 1);
        time_B2 += time_B2_vec[idx];
    }

    cout << "STATS:" << endl;
    cout << "Total Search iterations = " << iter << endl;
    cout << "(1) PD Dist Table Construction = " << time_K1 << " ms" << endl;
    cout << "(2) Distance Computations = " << time_B1 << " ms" << endl;
    ;
    cout << "(3) Sort and Merge = " << time_B2 << " ms" << endl;
    cout << "(4) total neighbor_filtering_time = " << time_neighbor_filtering << " ms" << endl;
    cout << "(5) Pre-fetch time = " << time_prfetch << " ms" << endl;
    cout << "(6) Time elapsed in L2 Dist computation (GPU)= " << fp_set_time_gpu << " ms" << endl;

    cout << "(7) total transfer_time (CPU <--> GPU) = " << time_transfer / 1000 << " ms" << endl;
    cout << "(8) total neigbbour seek time = " << seek_neighbours_time / 1000 << " ms" << endl;

    double totalTime = time_K1 + time_B1 + time_neighbor_filtering + (time_transfer / 1000) +
                       (seek_neighbours_time / 1000) + time_prfetch + fp_set_time_gpu;
    cout << "Total time from timers = (1) + (2) + (4) + (5) + (6) + (8) = " << totalTime << " ms"
         << endl;

    double totalTime_wallclock = milliEnd - milliStart;
    double throughput = (numQueries * 1000.0) / totalTime_wallclock;
    cout << "Wall Clock Time = " << totalTime_wallclock << endl;
    cout << "Throughput = " << throughput << " QPS" << endl;
    cout << "Throughput (Exclude Mem Transfers) = "
         << (numQueries * 1000.0) / (totalTime_wallclock - time_transfer / 1000) << " QPS" << endl;
#endif

#ifdef _DBG_CAND
    unsigned total_size = 0;
    unsigned* FPSetCoordsList_Counts_temp = (unsigned*)malloc(sizeof(unsigned) * numQueries);

    gpuErrchk(cudaMemcpy(FPSetCoordsList_Counts_temp, m_objGPUInst.d_FPSetCoordsList_Counts,
                         sizeof(unsigned) * numQueries, cudaMemcpyDeviceToHost));
    for (int i = 0; i < numQueries; i++)
    {
        total_size += FPSetCoordsList_Counts_temp[i];
    }
    free(FPSetCoordsList_Counts_temp);
    cout << "Total candidates " << total_size << endl;
#endif
}

template <typename T>
__global__ void populate_pqDist_par(float* d_pqTable_T, float* d_pqDistTables, T* d_queriesFP,
                                    unsigned* d_chunksOffset, float* d_centroid, unsigned n_chunks,
                                    unsigned long long D, unsigned n_DimAdjust)
{
    extern __shared__ char array[];
    T* query_vec = (T*)array;
    float* shm_centroid = (float*)(array + sizeof(T) * D);
    unsigned queryID = blockIdx.x;
    unsigned pqDistTables_offset = queryID * 256 * n_chunks;

    unsigned gid = queryID * (D - n_DimAdjust);
    unsigned tid = threadIdx.x;

    for (unsigned i = tid; i < D - (n_DimAdjust); i += blockDim.x)
    {
        query_vec[i] = d_queriesFP[gid + i];
        shm_centroid[i] = d_centroid[i];
    }

    if (tid >= D - (n_DimAdjust))
    {
        for (unsigned i = tid; i < D; i += blockDim.x)
        {
            query_vec[i] = 0;
            shm_centroid[i] = d_centroid[i];
        }
    }

    __syncthreads();

    for (unsigned chunk = 0; chunk < n_chunks; chunk++)
    {
        unsigned chunk_start = pqDistTables_offset + (256 * chunk);
        for (unsigned j = d_chunksOffset[chunk]; j < d_chunksOffset[chunk + 1]; j++)
        {
            const float* centers_dim_vec = d_pqTable_T + (256 * j);
            for (unsigned idx = tid; idx < 256; idx += blockDim.x)
            {
                float diff = centers_dim_vec[idx] - ((T)query_vec[j] - shm_centroid[j]);

                d_pqDistTables[chunk_start + idx] += (float)(diff * diff);
            }
        }
    }
}

__global__ void neighbor_filtering_new(unsigned* d_neighbors, unsigned* d_neighbors_temp,
                                       unsigned* d_numNeighbors_query,
                                       unsigned* d_numNeighbors_query_temp,
                                       bool* d_processed_bit_vec, unsigned R, unsigned uShardStart,
                                       unsigned uShardEnd)
{
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned offset_neighbors = queryID * (R + 1);
    unsigned offset_bit_vec = queryID * BF_MEMORY;
    unsigned numNeighbors = d_numNeighbors_query_temp[queryID];
    bool* d_processed_bit_vec_start = d_processed_bit_vec + offset_bit_vec;

    for (unsigned ii = tid; ii < numNeighbors; ii += blockDim.x)
    {
        unsigned nbr = d_neighbors_temp[offset_neighbors + ii];

        bool bInShard = phms_gc_member(nbr, uShardStart, uShardEnd);
        if (!bInShard)
            continue;

        if (!(phms_bloom_test(d_processed_bit_vec_start, hashFn1_d(nbr)) &&
              phms_bloom_test(d_processed_bit_vec_start, hashFn2_d(nbr))))
        {
            phms_bloom_set(d_processed_bit_vec_start, hashFn1_d(nbr));
            phms_bloom_set(d_processed_bit_vec_start, hashFn2_d(nbr));
            unsigned old = atomicAdd(&d_numNeighbors_query[queryID], 1);
            d_neighbors[offset_neighbors + old] = nbr;
        }
    }
}

__device__ unsigned hashFn1_d(unsigned x)
{

    uint64_t hash = 0xcbf29ce4;
    hash = (hash ^ (x & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;

    return hash % (BF_ENTRIES);
}

__device__ unsigned hashFn2_d(unsigned x)
{

    uint64_t hash = 0x84222325;
    hash = (hash ^ (x & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
    return hash % (BF_ENTRIES);
}

__global__ void compute_neighborDist_par(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                         uint8_t* d_compressedVectors, float* d_pqDistTables,
                                         float* d_neighborsDist_query, unsigned n_chunks,
                                         unsigned R, unsigned uShardStart)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    unsigned pqDistTables_offset = queryID * 256 * n_chunks;
    float* d_pqDistTables_start = d_pqDistTables + pqDistTables_offset;

    unsigned queryNeighbors_offset = queryID * (R + 1);
    unsigned* d_neighbors_start = d_neighbors + queryNeighbors_offset;
    float* d_neighborsDist_query_start = d_neighborsDist_query + queryNeighbors_offset;
    uint8_t* d_compressedVectors_start = NULL;

#define THREADS_PER_NEIGHBOR 8
    typedef cub::WarpReduce<float, THREADS_PER_NEIGHBOR> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[MAX_R + 1];

    for (unsigned j = tid / THREADS_PER_NEIGHBOR; j < numNeighbors;
         j += (blockDim.x) / THREADS_PER_NEIGHBOR)
    {
        unsigned nid = d_neighbors_start[j];

        unsigned long long cv_offset = phms_gc_row(nid, uShardStart) * n_chunks;
        d_compressedVectors_start = d_compressedVectors + cv_offset;
        float sum = 0.0f;

        for (unsigned long long i = tid % THREADS_PER_NEIGHBOR; i < n_chunks;
             i += THREADS_PER_NEIGHBOR)
        {
            sum += d_pqDistTables_start[(i * 256) + d_compressedVectors_start[i]];
        }

        float computed = WarpReduce(temp_storage[j]).Sum(sum);
        if (tid % THREADS_PER_NEIGHBOR == 0)
            d_neighborsDist_query_start[j] = computed;
    }
}

__global__ void neighbor_filtering_distance_fused(
    unsigned* d_neighbors, unsigned* d_neighbors_temp, unsigned* d_numNeighbors_query,
    unsigned* d_numNeighbors_query_temp, bool* d_processed_bit_vec, uint8_t* d_compressedVectors,
    float* d_pqDistTables, float* d_neighborsDist_query, unsigned n_chunks, unsigned R,
    unsigned uShardStart, unsigned uShardEnd, const unsigned* input_offsets,
    const unsigned* input_masks)
{
    const unsigned tid = threadIdx.x;
    const unsigned queryID = blockIdx.x;
    const unsigned row_offset = queryID * (R + 1);
    const unsigned input_count = d_numNeighbors_query_temp[queryID];

    __shared__ unsigned filtered_count;
    __shared__ unsigned filtered_neighbors[MAX_R + 1];
    if (tid == 0)
        filtered_count = 0;
    __syncthreads();

    bool* processed = d_processed_bit_vec + queryID * BF_MEMORY;
    const unsigned long long mask =
        input_masks ? ((unsigned long long)input_masks[queryID] |
                       ((unsigned long long)input_masks[gridDim.x + queryID] << 32))
                    : 0;
    __shared__ unsigned input_ids[MAX_R + 1], h1s[MAX_R + 1], h2s[MAX_R + 1];
    __shared__ unsigned active[MAX_R + 1], keep[MAX_R + 1], conflict;
    if (tid == 0)
        conflict = 0;
    for (unsigned i = tid; i < input_count; i += blockDim.x)
    {
        active[i] = 0;
        keep[i] = 0;
        if (input_offsets && !(mask & (1ull << i)))
            continue;
        const unsigned index = input_offsets
                                   ? input_offsets[queryID] + __popcll(mask & ((1ull << i) - 1))
                                   : row_offset + i;
        const unsigned id = d_neighbors_temp[index];
        if (!phms_gc_member(id, uShardStart, uShardEnd))
            continue;
        active[i] = 1;
        input_ids[i] = id;
        h1s[i] = hashFn1_d(id);
        h2s[i] = hashFn2_d(id);
    }
    __syncthreads();
    for (unsigned i = tid; i < input_count; i += blockDim.x)
        if (active[i])
        {
            for (unsigned j = 0; j < i; ++j)
                if (active[j] &&
                    (h1s[i] == h1s[j] || h1s[i] == h2s[j] || h2s[i] == h1s[j] || h2s[i] == h2s[j]))
                {
                    atomicExch(&conflict, 1u);
                    break;
                }
        }
    __syncthreads();
    if (conflict)
    {
        if (tid == 0)
            for (unsigned i = 0; i < input_count; ++i)
                if (active[i])
                {
                    if (!(phms_bloom_test(processed, h1s[i]) && phms_bloom_test(processed, h2s[i])))
                    {
                        keep[i] = 1;
                        phms_bloom_set(processed, h1s[i]);
                        phms_bloom_set(processed, h2s[i]);
                    }
                }
    }
    else
    {
        for (unsigned i = tid; i < input_count; i += blockDim.x)
            if (active[i])
            {
                const unsigned* words = reinterpret_cast<const unsigned*>(processed);
                bool first = (words[h1s[i] >> 5] & (1u << (h1s[i] & 31))) != 0;
                bool second = (words[h2s[i] >> 5] & (1u << (h2s[i] & 31))) != 0;
                keep[i] = !(first && second);
            }
        __syncthreads();
        for (unsigned i = tid; i < input_count; i += blockDim.x)
            if (keep[i])
            {
                phms_bloom_set(processed, h1s[i]);
                phms_bloom_set(processed, h2s[i]);
            }
    }
    __syncthreads();
    for (unsigned i = tid; i < input_count; i += blockDim.x)
        if (keep[i])
        {
            unsigned slot = 0;
            for (unsigned j = 0; j < i; ++j)
                slot += keep[j];
            filtered_neighbors[slot] = input_ids[i];
        }
    if (tid == 0)
    {
        unsigned n = 0;
        for (unsigned i = 0; i < input_count; ++i)
            n += keep[i];
        filtered_count = n;
    }
    __syncthreads();

    const unsigned count = filtered_count;
    if (tid == 0)
        d_numNeighbors_query[queryID] = count;
    for (unsigned i = tid; i < count; i += blockDim.x)
        d_neighbors[row_offset + i] = filtered_neighbors[i];

#define FUSED_THREADS_PER_NEIGHBOR 8
    typedef cub::WarpReduce<float, FUSED_THREADS_PER_NEIGHBOR> FusedWarpReduce;
    __shared__ typename FusedWarpReduce::TempStorage reduce_storage[MAX_R + 1];

    float* pq_table = d_pqDistTables + queryID * 256 * n_chunks;
    float* output_distances = d_neighborsDist_query + row_offset;
    for (unsigned j = tid / FUSED_THREADS_PER_NEIGHBOR; j < count;
         j += blockDim.x / FUSED_THREADS_PER_NEIGHBOR)
    {
        const unsigned neighbor = filtered_neighbors[j];
        const unsigned long long code_offset = phms_gc_row(neighbor, uShardStart) * n_chunks;
        const uint8_t* codes = d_compressedVectors + code_offset;
        float sum = 0.0f;
        for (unsigned i = tid % FUSED_THREADS_PER_NEIGHBOR; i < n_chunks;
             i += FUSED_THREADS_PER_NEIGHBOR)
        {
            sum += pq_table[i * 256 + codes[i]];
        }
        const float distance = FusedWarpReduce(reduce_storage[j]).Sum(sum);
        if (tid % FUSED_THREADS_PER_NEIGHBOR == 0)
            output_distances[j] = distance;
    }
#undef FUSED_THREADS_PER_NEIGHBOR
}

__global__ void ngpu_expansion_fused(unsigned* d_neighbors, unsigned* d_neighbors_input,
                                     unsigned* d_numNeighbors_query, unsigned* d_counts_input,
                                     float* d_neighborsDist_query, unsigned* d_BestLSets,
                                     float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                     unsigned* d_BestLSets_count, unsigned* d_mark, unsigned iter,
                                     unsigned uWLLen, unsigned long long MEDOID,
                                     bool* d_processed_bit_vec, uint8_t* d_compressedVectors,
                                     float* d_pqDistTables, unsigned n_chunks, unsigned R,
                                     unsigned uShardStart, unsigned uShardEnd,
                                     const unsigned* query_seeds)
{
    const unsigned tid = threadIdx.x;
    const unsigned queryID = blockIdx.x;
    if (query_seeds)
        MEDOID = query_seeds[queryID];
    const unsigned row_offset = queryID * (R + 1);
    const unsigned best_offset = queryID * uWLLen;

    __shared__ float merge_neighbor_dist[MAX_R + 1];
    __shared__ float merge_current_dist[MAX_L];
    __shared__ float merge_output_dist[MAX_L];
    __shared__ unsigned merge_positions[MAX_R + MAX_L + 1];
    __shared__ unsigned merge_output_ids[MAX_L];
    __shared__ bool merge_output_visited[MAX_L];
    __shared__ unsigned merge_best_size;
    __shared__ unsigned merge_neighbor_bound;
    __shared__ unsigned merge_output_size;

    const unsigned previous_count = d_numNeighbors_query[queryID];
    if (tid == 0)
    {
        merge_best_size = d_BestLSets_count[queryID];
        merge_neighbor_bound = 0;
        merge_output_size = merge_best_size;
        if (previous_count > 0)
        {
            if (iter == 1)
            {
                merge_neighbor_bound = min(previous_count, uWLLen);
                merge_output_size = merge_neighbor_bound;
                d_BestLSets_count[queryID] = merge_output_size;
            }
            else
            {
                const float max_best = merge_best_size
                                           ? d_BestLSetsDist[best_offset + merge_best_size - 1]
                                           : 3.402823E+38f;
                unsigned bound = 0;
                while (bound < min(uWLLen, previous_count) &&
                       d_neighborsDist_query[row_offset + bound] < max_best)
                {
                    ++bound;
                }
                merge_neighbor_bound = max(bound, min(uWLLen - merge_best_size, previous_count));
                merge_output_size = min(merge_best_size + merge_neighbor_bound, uWLLen);
                d_BestLSets_count[queryID] = merge_output_size;
            }
        }
    }
    __syncthreads();

    if (previous_count > 0)
    {
        if (iter == 1)
        {
            for (unsigned i = tid; i < merge_neighbor_bound; i += blockDim.x)
            {
                const unsigned neighbor = d_neighbors[row_offset + i];
                d_BestLSets[best_offset + i] = neighbor;
                d_BestLSetsDist[best_offset + i] = d_neighborsDist_query[row_offset + i];
                d_BestLSets_visited[best_offset + i] = (neighbor == MEDOID);
            }
        }
        else
        {
            for (unsigned i = tid; i < merge_neighbor_bound; i += blockDim.x)
            {
                merge_neighbor_dist[i] = d_neighborsDist_query[row_offset + i];
            }
            for (unsigned i = tid; i < merge_best_size; i += blockDim.x)
                merge_current_dist[i] = d_BestLSetsDist[best_offset + i];
            __syncthreads();

            if (tid < merge_neighbor_bound)
            {
                merge_positions[tid] = lower_bound_d(merge_current_dist, 0, merge_best_size,
                                                     merge_neighbor_dist[tid]) +
                                       tid;
            }
            if (tid >= merge_neighbor_bound && tid < merge_neighbor_bound + merge_best_size)
            {
                const unsigned best_i = tid - merge_neighbor_bound;
                merge_positions[tid] = upper_bound_d(merge_neighbor_dist, 0, merge_neighbor_bound,
                                                     merge_current_dist[best_i]) +
                                       best_i;
            }
            __syncthreads();

            if (tid < merge_neighbor_bound && merge_positions[tid] < merge_output_size)
            {
                const unsigned out = merge_positions[tid];
                merge_output_dist[out] = merge_neighbor_dist[tid];
                merge_output_ids[out] = d_neighbors[row_offset + tid];
                merge_output_visited[out] = false;
            }
            if (tid >= merge_neighbor_bound && tid < merge_neighbor_bound + merge_best_size &&
                merge_positions[tid] < merge_output_size)
            {
                const unsigned best_i = tid - merge_neighbor_bound;
                const unsigned out = merge_positions[tid];
                merge_output_dist[out] = merge_current_dist[best_i];
                merge_output_ids[out] = d_BestLSets[best_offset + best_i];
                merge_output_visited[out] = d_BestLSets_visited[best_offset + best_i];
            }
            __syncthreads();

            for (unsigned i = tid; i < merge_output_size; i += blockDim.x)
            {
                d_BestLSetsDist[best_offset + i] = merge_output_dist[i];
                d_BestLSets[best_offset + i] = merge_output_ids[i];
                d_BestLSets_visited[best_offset + i] = merge_output_visited[i];
            }
        }
    }
    __syncthreads();

    for (unsigned i = tid; i < merge_output_size; i += blockDim.x)
    {
        if (d_mark[queryID] == d_BestLSets[best_offset + i])
            d_BestLSets_visited[best_offset + i] = true;
    }
    __syncthreads();

    __shared__ unsigned filtered_count;
    __shared__ unsigned filtered_neighbors[MAX_R + 1];
    if (tid == 0)
        filtered_count = 0;
    __syncthreads();

    const unsigned input_count = d_counts_input[queryID];
    bool* processed = d_processed_bit_vec + queryID * BF_MEMORY;
    for (unsigned i = tid; i < input_count; i += blockDim.x)
    {
        const unsigned neighbor = d_neighbors_input[row_offset + i];
        const bool in_shard = phms_gc_member(neighbor, uShardStart, uShardEnd);
        if (!in_shard)
            continue;
        const unsigned hash1 = hashFn1_d(neighbor);
        const unsigned hash2 = hashFn2_d(neighbor);
        if (!(phms_bloom_test(processed, hash1) && phms_bloom_test(processed, hash2)))
        {
            phms_bloom_set(processed, hash1);
            phms_bloom_set(processed, hash2);
            const unsigned slot = atomicAdd(&filtered_count, 1u);
            filtered_neighbors[slot] = neighbor;
        }
    }
    __syncthreads();

    const unsigned count = filtered_count;
    if (tid == 0)
        d_numNeighbors_query[queryID] = count;
    for (unsigned i = tid; i < count; i += blockDim.x)
        d_neighbors[row_offset + i] = filtered_neighbors[i];

#define EXPANSION_THREADS_PER_NEIGHBOR 8
    typedef cub::WarpReduce<float, EXPANSION_THREADS_PER_NEIGHBOR> ExpansionWarpReduce;
    __shared__ typename ExpansionWarpReduce::TempStorage expansion_reduce_storage[MAX_R + 1];

    float* pq_table = d_pqDistTables + queryID * 256 * n_chunks;
    for (unsigned j = tid / EXPANSION_THREADS_PER_NEIGHBOR; j < count;
         j += blockDim.x / EXPANSION_THREADS_PER_NEIGHBOR)
    {
        const unsigned neighbor = filtered_neighbors[j];
        const unsigned long long code_offset = phms_gc_row(neighbor, uShardStart) * n_chunks;
        const uint8_t* codes = d_compressedVectors + code_offset;
        float sum = 0.0f;
        for (unsigned i = tid % EXPANSION_THREADS_PER_NEIGHBOR; i < n_chunks;
             i += EXPANSION_THREADS_PER_NEIGHBOR)
        {
            sum += pq_table[i * 256 + codes[i]];
        }
        const float distance = ExpansionWarpReduce(expansion_reduce_storage[j]).Sum(sum);
        if (tid % EXPANSION_THREADS_PER_NEIGHBOR == 0)
            d_neighborsDist_query[row_offset + j] = distance;
    }
#undef EXPANSION_THREADS_PER_NEIGHBOR
}

template <typename T>
__global__ void compute_L2Dist(T* d_FPSetCoordsList, unsigned* d_FPSetCoordsList_Counts,
                               T* d_queriesFP, unsigned* d_L2ParentIds, float* d_L2distances,
                               unsigned d_numQueries, unsigned long long D, unsigned n_DimAdjust)
{
    extern __shared__ char array[];

    T* query_vec = (T*)array;
    unsigned queryID = blockIdx.x;
    unsigned numNodes = d_FPSetCoordsList_Counts[queryID];
    unsigned tid = threadIdx.x;
    unsigned gid = queryID * (D - n_DimAdjust);

    const unsigned long long FPSetCoords_size = D;
    const unsigned long long FPSetCoords_rowsize = FPSetCoords_size * d_numQueries;

    for (unsigned ii = tid; ii < D - (n_DimAdjust); ii += blockDim.x)
    {
        query_vec[ii] = d_queriesFP[gid + ii];
    }

    if (tid >= D - (n_DimAdjust))
    {
        for (unsigned i = tid; i < D; i += blockDim.x)
        {
            query_vec[i] = 0;
        }
    }

    __syncthreads();

    for (unsigned ii = tid; ii < numNodes; ii += blockDim.x)
    {
        float L2Dist = 0.0;
        for (unsigned jj = 0; jj < D; ++jj)
        {
            float diff =
                d_FPSetCoordsList[(FPSetCoords_rowsize * ii) + (queryID * FPSetCoords_size) + jj] -
                query_vec[jj];
            L2Dist = L2Dist + (diff * diff);
        }
        d_L2distances[(d_numQueries * ii) + queryID] = L2Dist;
    }
}

__global__ void compute_NearestNeighbours(unsigned* d_L2ParentIds, unsigned* d_L2ParentIds_aux,
                                          unsigned* d_FPSetCoordsList_Counts, float* d_L2distances,
                                          float* d_L2distances_aux,
                                          result_ann_t* d_nearestNeighbours, unsigned d_numQueries,
                                          unsigned d_recall)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_FPSetCoordsList_Counts[queryID];

    if (tid >= numNeighbors || numNeighbors <= 0)
        return;

    __shared__ unsigned shm_pos[MAX_PARENTS_PERQUERY + 1];

    for (unsigned subArraySize = 2; subArraySize < 2 * numNeighbors; subArraySize *= 2)
    {
        unsigned subArrayID = tid / subArraySize;
        unsigned start = subArrayID * subArraySize;
        unsigned mid = min(start + subArraySize / 2, numNeighbors);
        unsigned end = min(start + subArraySize, numNeighbors);

        if (tid >= start && tid < mid)
        {
            unsigned lowerBound = lower_bound_d_ex(
                &d_L2distances[mid * (d_numQueries)], 0, end - mid,
                d_L2distances[(tid * (d_numQueries)) + queryID], d_numQueries, queryID);
            shm_pos[tid] = lowerBound + tid;
        }

        if (tid >= mid && tid < end)
        {
            unsigned upperBound = upper_bound_d_ex(
                &d_L2distances[start * (d_numQueries)], 0, mid - start,
                d_L2distances[(tid * (d_numQueries)) + queryID], d_numQueries, queryID);
            shm_pos[tid] = start + (upperBound + tid - mid);
        }
        __syncthreads();
        __threadfence_block();

        for (int i = tid; i < numNeighbors; i += blockDim.x)
        {
            d_L2distances_aux[(shm_pos[i] * (d_numQueries)) + queryID] =
                d_L2distances[(i * (d_numQueries)) + queryID];
            d_L2ParentIds_aux[(shm_pos[i] * (d_numQueries)) + queryID] =
                d_L2ParentIds[(i * (d_numQueries)) + queryID];
        }
        __syncthreads();
        for (int i = tid; i < numNeighbors; i += blockDim.x)
        {
            d_L2distances[(i * (d_numQueries)) + queryID] =
                d_L2distances_aux[(i * (d_numQueries)) + queryID];
            d_L2ParentIds[(i * (d_numQueries)) + queryID] =
                d_L2ParentIds_aux[(i * (d_numQueries)) + queryID];
        }
        __syncthreads();
    }
    for (unsigned ii = tid; ii < d_recall; ii += blockDim.x)
    {
        d_nearestNeighbours[((d_recall * queryID)) + ii] =
            d_L2ParentIds[((d_numQueries)*ii) + queryID];
    }
}

__device__ __forceinline__ void ngpu_offer_parent(unsigned candidate_id, float candidate_dist,
                                                  unsigned& id0, float& dist0, unsigned& id1,
                                                  float& dist1)
{
    if (candidate_dist >= 3.402823E+38f)
        return;
    if (candidate_id == id0)
    {
        if (candidate_dist < dist0)
            dist0 = candidate_dist;
        return;
    }
    if (candidate_id == id1)
    {
        if (candidate_dist < dist1)
            dist1 = candidate_dist;
        return;
    }
    if (candidate_dist < dist0)
    {
        id1 = id0;
        dist1 = dist0;
        id0 = candidate_id;
        dist0 = candidate_dist;
    }
    else if (candidate_dist < dist1)
    {
        id1 = candidate_id;
        dist1 = candidate_dist;
    }
}

__global__ void compute_parent1_ngpu2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                      float* d_neighborsDist_query, unsigned* d_ngpu_parents,
                                      bool* d_nextIter, unsigned d_numQueries,
                                      unsigned long long MEDOID, unsigned R,
                                      const unsigned* query_seeds)
{
    const unsigned queryID = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryID >= d_numQueries)
        return;
    if (query_seeds)
        MEDOID = query_seeds[queryID];
    const unsigned offset = queryID * (R + 1);
    const unsigned count_in = d_numNeighbors_query[queryID];
    unsigned ids[NGPU_PARENTS_PER_ROUND];
    float distances[NGPU_PARENTS_PER_ROUND];
    for (unsigned i = 0; i < NGPU_PARENTS_PER_ROUND; ++i)
    {
        ids[i] = INVALID_NODE_ID;
        distances[i] = 3.402823E+38f;
    }
    for (unsigned i = 0; i < count_in; ++i)
    {
        const unsigned id = d_neighbors[offset + i];
        if (id != MEDOID)
            phms_v3::offer_parent(id, d_neighborsDist_query[offset + i], ids, distances);
    }
    unsigned count = 0;
    for (unsigned i = 0; i < NGPU_PARENTS_PER_ROUND; ++i)
        if (distances[i] < 3.402823E+38f)
            ++count;
    const unsigned out = queryID * NGPU_PARENT_RECORD_SIZE;
    d_ngpu_parents[out] = count;
    for (unsigned i = 0; i < count; ++i)
    {
        d_ngpu_parents[out + 1 + 2 * i] = ids[i];
        d_ngpu_parents[out + 2 + 2 * i] = __float_as_uint(distances[i]);
    }
    if (count)
        *d_nextIter = true;
}

__global__ void compute_parent2_ngpu2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                      float* d_neighborsDist_query, unsigned* d_BestLSets,
                                      float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                      unsigned* d_BestLSets_count, unsigned* d_ngpu_parents,
                                      bool* d_nextIter, unsigned d_numQueries, unsigned uWLLen,
                                      unsigned long long MEDOID, unsigned R,
                                      const unsigned* query_seeds)
{
    const unsigned queryID = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryID >= d_numQueries)
        return;
    if (query_seeds)
        MEDOID = query_seeds[queryID];
    unsigned ids[NGPU_PARENTS_PER_ROUND];
    float distances[NGPU_PARENTS_PER_ROUND];
    for (unsigned i = 0; i < NGPU_PARENTS_PER_ROUND; ++i)
    {
        ids[i] = INVALID_NODE_ID;
        distances[i] = 3.402823E+38f;
    }
    const unsigned neighbor_offset = queryID * (R + 1);
    const unsigned count_in = d_numNeighbors_query[queryID];
    for (unsigned i = 0; i < count_in; ++i)
    {
        const unsigned id = d_neighbors[neighbor_offset + i];
        if (id != MEDOID)
            phms_v3::offer_parent(id, d_neighborsDist_query[neighbor_offset + i], ids, distances);
    }
    const unsigned list_offset = queryID * uWLLen;
    const unsigned list_size = d_BestLSets_count[queryID];
    for (unsigned i = 0; i < list_size; ++i)
    {
        if (!d_BestLSets_visited[list_offset + i])
        {
            phms_v3::offer_parent(d_BestLSets[list_offset + i], d_BestLSetsDist[list_offset + i],
                                  ids, distances);
        }
    }
    unsigned count = 0;
    for (unsigned i = 0; i < NGPU_PARENTS_PER_ROUND; ++i)
        if (distances[i] < 3.402823E+38f)
            ++count;
    const unsigned out = queryID * NGPU_PARENT_RECORD_SIZE;
    d_ngpu_parents[out] = count;
    for (unsigned i = 0; i < count; ++i)
    {
        d_ngpu_parents[out + 1 + 2 * i] = ids[i];
        d_ngpu_parents[out + 2 + 2 * i] = __float_as_uint(distances[i]);
    }
    if (count)
        *d_nextIter = true;
}

__global__ void compute_parent2(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                float* d_neighborsDist_query, unsigned* d_BestLSets,
                                float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
                                unsigned* d_mark, unsigned* d_iter, unsigned* d_L2ParentIds,
                                unsigned* d_FPSetCoordsList_Counts, unsigned d_numQueries,
                                unsigned uWLLen, unsigned long long MEDOID, unsigned R)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = (blockIdx.x * blockDim.x) + tid;
    if (queryID >= (d_numQueries))
        return;

    unsigned numNeighbors = d_numNeighbors_query[queryID];
    unsigned Best_L_Set_size = d_BestLSets_count[queryID];

    float dist = 3.402823E+38;
    unsigned index = 0;

    unsigned offset = queryID * (R + 1);

    for (unsigned ii = 0; ii < numNeighbors; ++ii)
    {
        if (d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset + ii] != MEDOID)
        {
            index = offset + ii;
            dist = d_neighborsDist_query[index];
        }
    }
    unsigned parentOffset = queryID * (SIZEPARENTLIST);
    unsigned parentIndex = 0;
    unsigned LOffset = uWLLen * queryID;
    unsigned LIndex = LOffset;
    float selectedDist = 3.402823E+38f;

    for (unsigned ii = 0; ii < Best_L_Set_size; ++ii)
    {
        LIndex = LOffset + ii;
        if (!d_BestLSets_visited[LIndex])
        {
            parentIndex++;
            if (dist < d_BestLSetsDist[LIndex])
            {
                d_parents[parentOffset + parentIndex] = d_neighbors[index];
                if (d_mark != nullptr)
                    d_mark[queryID] = d_neighbors[index];
                selectedDist = dist;
            }
            else
            {
                d_parents[parentOffset + parentIndex] = d_BestLSets[LIndex];
                selectedDist = d_BestLSetsDist[LIndex];
            }
            break;
        }
    }

    if (parentIndex == 0 && dist < d_BestLSetsDist[LOffset + Best_L_Set_size - 1])
    {
        parentIndex++;
        d_parents[parentOffset + parentIndex] = d_neighbors[index];
        if (d_mark != nullptr)
            d_mark[queryID] = d_neighbors[index];
        selectedDist = dist;
    }

    if (selectedDist >= 3.402823E+38f)
    {
        parentIndex = 0;
    }
    d_parents[parentOffset + 2] = __float_as_uint(selectedDist);

    d_parents[parentOffset] = parentIndex;

    if (parentIndex != 0)
    {
        *d_nextIter = true;
        if (d_FPSetCoordsList_Counts != nullptr && d_L2ParentIds != nullptr)
        {
            d_FPSetCoordsList_Counts[queryID]++;
            d_L2ParentIds[((*d_iter) * d_numQueries) + queryID] =
                d_parents[parentOffset + parentIndex];
        }
    }
}

__global__ void compute_parent1(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                float* d_neighborsDist_query, unsigned* d_BestLSets,
                                float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                unsigned* d_parents, bool* d_nextIter, unsigned* d_BestLSets_count,
                                unsigned* d_mark, unsigned* d_iter, unsigned* d_L2ParentIds,
                                unsigned* d_FPSetCoordsList_Counts, unsigned d_numQueries,
                                unsigned long long MEDOID, unsigned R)
{
    unsigned tid = threadIdx.x;
    unsigned queryID = (blockIdx.x * blockDim.x) + tid;
    if (queryID >= d_numQueries)
        return;

    unsigned numNeighbors = d_numNeighbors_query[queryID];

    float dist = 3.402823E+38;
    unsigned index = 0;

    unsigned offset = queryID * (R + 1);
    float selectedDist = 3.402823E+38f;

    for (unsigned ii = 0; ii < numNeighbors; ++ii)
    {
        if (d_neighborsDist_query[offset + ii] < dist && d_neighbors[offset + ii] != MEDOID)
        {
            index = offset + ii;
            dist = d_neighborsDist_query[index];
            selectedDist = dist;
        }
    }

    unsigned parentIndex = 0;

    parentIndex++;
    d_parents[queryID * (SIZEPARENTLIST) + parentIndex] = d_neighbors[index];
    if (d_mark != nullptr)
        d_mark[queryID] = d_neighbors[index];
    if (selectedDist >= 3.402823E+38f)
    {
        parentIndex = 0;
    }
    d_parents[queryID * SIZEPARENTLIST + 2] = __float_as_uint(selectedDist);

    d_parents[queryID * (SIZEPARENTLIST)] = parentIndex;
    {
        *d_nextIter = true;
        if (d_FPSetCoordsList_Counts != nullptr && d_L2ParentIds != nullptr)
        {
            d_FPSetCoordsList_Counts[queryID]++;
            d_L2ParentIds[((*d_iter) * d_numQueries) + queryID] =
                d_parents[queryID * (SIZEPARENTLIST) + parentIndex];
        }
    }
}

__global__ void neighbor_sort_warp_v3(unsigned* neighbors, unsigned* counts, float* distances,
                                      bool* next_iter, unsigned degree, unsigned queries)
{
    const unsigned lane = threadIdx.x & 31u;
    const unsigned query = blockIdx.x * 4u + threadIdx.x / 32u;
    if (query >= queries)
        return;
    const unsigned n = counts[query];
    const size_t base = (size_t)query * (degree + 1);
    if (!lane)
        *next_iter = false;
    if (n > 32)
    {
        if (!lane)
            for (unsigned i = 1; i < n; ++i)
            {
                const float key = distances[base + i];
                const unsigned id = neighbors[base + i];
                unsigned j = i;
                while (j && distances[base + j - 1] > key)
                {
                    distances[base + j] = distances[base + j - 1];
                    neighbors[base + j] = neighbors[base + j - 1];
                    --j;
                }
                distances[base + j] = key;
                neighbors[base + j] = id;
            }
        return;
    }
    float key = lane < n ? distances[base + lane] : __uint_as_float(0x7f800000u);
    unsigned order = lane, id = lane < n ? neighbors[base + lane] : 0;
    for (unsigned size = 2; size <= 32; size <<= 1)
    {
        for (unsigned stride = size / 2; stride; stride >>= 1)
        {
            const float other = __shfl_xor_sync(0xffffffffu, key, stride);
            const unsigned other_order = __shfl_xor_sync(0xffffffffu, order, stride);
            const unsigned other_id = __shfl_xor_sync(0xffffffffu, id, stride);
            const bool take_min = ((lane & size) == 0) == ((lane & stride) == 0);
            const bool other_less = other < key || (other == key && other_order < order);
            const bool self_less = key < other || (key == other && order < other_order);
            if (take_min ? other_less : self_less)
            {
                key = other;
                order = other_order;
                id = other_id;
            }
        }
    }
    if (lane < n)
    {
        neighbors[base + lane] = id;
        distances[base + lane] = key;
    }
}

__global__ void compute_BestLSets_par_sort_msort(unsigned* d_neighbors, unsigned* d_neighbors_aux,
                                                 unsigned* d_numNeighbors_query,
                                                 float* d_neighborsDist_query,
                                                 float* d_neighborsDist_query_aux, bool* d_nextIter,
                                                 unsigned R)
{

    unsigned tid = threadIdx.x;
    unsigned queryID = blockIdx.x;
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    *d_nextIter = false;

    if (tid >= numNeighbors || numNeighbors <= 0)
        return;

    __shared__ unsigned shm_pos[MAX_R + 1];
    unsigned offset = queryID * (MAX_R + 1);

    for (unsigned subArraySize = 2; subArraySize < 2 * numNeighbors; subArraySize *= 2)
    {
        unsigned subArrayID = tid / subArraySize;
        unsigned start = subArrayID * subArraySize;
        unsigned mid = min(start + subArraySize / 2, numNeighbors);
        unsigned end = min(start + subArraySize, numNeighbors);

        if (tid >= start && tid < mid)
        {
            unsigned lowerBound = lower_bound_d(&d_neighborsDist_query[offset + mid], 0, end - mid,
                                                d_neighborsDist_query[offset + tid]);
            shm_pos[tid] = lowerBound + tid;
        }

        if (tid >= mid && tid < end)
        {
            unsigned upperBound = upper_bound_d(&d_neighborsDist_query[offset + start], 0,
                                                mid - start, d_neighborsDist_query[offset + tid]);
            shm_pos[tid] = start + (upperBound + tid - mid);
        }
        __syncthreads();
        __threadfence_block();

        for (int i = tid; i < numNeighbors; i += blockDim.x)
        {
            d_neighborsDist_query_aux[offset + shm_pos[i]] = d_neighborsDist_query[offset + i];
            d_neighbors_aux[offset + shm_pos[i]] = d_neighbors[offset + i];
        }
        __syncthreads();
        for (int i = tid; i < numNeighbors; i += blockDim.x)
        {
            d_neighborsDist_query[offset + i] = d_neighborsDist_query_aux[offset + i];
            d_neighbors[offset + i] = d_neighbors_aux[offset + i];
        }
        __syncthreads();
    }
}

__global__ void compute_BestLSets_par_merge(unsigned* d_neighbors, unsigned* d_numNeighbors_query,
                                            float* d_neighborsDist_query, unsigned* d_BestLSets,
                                            float* d_BestLSetsDist, bool* d_BestLSets_visited,
                                            unsigned* d_parents, unsigned iter, bool* d_nextIter,
                                            unsigned* d_BestLSets_count, unsigned* d_mark,
                                            unsigned uWLLen, unsigned long long MEDOID, unsigned R,
                                            const unsigned* query_seeds)
{
    extern __shared__ char array[];
    float* shm_neighborsDist_query = (float*)array;
    __shared__ float shm_currBestLSetsDist[MAX_L];
    __shared__ float shm_BestLSetsDist[MAX_L];
    __shared__ unsigned shm_pos[MAX_R + MAX_L + 1];
    __shared__ unsigned shm_BestLSets[MAX_L];
    __shared__ bool shm_BestLSets_visited[MAX_L];

    unsigned queryID = blockIdx.x;
    if (query_seeds)
        MEDOID = query_seeds[queryID];
    unsigned numNeighbors = d_numNeighbors_query[queryID];
    unsigned tid = threadIdx.x;
    __shared__ unsigned old_best_count;
    if (tid == 0)
        old_best_count = d_BestLSets_count[queryID];
    __syncthreads();

    unsigned Best_L_Set_size = 0;
    unsigned newBest_L_Set_size = old_best_count;
    unsigned nbrsBound;
    unsigned offset = queryID * (R + 1);

    if (numNeighbors > 0)
    {

        if (iter == 1)
        {
            nbrsBound = min(numNeighbors, uWLLen);
            for (unsigned ii = tid; ii < nbrsBound; ii += blockDim.x)
            {
                unsigned nbr = d_neighbors[offset + ii];
                d_BestLSets[queryID * uWLLen + tid] = nbr;
                d_BestLSetsDist[queryID * uWLLen + tid] = d_neighborsDist_query[offset + ii];
                d_BestLSets_visited[queryID * uWLLen + tid] = (nbr == MEDOID);
            }
            __syncthreads();
            newBest_L_Set_size = nbrsBound;
            d_BestLSets_count[queryID] = nbrsBound;
        }
        else
        {
            Best_L_Set_size = old_best_count;
            float maxBestLSetDist = Best_L_Set_size
                                        ? d_BestLSetsDist[uWLLen * queryID + Best_L_Set_size - 1]
                                        : 3.402823E+38f;
            for (nbrsBound = 0; nbrsBound < min(uWLLen, numNeighbors); ++nbrsBound)
            {
                if (d_neighborsDist_query[offset + nbrsBound] >= maxBestLSetDist)
                {
                    break;
                }
            }

            nbrsBound = max(nbrsBound, min(uWLLen - Best_L_Set_size, numNeighbors));
            newBest_L_Set_size = min(Best_L_Set_size + nbrsBound, uWLLen);

            d_BestLSets_count[queryID] = newBest_L_Set_size;

            for (int i = tid; i < nbrsBound; i += blockDim.x)
            {
                shm_neighborsDist_query[i] = d_neighborsDist_query[offset + i];
            }
            for (int i = tid; i < Best_L_Set_size; i += blockDim.x)
            {
                shm_currBestLSetsDist[i] = d_BestLSetsDist[uWLLen * queryID + i];
            }
            __syncthreads();
            if (tid < nbrsBound)
            {
                shm_pos[tid] = lower_bound_d(shm_currBestLSetsDist, 0, Best_L_Set_size,
                                             shm_neighborsDist_query[tid]) +
                               tid;
            }
            if (tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size))
            {
                shm_pos[tid] = upper_bound_d(shm_neighborsDist_query, 0, nbrsBound,
                                             shm_currBestLSetsDist[tid - nbrsBound]) +
                               (tid - nbrsBound);
            }

            __syncthreads();
            __threadfence_block();

            if (tid < nbrsBound && shm_pos[tid] < newBest_L_Set_size)
            {
                shm_BestLSetsDist[shm_pos[tid]] = shm_neighborsDist_query[tid];
                shm_BestLSets[shm_pos[tid]] = d_neighbors[offset + tid];
                shm_BestLSets_visited[shm_pos[tid]] = false;
            }
            if (tid >= nbrsBound && tid < (nbrsBound + Best_L_Set_size) &&
                shm_pos[tid] < newBest_L_Set_size)
            {
                shm_BestLSetsDist[shm_pos[tid]] = shm_currBestLSetsDist[tid - nbrsBound];
                shm_BestLSets[shm_pos[tid]] = d_BestLSets[queryID * uWLLen + (tid - nbrsBound)];
                shm_BestLSets_visited[shm_pos[tid]] =
                    d_BestLSets_visited[queryID * uWLLen + (tid - nbrsBound)];
            }

            __syncthreads();
            __threadfence_block();

            if (tid < newBest_L_Set_size)
            {
                d_BestLSetsDist[uWLLen * queryID + tid] = shm_BestLSetsDist[tid];
                d_BestLSets[uWLLen * queryID + tid] = shm_BestLSets[tid];
                d_BestLSets_visited[uWLLen * queryID + tid] = shm_BestLSets_visited[tid];
            }
            __syncthreads();
            __threadfence_block();
        }
    }
    for (int i = tid; i < newBest_L_Set_size; i = i + blockDim.x)
    {
        if (d_mark[queryID] == d_BestLSets[uWLLen * queryID + tid])
            d_BestLSets_visited[uWLLen * queryID + tid] = true;
    }
}

extern "C" void phms_v4_merge_sparse_test(unsigned queries, unsigned L, unsigned* neighbors,
                                          unsigned* counts, float* dist, unsigned* best,
                                          float* bestdist, bool* visited, unsigned* bestcounts,
                                          unsigned* marks)
{
    compute_BestLSets_par_merge<<<queries, 2 * L, 65 * sizeof(float)>>>(
        neighbors, counts, dist, best, bestdist, visited, nullptr, 2, nullptr, bestcounts, marks, L,
        99999999, 64);
}

__device__ unsigned lower_bound_d(float arr[], unsigned lo, unsigned hi, float target)
{
    unsigned mid;

    while (lo < hi)
    {
        mid = (lo + hi) / 2;
        float val = arr[mid];
        if (target <= val)
            hi = mid;
        else
            lo = mid + 1;
    }

    return lo;
}

__device__ unsigned upper_bound_d(float arr[], unsigned lo, unsigned hi, float target)
{

    unsigned mid;

    while (lo < hi)
    {
        mid = (lo + hi) / 2;
        float val = arr[mid];
        if (target >= val)
            lo = mid + 1;
        else
            hi = mid;
    }

    return lo;
}

__device__ unsigned lower_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,
                                     unsigned row_size, unsigned queryID)
{
    unsigned mid;

    while (lo < hi)
    {
        mid = (lo + hi) / 2;
        float val = arr[(mid * row_size) + queryID];
        if (target <= val)
            hi = mid;
        else
            lo = mid + 1;
    }

    return lo;
}

__device__ unsigned upper_bound_d_ex(float arr[], unsigned lo, unsigned hi, float target,
                                     unsigned row_size, unsigned queryID)
{

    unsigned mid;

    while (lo < hi)
    {
        mid = (lo + hi) / 2;
        float val = arr[(mid * row_size) + queryID];
        if (target >= val)
            lo = mid + 1;
        else
            hi = mid;
    }

    return lo;
}

#if 0
void phms_set_searchparams_c(int recall, int worklist_length, DistFunc nDistFunc)
{
	m_objSearchParams.recall = recall;
	m_objSearchParams.worklist_length = worklist_length;
	m_objSearchParams.uDistFunc = nDistFunc;
	assert(m_objSearchParams.worklist_length <= MAX_L);
}
void phms_load_c(char* pszPath)
{
	phms_load<uint8_t>(pszPath);
}

void phms_query_c(uint8_t* query_array, int num_queries,
					result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist )
{
	phms_query<uint8_t>(query_array,num_queries, nearestNeighbours, nearestNeighbours_dist );
}
#endif
