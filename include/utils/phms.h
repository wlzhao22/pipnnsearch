#ifndef PHMS_H_
#define PHMS_H_

#include <cstdint>
#include <vector>

#define MAX_L 512

typedef unsigned long result_ann_t;

typedef enum _DistFunc
{
    ENUM_DIST_L2 = 0,
    ENUM_DIST_MIPS,
} DistFunc;
#define MIPS_EXTRA_DIM (1)

template <typename T> class PHMSSearch
{
    void* m_pImpl;

  public:
    PHMSSearch();
    virtual ~PHMSSearch();

    bool phms_load(char* indexfile_path_prefix);

    void phms_alloc(int numQueries);

    void phms_prepare_ngpu_graph(int numQueries);

    void phms_init(int numQueries);

    void phms_set_searchparams(int recall, int worklist_length, DistFunc nDistFunc = ENUM_DIST_L2);
    void phms_set_expansion_budget(unsigned budget);

    void phms_query(T* query_array, int num_queries, result_ann_t* nearestNeighbours,
                    float* nearestNeighbours_dist);

    void phms_free();

    void phms_select_resident_queries(const T* device_source, unsigned total_queries,
                                      const unsigned* host_ids, unsigned count,
                                      const T* host_source = nullptr);

    void phms_unload();

    bool phms_load_sharded(char* indexfile_path_prefix, unsigned shard_start, unsigned shard_end);

    static void ngpu_phms_query(std::vector<PHMSSearch<T>*>& gpu_objs, T* query_array,
                                int num_queries, result_ann_t* nearestNeighbours,
                                float* nearestNeighbours_dist,
                                const std::vector<unsigned>& shard_ends,
                                const unsigned* query_seeds = nullptr,
                                std::vector<std::vector<unsigned>>* debug_tracks = nullptr);
};
template class PHMSSearch<float>;
template class PHMSSearch<uint8_t>;
template class PHMSSearch<int8_t>;

#if 0
extern "C" void phms_load_c( char* indexfile_path_prefix);

extern "C"  void phms_set_searchparams_c(int recall, int worklist_length, DistFunc nDistFunc=ENUM_DIST_L2);

extern "C" void phms_query_c(uint8_t* query_array,
                    int num_queries,
                    result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist );

extern "C" void phms_unload_c( );
#endif

#endif
