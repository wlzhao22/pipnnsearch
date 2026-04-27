#ifndef XGNND_H_
#define XGNND_H_

#include <cstdint>

#define MAX_L 512 // L_search Upper bound


typedef unsigned long result_ann_t ; // big-ann-benchmarks requires the final ANNs to be returned as int64_t

// Type of Similarity distnace measure
typedef enum _DistFunc
{
	ENUM_DIST_L2 = 0, // Euclidean Distance
	ENUM_DIST_MIPS, // Max Inner Product Search
} DistFunc;
#define MIPS_EXTRA_DIM (1) // To transform MIPS to L2 distance caluclation, extra dim is added to base dataset adn query
                          // The index file o/p from DiskANN already has 1 DIM added to dataset. We(XGNND) add 1 DIM to the query
                          // at rum time


template<typename T>
class XGNNDSearch
{
    void* m_pImpl;

    public:
    XGNNDSearch();
    virtual ~XGNNDSearch();

    /*! @brief Load the graph index, compressed vectors etc into CPU/GPU memory.
    *
    * The graph index, compressed vectors has to be generated using DiskANN.
    * Search can be performed xgnnd_query() .
    *
    * @param[in] indexfile_path_prefix Absolute path location where DiskANN generated files are present.
    *               (including the file prefix)
    */
    bool xgnnd_load( char* indexfile_path_prefix);

    void xgnnd_alloc(int numQueries);


    void xgnnd_init(int numQueries);

    void xgnnd_set_searchparams(int recall,
                            int worklist_length,
                            DistFunc nDistFunc=ENUM_DIST_L2);


    /*! @brief Runs search queries on the laoded index..
    *
    * The graph index, compressed vectors has to be generated using DiskANN.
    * Search can be performed xgnnd_query() .
    *
    * @param[in] query_file Absolute path of the query file in bin format.
    * @param[in] groundtruth_file Absolute path of the groundtruth file in bin format (generated using DiskANN).
    * @param[in] num_queries Number of queries to be used for the search.
    * @param[in] recal_param k-recall@k.
    */
    void xgnnd_query(T* query_array,
                    int num_queries,
                    result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist );

    void xgnnd_free();

    void xgnnd_unload();

};
template class XGNNDSearch<float>;
template class XGNNDSearch<uint8_t>;
template class XGNNDSearch<int8_t>;

#if 0
// Note:  Equivalent "C" APIs have also been provided. For invocation from Python scripts (using CDLL package)
extern "C" void xgnnd_load_c( char* indexfile_path_prefix);

extern "C"  void xgnnd_set_searchparams_c(int recall, int worklist_length, DistFunc nDistFunc=ENUM_DIST_L2);

extern "C" void xgnnd_query_c(uint8_t* query_array,
                    int num_queries,
                    result_ann_t* nearestNeighbours,
					float* nearestNeighbours_dist );

extern "C" void xgnnd_unload_c( );
#endif

#endif //XGNND_H_
