#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include <sys/stat.h>
#include <cmath>
#include <vector>
#include <set>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include "xgnnd.h"
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
#include <atomic>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <condition_variable>
#include <sys/stat.h>   // 顶部要有

static inline bool file_exists_cstr(const char* name) {
    struct stat buffer;
    return ::stat(name, &buffer) == 0;
}

unsigned long long checkpoint_time_millisec();

double calculate_recall(unsigned num_queries, unsigned *gold_std,
                        float *gs_dist, unsigned dim_gs,
                        result_ann_t *our_results, unsigned dim_or,
                        unsigned recall_at);

inline void load_truthset(const std::string &bin_file, uint32_t *&ids,
                          float *&dists, size_t &npts, size_t &dim);


template <typename T>
int run_anns_xgnnd(int argc, char **argv)
{
    const int numQueries = atoi(argv[4]);
    const int k          = atoi(argv[5]);
    const int batch_size = atoi(argv[8]); 
	const int batch_threads = atoi(argv[9]);
    std::cout << "batch size " << batch_size << std::endl;
	std::cout << "batch_threads" << batch_threads << std::endl;
    DistFunc uDistFunc = ENUM_DIST_L2;
    if (std::string(argv[7]) == "mips") uDistFunc = ENUM_DIST_MIPS;

    auto setup_omp = [&](int nthreads){
        omp_set_dynamic(0);
        omp_set_num_threads(nthreads);
        setenv("OMP_DYNAMIC", "FALSE", 1);
        setenv("OMP_PROC_BIND", "TRUE", 1);
        setenv("OMP_PLACES", "cores", 1);
    };

    const int xgnnd_cpu_threads = batch_threads;
    setup_omp(xgnnd_cpu_threads);
    {
        static std::string s;
        s = std::to_string(xgnnd_cpu_threads);
        setenv("XGNND_CPU_THREADS", s.c_str(), 1);
    }


    int DeviceID = 1;
    cudaSetDevice(DeviceID);
    cudaFree(0); // warmup driver/runtime

    std::string queryPointsFP_file = std::string(argv[2]);
    std::ifstream in4(queryPointsFP_file, std::ios::binary);
    if (!in4.is_open()) {
        printf("Error.. Could not open the Query File: %s\n", queryPointsFP_file.c_str());
        return -1;
    }
    in4.seekg(4);
    int Dim = 0;
    in4.read((char *)&Dim, sizeof(int));

    T *queriesFP = (T *)malloc(sizeof(T) * (size_t)numQueries * (size_t)Dim);
    if (!queriesFP) {
        printf("Error.. malloc failed for queriesFP\n");
        return -1;
    }
    in4.read((char *)queriesFP, sizeof(T) * (size_t)Dim * (size_t)numQueries);
    in4.close();

    // 输出 buffer
    auto *all_nn   = (result_ann_t*)malloc(sizeof(result_ann_t) * (size_t)k * (size_t)numQueries);
    auto *all_dist = (float*)malloc(sizeof(float) * (size_t)k * (size_t)numQueries);
    if (!all_nn || !all_dist) {
        printf("Error.. malloc failed for outputs\n");
        free(queriesFP);
        if (all_nn) free(all_nn);
        if (all_dist) free(all_dist);
        return -1;
    }

    auto file_exists = [](const char* path)->bool {
        std::ifstream f(path, std::ios::binary);
        return f.good();
    };

    unsigned *gt_ids = nullptr;
    float    *gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;

    if (!file_exists(argv[3])) {
        std::cout << "Groundtruth file could not be loaded: " << argv[3] << std::endl;
        free(queriesFP);
        free(all_nn);
        free(all_dist);
        return -1;
    }
    load_truthset(argv[3], gt_ids, gt_dists, gt_num, gt_dim);


    int num_batches = batch_size;
    if (num_batches <= 0) num_batches = 1; 

    int queries_per_batch = numQueries / num_batches;
    int remaining_queries = numQueries % num_batches;

    std::vector<int> batch_starts(num_batches);
    std::vector<int> batch_sizes(num_batches);
    int start_idx = 0;

    for (int i = 0; i < num_batches; ++i) {
        batch_starts[i] = start_idx;
        batch_sizes[i] = queries_per_batch + (i < remaining_queries ? 1 : 0);
        start_idx += batch_sizes[i];
    }

    cudaSetDevice(DeviceID);
    std::vector<XGNNDSearch<T>> objs(num_batches);

    for (int i = 0; i < num_batches; ++i) {
        if (!objs[i].xgnnd_load(argv[1])) {
            std::cout << "Error: objs[" << i << "].xgnnd_load failed\n";
            // cleanup skipped for brevity on failure
            return -1;
        }
    }

    std::cout << "L\t" << "Time(ms)\t" << "QPS\t\t" << k << "-r@" << k << std::endl;
    std::cout << "--\t" << "-------\t\t" << "---\t\t" << "------" << std::endl;

    while (true)
    {
        int L = 0;
        std::cout << "Enter value of WorkList Length" << std::endl;
        std::cin >> L;

        if (L < k) {
            std::cout << "Error: WorkList Length must be at least recall_at (k)." << std::endl;
            continue;
        }

        // set params + alloc (per L)
        for (int i = 0; i < num_batches; ++i) {
            objs[i].xgnnd_set_searchparams(k, L, uDistFunc);
            objs[i].xgnnd_alloc(batch_sizes[i]);
        }

        // ----------------------------------------------------
        // Warmup 
        // ----------------------------------------------------
        // {
        //     std::vector<std::thread> threads;
        //     for (int i = 0; i < num_batches; ++i) {
        //         threads.emplace_back([&, i](){
        //             cudaSetDevice(DeviceID);
        //             objs[i].xgnnd_init(batch_sizes[i]);
        //             objs[i].xgnnd_query(queriesFP + (size_t)batch_starts[i] * (size_t)Dim, batch_sizes[i],
        //                                all_nn + (size_t)batch_starts[i] * (size_t)k, 
        //                                all_dist + (size_t)batch_starts[i] * (size_t)k);
        //         });
        //     }
        //     for (auto& t : threads) t.join();
        //     cudaSetDevice(DeviceID);
        //     cudaDeviceSynchronize();
        // }

        // ----------------------------------------------------
        // Run Loop (5 times)
        // ----------------------------------------------------
        for (int run = 0; run < 5; ++run)
        {
            for (int i = 0; i < num_batches; ++i) {
                cudaSetDevice(DeviceID); 
                objs[i].xgnnd_init(batch_sizes[i]);
            }
            cudaDeviceSynchronize();

            std::vector<std::thread> run_threads;
            std::atomic<int>  ready{0};
            std::atomic<bool> go{false};


            for (int i = 1; i < num_batches; ++i) {
                run_threads.emplace_back([&, i](){
                    ready.fetch_add(1, std::memory_order_release);
                    while (!go.load(std::memory_order_acquire)) { /* spin */ }

                    cudaSetDevice(DeviceID);
                    objs[i].xgnnd_query(queriesFP + (size_t)batch_starts[i] * (size_t)Dim, batch_sizes[i],
                                       all_nn + (size_t)batch_starts[i] * (size_t)k, 
                                       all_dist + (size_t)batch_starts[i] * (size_t)k);
                });
            }

            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) < num_batches) {}

            auto t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);

            if (num_batches > 0) {
                cudaSetDevice(DeviceID);
                objs[0].xgnnd_query(queriesFP + (size_t)batch_starts[0] * (size_t)Dim, batch_sizes[0],
                                   all_nn + (size_t)batch_starts[0] * (size_t)k, 
                                   all_dist + (size_t)batch_starts[0] * (size_t)k);
            }

            for (auto& t : run_threads) {
                t.join();
            }

            cudaSetDevice(DeviceID);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::steady_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double qps = (numQueries * 1000.0) / total_ms;

            float recall = calculate_recall(numQueries, gt_ids, gt_dists, gt_dim,
                                            all_nn, k, k);

            std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
            std::cout.precision(2);
            std::cout << L << "\t" << total_ms << "\t\t" << qps << "\t\t" << recall << std::endl;
			cout << L << "," << qps << "," << std::fixed << std::setprecision(4) << (recall / 100.0) << endl;
        }

        // Free resources for next L loop
        for (int i = 0; i < num_batches; ++i) {
            objs[i].xgnnd_free();
        }

        char c = 'n';
        std::cout << "Try Next run ? [y|n]" << std::endl;
        std::cin >> c;
        if (c == 'n') break;
    }

    // Unload resources
    for (int i = 0; i < num_batches; ++i) {
        objs[i].xgnnd_unload();
    }

    delete[] gt_ids;
    delete[] gt_dists;

    free(queriesFP);
    free(all_nn);
    free(all_dist);
    return 0;
}



inline unsigned long long checkpoint_time_millisec ()
{
	const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	return millis;
}

double calculate_recall(unsigned num_queries, unsigned *gold_std,
						float *gs_dist, unsigned dim_gs,
						result_ann_t *our_results, unsigned dim_or,
						unsigned recall_at)
{
	double total_recall = 0;
	std::set<unsigned> gt, res;

	for (size_t i = 0; i < num_queries; i++)
	{
		// cout << "Query : " << i << endl;
		gt.clear();
		res.clear();
		unsigned *gt_vec = gold_std + dim_gs * i;
		result_ann_t *res_vec = our_results + dim_or * i;
		size_t tie_breaker = recall_at;

		if (gs_dist != nullptr)
		{
			tie_breaker = recall_at - 1;
			float *gt_dist_vec = gs_dist + dim_gs * i;
			while (tie_breaker < dim_gs &&
				   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
				tie_breaker++;
		}

		gt.insert(gt_vec, gt_vec + tie_breaker);
		res.insert(res_vec, res_vec + recall_at);
		/*cout << "Results: ";
		for (int nIterInner = 0; nIterInner < recall_at; nIterInner++)
		{
			cout << our_results[(i*recall_at) + nIterInner] << "\t" ;
		}
		cout << endl;
		cout << "GT: ";*/
		unsigned cur_recall = 0;
		for (auto &v : gt)
		{
			// cout << v << "\t" ;
			if (res.find(v) != res.end())
			{
				cur_recall++;
			}
		}
		// cout << endl;
		total_recall += cur_recall;
	}

	//std::cout << "total_recall = " << total_recall << " " << "num_queries = " << num_queries << " recall_at " << recall_at << endl;
	return total_recall / (num_queries) * (100.0 / recall_at);
}

inline bool file_exists(const std::string &name)
{
	struct stat buffer;
	auto val = stat(name.c_str(), &buffer);
	//std::cout << " Stat(" << name.c_str() << ") returned: " << val 		  << std::endl;
	return (val == 0);
}
class cached_ifstream
{
public:
	cached_ifstream()
	{
	}
	cached_ifstream(const std::string &filename, uint64_t cacheSize)
		: cache_size(cacheSize), cur_off(0)
	{
		this->open(filename, cache_size);
	}
	~cached_ifstream()
	{
		delete[] cache_buf;
		reader.close();
	}

	void open(const std::string &filename, uint64_t cacheSize);
	size_t get_file_size();

	void read(char *read_buf, uint64_t n_bytes);

private:
	// underlying ifstream
	std::ifstream reader;
	// # bytes to cache in one shot read
	uint64_t cache_size = 0;
	// underlying buf for cache
	char *cache_buf = nullptr;
	// offset into cache_buf for cur_pos
	uint64_t cur_off = 0;
	// file size
	uint64_t fsize = 0;
};

/*Helper function*/
void cached_ifstream ::open(const std::string &filename, uint64_t cacheSize)
{
	this->cur_off = 0;
	reader.open(filename, std::ios::binary | std::ios::ate);
	fsize = reader.tellg();
	reader.seekg(0, std::ios::beg);
	assert(reader.is_open());
	assert(cacheSize > 0);
	cacheSize = (std::min)(cacheSize, fsize);
	this->cache_size = cacheSize;
	cache_buf = new char[cacheSize];
	reader.read(cache_buf, cacheSize);
	//cout << "Opened: " << filename.c_str() << ", size: " << fsize  << ", cache_size: " << cacheSize << std::endl;
}

size_t cached_ifstream ::get_file_size()
{
	return fsize;
}
void cached_ifstream ::read(char *read_buf, uint64_t n_bytes)
{
	assert(cache_buf != nullptr);
	assert(read_buf != nullptr);
	if (n_bytes <= (cache_size - cur_off))
	{
		// case 1: cache contains all data
		memcpy(read_buf, cache_buf + cur_off, n_bytes);
		cur_off += n_bytes;
	}
	else
	{
		// case 2: cache contains some data
		uint64_t cached_bytes = cache_size - cur_off;
		if (n_bytes - cached_bytes > fsize - reader.tellg())
		{
			std::stringstream stream;
			stream << "Reading beyond end of file" << std::endl;
			stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes
				   << " fsize: " << fsize << " current pos:" << reader.tellg()
				   << std::endl;
			cout << stream.str() << std::endl;
			exit(1);
		}
		memcpy(read_buf, cache_buf + cur_off, cached_bytes);

		reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
		cur_off = cache_size;

		uint64_t size_left = fsize - reader.tellg();

		if (size_left >= cache_size)
		{
			reader.read(cache_buf, cache_size);
			cur_off = 0;
		}
	}
}
inline void open_file_to_write(std::ofstream &writer,
							   const std::string &filename)
{
	writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	if (!file_exists(filename))
		writer.open(filename, std::ios::binary | std::ios::out);
	else
		writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);

	if (writer.fail())
	{
		char buff[1024];
		strerror_r(errno, buff, 1024);

		cerr << std::string("Failed to open file") + filename +
					" for write because " + buff
			 << std::endl;
	}
}

template <typename T>
inline uint64_t save_bin(const std::string &filename, T *data, size_t npts,
						 size_t ndims, size_t offset = 0)
{
	std::ofstream writer;
	open_file_to_write(writer, filename);

	cout << "Writing bin: " << filename.c_str() << std::endl;
	writer.seekp(offset, writer.beg);
	int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
	size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
	writer.write((char *)&npts_i32, sizeof(int));
	writer.write((char *)&ndims_i32, sizeof(int));
	cout << "bin: #pts = " << npts << ", #dims = " << ndims
		 << ", size = " << bytes_written << "B" << std::endl;

	writer.write((char *)data, npts * ndims * sizeof(T));
	writer.close();
	cout << "Finished writing bin." << std::endl;
	return bytes_written;
}
// compute ground truth

inline void load_truthset(const std::string &bin_file, uint32_t *&ids,
						  float *&dists, size_t &npts, size_t &dim)
{
	uint64_t read_blk_size = 64 * 1024 * 1024;
	cached_ifstream reader(bin_file, read_blk_size);
	//std::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
	size_t actual_file_size = reader.get_file_size();

	int npts_i32, dim_i32;
	reader.read((char *)&npts_i32, sizeof(int));
	reader.read((char *)&dim_i32, sizeof(int));
	npts = (unsigned)npts_i32;
	dim = (unsigned)dim_i32;

	//std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."  << std::endl;

	size_t expected_actual_file_size =
		2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);
	// if (actual_file_size != expected_actual_file_size)
	// {
	// 	std::stringstream stream;
	// 	stream << "Error. File size mismatch. Actual size is " << actual_file_size
	// 		   << " while expected size is  " << expected_actual_file_size
	// 		   << " npts = " << npts << " dim = " << dim << std::endl;
	// 	std::cout << stream.str();
	// 	//      throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
	// 	//                                  __LINE__);
	// 	exit(1);
	// }

	ids = new uint32_t[npts * dim];
	reader.read((char *)ids, npts * dim * sizeof(uint32_t));
	// dists = new float[npts * dim];
	// reader.read((char *)dists, npts * dim * sizeof(float));
}
const string DT_UINT8("uint8");
const string DT_INT8("int8");
const string DISTFUNC_MIPS("mips");
const string MODE_INTERACTIVE("interactive");
const string MODE_AUTO("auto");

const string DT_FLOAT("float");
template <typename T>
void preprocess_query_file(string queryPointsFP_file, int numQueries)
{
	// load the queries
	T *queriesFP = NULL;
	// ToDo : handle case of non-float datatype
	float *queriesFP_transformed = NULL;

	ifstream in4(queryPointsFP_file, std::ios::binary);
	if (!in4.is_open())
	{
		printf("Error.. Could not open the Query File: %s\n", queryPointsFP_file.c_str());
		return;
	}

	in4.seekg(4);
	int Dim = 0;
	in4.read((char *)&Dim, sizeof(int));

	queriesFP = (T *)malloc(sizeof(T) * numQueries * Dim); // full floating point coordinates of queries
	queriesFP_transformed = (float *)malloc(sizeof(float) * numQueries * (Dim + 1));
	if (NULL == queriesFP || NULL == queriesFP_transformed)
	{
		printf("Error.. Malloc failed for queriesFP");
		return;
	}

	in4.read((char *)queriesFP, sizeof(T) * Dim * numQueries);
	in4.close();

	unsigned numCPUthreads = 64;
	omp_set_num_threads(numCPUthreads);

#pragma omp parallel
	{
		int CPUthreadno = omp_get_thread_num();
		for (unsigned ii = CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
		{
			float query_norm = 0;
			float *query1 = queriesFP + (ii * Dim);
			for (uint32_t i = 0; i < Dim; i++)
			{
				query_norm += (query1[i] * query1[i]);
			}
			query_norm = std::sqrt(query_norm);
			float *query1_transformed = queriesFP_transformed + (ii * (Dim + 1));
			query1_transformed[(Dim)] = 0;
			for (uint32_t i = 0; i < Dim; i++)
			{
				query1_transformed[i] = query1[i] / query_norm;
			}
		}
	}
	save_bin(queryPointsFP_file + "_transformed", queriesFP_transformed, numQueries, Dim + 1);
	free(queriesFP);
	free(queriesFP_transformed);
}

template <typename T>
int run_anns(int argc, char **argv)
{
	XGNNDSearch<T> objXGNND;
	if (false == objXGNND.xgnnd_load(argv[1]))
	{
		cout << "Error: Bang_load failed" << endl;
		return -1;
	}	
	//sleep(10);
	// load the queries
	T *queriesFP = NULL;
	int numQueries = atoi(argv[4]);

	string queryPointsFP_file = string(argv[2]);
	ifstream in4(queryPointsFP_file, std::ios::binary);
	if (!in4.is_open())
	{
		printf("Error.. Could not open the Query File: %s\n", queryPointsFP_file.c_str());
		return -1;
	}

	in4.seekg(4);
	int Dim = 0;
	in4.read((char *)&Dim, sizeof(int));

	queriesFP = (T *)malloc(sizeof(T) * numQueries * Dim); // full floating point coordinates of queries

	if (NULL == queriesFP)
	{
		printf("Error.. Malloc failed for queriesFP");
		return -1;
	}

	in4.read((char *)queriesFP, sizeof(T) * Dim * numQueries);
	in4.close();

	int recall_param = atoi(argv[5]);
	int nWLLen = recall_param;
	int nStepSize = 12;
	int nIter = 0;
	DistFunc uDistFunc = ENUM_DIST_L2;

	if (DISTFUNC_MIPS == argv[7])
		uDistFunc = ENUM_DIST_MIPS;

	string strMode("MODE_AUTO");
	if(argc == 8)
		strMode = MODE_INTERACTIVE;
	
		do
		{
			if (MODE_INTERACTIVE == strMode)
			{
			cout << "Enter value of WorkList Length" << endl;
			cin >> nWLLen;
			if (nWLLen < recall_param )
			{
				cout << " Error: WorkList Length must be at least recall_at" << endl;
				continue;
			}

//            if (nIter == 0)
            {
                cout << "L\t" << "Time \t" << "QPS\t" << "\t" << recall_param <<"-r@" << recall_param << endl;
			    cout << "--\t" << "---- \t" << "---\t" << "\t------" << endl;
            }
			}
			else 
			{
                if (nIter == 0)
                {
                    cout << "L\t" << "Time \t" << "QPS\t" << "\t" << recall_param <<"-r@" << recall_param << endl;
                    cout << "--\t" << "---- \t" << "---\t" << "\t------" << endl;
                }
                else
				    nWLLen = nWLLen + (nStepSize);

                if (nWLLen >  MAX_L)
                    break;
			}

                
			objXGNND.xgnnd_set_searchparams(recall_param, nWLLen, uDistFunc);
			objXGNND.xgnnd_alloc(numQueries);

			for (int nCounter = 0; nCounter < 5; nCounter++)
			{
				result_ann_t *nearestNeighbours = (result_ann_t *)malloc(sizeof(result_ann_t) * recall_param * numQueries);
				float *nearestNeighbours_dist = (float *)malloc(sizeof(float) * recall_param * numQueries);
				vector<vector<result_ann_t>> final_bestL1; // Per query vector to store the visited parent and its distance to query point
				final_bestL1.resize(numQueries);

	
				objXGNND.xgnnd_init(numQueries);
				auto milliStart = checkpoint_time_millisec ();

				objXGNND.xgnnd_query(queriesFP, numQueries, nearestNeighbours, nearestNeighbours_dist);

				auto milliEnd = checkpoint_time_millisec ();
				double totalTime_wallclock = milliEnd - milliStart;
				double throughput = (numQueries * 1000.0) / totalTime_wallclock;
				// std::cout << " num queries " << numQueries << std::endl;
				
						
				// compute recall
				unsigned numCPUthreads = 64;
				omp_set_num_threads(numCPUthreads);
#pragma omp parallel
				{
					int CPUthreadno = omp_get_thread_num();
					vector<result_ann_t> query_best;

					for (unsigned ii = CPUthreadno; ii < numQueries; ii = ii + numCPUthreads)
					{
						query_best.clear();
						for (unsigned jj = 0; jj < recall_param; ++jj)
						{
							// query_best.push_back(nearestNeighbours[ ( numQueries * jj) + ii]);
							query_best.push_back(nearestNeighbours[(recall_param * ii) + jj]);
						}

#pragma omp critical
						{
							final_bestL1[ii] = query_best;
							//				printf("final_bestL1[%d] size = %lu \n", ii, final_bestL1[ii].size());
						}
					}
				}
				// Computing the recall

				unsigned *gt_ids = nullptr;
				float *gt_dists = nullptr;
				size_t gt_num, gt_dim;
				std::vector<uint64_t> Lvec;
				bool calc_recall_flag = false;

				uint64_t curL = nWLLen;
				if (curL >= recall_param)
					Lvec.push_back(curL);

				// Like DiskANN o/p we wannted to run with multiple L's in a single invocation of the program
				// So, test_id to each run with a specific L value. But, currently, we have only one L value
				// i.e. Lvec.size() == 1
				if (Lvec.size() == 0)
				{
					std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
							  << std::endl;
					exit(1);
				}
				if (file_exists(argv[3]))
				{
					load_truthset(argv[3], gt_ids, gt_dists, gt_num, gt_dim);
					calc_recall_flag = true;
				}
                else
                {
					std::cout << "Groundtruth file could not be loaded:" << argv[3] << std::endl;
                    exit(1);
                }
				std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
				std::cout.precision(2);
				std::string recall_string = "Recall@" + std::to_string(recall_param);

				std::vector<std::vector<result_ann_t>> query_result_ids(Lvec.size());

				unsigned total_size = 0;
				float recall = 0;
				for (unsigned test_id = 0; test_id < Lvec.size(); test_id++)
				{
					unsigned L1 = Lvec[test_id];
					query_result_ids[test_id].resize(recall_param * numQueries);

					for (unsigned ii = 0; ii < numQueries; ++ii)
					{
						// total_size += final_bestL1[ii].size();
						for (unsigned jj = 0; jj < recall_param; ++jj)
						{
							query_result_ids[test_id][ii * recall_param + jj] = final_bestL1[ii][jj];
						}
					}

					if (calc_recall_flag)
						recall = calculate_recall(numQueries, gt_ids, gt_dists, gt_dim, 
						query_result_ids[test_id].data(), recall_param, recall_param);
					//cout << "Ls\t" << recall_string << endl;
					//cout << L1 << "\t" << recall << endl;
				}
				cout << nWLLen << "\t" << totalTime_wallclock << "\t" << throughput << "\t" <<  recall << endl;

				free(nearestNeighbours);
				nearestNeighbours = NULL;
				free(nearestNeighbours_dist);
				nearestNeighbours_dist = NULL;
				delete[] gt_ids;
				delete[] gt_dists; 
			}
			objXGNND.xgnnd_free();
			if (MODE_INTERACTIVE == strMode)
			{
                char c = 'n';
                cout << "Try Next run ? [y|n]" << endl;
                cin >> c;
                if (c == 'n')
                    break;
            }
            nIter++;
		} while (1);
	
	/*
	else if (MODE_AUTO == argv[8] )
	{

	}*/

	objXGNND.xgnnd_unload();
	free(queriesFP);
	
	return 0;
}


int main(int argc, char **argv)
{
	if (argc == 3)
	{
		int numQueries = atoi(argv[2]);
		preprocess_query_file<float>(argv[1], numQueries);
		return 0;
	}

	if (argc < 9)
	{
		cerr << "Too few parameters! " << argv[0] << " " << "<path with file prefix to the director with index files > \
		<query file> <GroundTruth File> <NumQueries> <recall parameter k> <data type : uint8, int8 or float> <dist funct: l2 or mips>"
			 << endl;
		exit(1);
	}

	if (DT_UINT8 == argv[6])
	{
		return run_anns<uint8_t>(argc, argv);
	}
	else if (DT_INT8 == argv[6])
	{
		 return run_anns<int8_t>(argc, argv);
	}
	else if (DT_FLOAT == argv[6])
	{
		std::cout << " here \n";
		return run_anns_xgnnd<float>(argc, argv);

	}
	else
	{
		cerr << "Invalid data type specified" << endl;
		exit(1);
	}
}
