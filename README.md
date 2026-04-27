This project shares the pipelined heterogous (CPU-GPU) NN search in C++. This method allows both the raw vectors and the graph index to be kept in CPU memory. The vectors and the graph entries are loaded into device memory when they are traversed by the queries. It, therefore, overcomes the device memory constraint that the most GPU-based NN search methods face. Moreover, it also allows the search to be carried out on multiple GPU in parallel.

The codes are fully implemented by Ben Zhang from Xiamen University.
