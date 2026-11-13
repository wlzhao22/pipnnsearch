#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

inline void need(bool ok, const std::string& why)
{
    if (!ok)
        throw std::runtime_error(why);
}

struct CpuResidentFile
{
    std::unique_ptr<unsigned char[]> storage;

    bool load(int fd, size_t bytes, const std::string& path)
    {
        const char* enabled = std::getenv("PHMS_CPU_PRELOAD_FILES");
        if (!enabled || std::string(enabled) != "1" || bytes < (1ull << 30))
            return false;

        std::cerr << "PRELOAD_BEGIN " << path << " bytes=" << bytes << std::endl;
        storage.reset(new unsigned char[bytes]);
        size_t offset = 0;
        size_t next = 4ull << 30;

        while (offset < bytes)
        {
            ssize_t got = pread(fd, storage.get() + offset,
                                std::min<size_t>(8u << 20, bytes - offset), offset);
            if (got < 0 && errno == EINTR)
                continue;
            if (got <= 0)
                throw std::runtime_error("sequential preload failed: " + path);

            offset += got;
            if (offset >= next)
            {
                std::cerr << "PRELOAD_PROGRESS " << path << " bytes=" << offset << "/" << bytes
                          << std::endl;
                next += 4ull << 30;
            }
        }

        std::cerr << "PRELOAD_DONE " << path << " bytes=" << bytes << std::endl;
        return true;
    }
};

struct Map
{
    CpuResidentFile resident;
    int fd = -1;
    size_t bytes = 0;
    const uint8_t* data = nullptr;

    explicit Map(const std::string& path)
    {
        fd = open(path.c_str(), O_RDONLY);
        need(fd >= 0, "cannot open " + path);

        struct stat s
        {
        };
        need(fstat(fd, &s) == 0 && s.st_size > 0, "empty/invalid file " + path);
        bytes = s.st_size;

        if (resident.load(fd, bytes, path))
        {
            data = resident.storage.get();
        }
        else
        {
            data = (const uint8_t*)mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            need(data != MAP_FAILED, "mmap failed " + path);
        }
    }

    ~Map()
    {
        if (!resident.storage && data && data != MAP_FAILED)
            munmap((void*)data, bytes);
        if (fd >= 0)
            close(fd);
    }

    Map(const Map&) = delete;

    template <class T> T at(size_t off) const
    {
        need(off <= bytes && sizeof(T) <= bytes - off, "file header outside bounds");
        T v;
        std::memcpy(&v, data + off, sizeof(v));
        return v;
    }
};

struct Data
{
    Map graph;
    Map codes;
    Map pivots;
    Map queries;
    Map truth;
    Map meta;
    uint32_t n;
    uint32_t d;
    uint32_t r;
    uint32_t m;
    uint32_t nq;
    uint32_t gtq;
    uint32_t gtk;
    uint64_t medoid;
    uint64_t stride;
    const float* centers;
    const float* centroid;
    const float* query;
    const uint32_t* offsets;
    const uint32_t* gt;

    Data(const std::string& prefix, const std::string& q, const std::string& t)
        : graph(prefix + "_disk.bin"), codes(prefix + "_pq_compressed.bin"),
          pivots(prefix + "_pq_pivots.bin"), queries(q), truth(t),
          meta(prefix + "_disk_metadata.bin")
    {
        need(meta.bytes == 32, "graph metadata must be 32 bytes");
        medoid = meta.at<uint64_t>(0);
        stride = meta.at<uint64_t>(8);
        need(meta.at<uint32_t>(16) == 2, "CPU simulator supports float L2 graph only");
        d = meta.at<uint32_t>(20);
        r = meta.at<uint32_t>(24);
        n = meta.at<uint32_t>(28);
        need(n && d && r && medoid < n && stride >= (uint64_t)d * 4 + (r + 1) * 4,
             "invalid metadata");
        need(graph.bytes == (uint64_t)n * stride, "graph size mismatch");

        need(codes.at<uint32_t>(0) == n, "PQ node count mismatch");
        m = codes.at<uint32_t>(4);
        need(m >= 1 && m <= d && codes.bytes == 8 + (uint64_t)n * m, "unsupported PQ chunks/size");

        const uint64_t co = pivots.at<uint64_t>(8);
        const uint64_t ce = pivots.at<uint64_t>(16);
        const uint64_t of = pivots.at<uint64_t>(24);
        need(co + 8 + (uint64_t)256 * d * 4 <= pivots.bytes &&
                 ce + 8 + (uint64_t)d * 4 <= pivots.bytes &&
                 of + 8 + (uint64_t)(m + 1) * 4 <= pivots.bytes,
             "invalid pivot sections");
        need(pivots.at<uint32_t>(co) == 256 && pivots.at<uint32_t>(co + 4) == d,
             "pivot shape mismatch");

        centers = (const float*)(pivots.data + co + 8);
        centroid = (const float*)(pivots.data + ce + 8);
        offsets = (const uint32_t*)(pivots.data + of + 8);
        need(offsets[0] == 0 && offsets[m] == d, "PQ dimension offsets mismatch");
        for (unsigned j = 0; j < m; ++j)
            need(offsets[j] < offsets[j + 1], "PQ offsets not increasing");

        nq = queries.at<uint32_t>(0);
        need(queries.at<uint32_t>(4) == d && queries.bytes == 8 + (uint64_t)nq * d * 4,
             "query fbin mismatch");
        query = (const float*)(queries.data + 8);

        gtq = truth.at<uint32_t>(0);
        gtk = truth.at<uint32_t>(4);
        need(gtq == nq && gtk >= 10 && truth.bytes == 8 + (uint64_t)gtq * gtk * 4,
             "GT ubin mismatch");
        gt = (const uint32_t*)(truth.data + 8);
    }

    const uint32_t* adjacency(unsigned id) const
    {
        need(id < n, "invalid graph node");
        const auto* a = (const uint32_t*)(graph.data + (uint64_t)id * stride + d * 4);
        need(a[0] <= r, "degree exceeds graph bound");
        return a;
    }
};

struct Candidate
{
    unsigned id;
    float distance;
};

inline bool less_candidate(const Candidate& a, const Candidate& b)
{
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

inline float phms_simd_l2(const float* x, const float* q, unsigned dimension)
{
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (unsigned j = 0; j < dimension; ++j)
    {
        float difference = x[j] - q[j];
        sum += difference * difference;
    }
    return sum;
}
