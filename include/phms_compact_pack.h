#ifndef PHMS_COMPACT_PACK_H
#define PHMS_COMPACT_PACK_H
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <omp.h>

inline void phms_pack_shard_rows(const uint32_t* const* records, unsigned* const* marks,
                                 unsigned* const* buffers, int nq, int parents, int gpus,
                                 unsigned r, int thread_stride, unsigned* thread_counts,
                                 unsigned* thread_bases, size_t* words, unsigned& errors)
{
    const int tid = omp_get_thread_num();
    const int nth = omp_get_num_threads();
    const int begin = (int)((int64_t)nq * tid / nth);
    const int end = (int)((int64_t)nq * (tid + 1) / nth);
    for (int p = 0; p < parents; ++p)
    {
        unsigned running[8] = {};
        for (int q = begin; q < end; ++q)
        {
            const uint32_t* rec = records[(size_t)p * nq + q];
            const uint8_t* prefix = reinterpret_cast<const uint8_t*>(rec);
            for (int g = 0; g < gpus; ++g)
            {
                unsigned count = rec ? unsigned(prefix[g + 1]) - prefix[g] : 0;
                unsigned degree = rec ? prefix[gpus] : 0;
                unsigned lo = rec ? rec[2 + 2 * g] : 0;
                unsigned hi = rec ? rec[3 + 2 * g] : 0;
                if (count > r || degree > r || (rec && prefix[g + 1] > degree) ||
                    (unsigned)(__builtin_popcount(lo) + __builtin_popcount(hi)) != count)
                {
#pragma omp atomic update
                    errors |= 1u;
                    count = degree = lo = hi = 0;
                }
                unsigned* dst = buffers[p * gpus + g];
                dst[q] = marks[p][q];
                dst[nq + q] = degree;
                dst[2 * (size_t)nq + q] = running[g];
                dst[3 * (size_t)nq + q] = lo;
                dst[4 * (size_t)nq + q] = hi;
                running[g] += count;
            }
        }
        for (int g = 0; g < gpus; ++g)
            thread_counts[(p * gpus + g) * thread_stride + tid] = running[g];
    }
#pragma omp barrier
#pragma omp single
    {
        for (int p = 0; p < parents; ++p)
            for (int g = 0; g < gpus; ++g)
            {
                unsigned sum = 0;
                for (int t = 0; t < nth; ++t)
                {
                    const size_t i = (p * gpus + g) * thread_stride + t;
                    thread_bases[i] = sum;
                    sum += thread_counts[i];
                }
                words[p * gpus + g] = 5 * (size_t)nq + sum;
                if (sum > (size_t)nq * r)
                    errors |= 2u;
            }
    }
    if (!errors)
    {
        for (int p = 0; p < parents; ++p)
            for (int q = begin; q < end; ++q)
            {
                const uint32_t* rec = records[(size_t)p * nq + q];
                const uint8_t* prefix = reinterpret_cast<const uint8_t*>(rec);
                for (int g = 0; g < gpus; ++g)
                {
                    unsigned* dst = buffers[p * gpus + g];
                    unsigned& offset = dst[2 * (size_t)nq + q];
                    offset += thread_bases[(p * gpus + g) * thread_stride + tid];
                    const unsigned count = rec ? unsigned(prefix[g + 1]) - prefix[g] : 0;
                    if (count)
                        std::memcpy(dst + 5 * (size_t)nq + offset, rec + 2 + 2 * gpus + prefix[g],
                                    count * 4);
                }
            }
    }
}
inline void phms_pack_shard_tile(const uint32_t* const* records, unsigned* const* marks,
                                 unsigned* const* buffers, int nq, int parents, int gpus,
                                 unsigned r, int begin, int end, unsigned tile_rows, unsigned quota,
                                 unsigned* overflow, unsigned& errors)
{
    const size_t fixed_payload = ((nq + tile_rows - 1) / tile_rows) * (size_t)tile_rows * quota;
    for (int p = 0; p < parents; ++p)
    {
        unsigned total[8] = {};
        for (int q = begin; q < end; ++q)
        {
            const auto* rec = records[(size_t)p * nq + q];
            if (!rec)
                continue;
            const auto* prefix = reinterpret_cast<const uint8_t*>(rec);
            for (int g = 0; g < gpus; ++g)
            {
                const unsigned count = unsigned(prefix[g + 1]) - prefix[g];
                if (count > r || prefix[gpus] > r || prefix[g + 1] > prefix[gpus])
                {
                    __atomic_fetch_or(&errors, 4u, __ATOMIC_RELAXED);
                    return;
                }
                total[g] += count;
            }
        }
        unsigned cursor[8] = {};
        for (int g = 0; g < gpus; ++g)
        {
            if (total[g] <= tile_rows * quota)
                cursor[g] = (begin / tile_rows) * tile_rows * quota;
            else
            {
                const unsigned tail =
                    __atomic_fetch_add(overflow + p * gpus + g, total[g], __ATOMIC_RELAXED);
                if ((size_t)tail + total[g] > (size_t)nq * r)
                {
                    __atomic_fetch_or(&errors, 8u, __ATOMIC_RELAXED);
                    return;
                }
                cursor[g] = fixed_payload + tail;
            }
        }
        for (int q = begin; q < end; ++q)
        {
            const auto* rec = records[(size_t)p * nq + q];
            const auto* prefix = reinterpret_cast<const uint8_t*>(rec);
            for (int g = 0; g < gpus; ++g)
            {
                auto* dst = buffers[p * gpus + g];
                dst[q] = marks[p][q];
                dst[nq + q] = rec ? prefix[gpus] : 0;
                dst[2 * (size_t)nq + q] = cursor[g];
                dst[3 * (size_t)nq + q] = rec ? rec[2 + 2 * g] : 0;
                dst[4 * (size_t)nq + q] = rec ? rec[3 + 2 * g] : 0;
                const unsigned count = rec ? unsigned(prefix[g + 1]) - prefix[g] : 0;
                if (count)
                    std::memcpy(dst + 5 * (size_t)nq + cursor[g], rec + 2 + 2 * gpus + prefix[g],
                                count * 4);
                cursor[g] += count;
            }
        }
    }
}
#endif
