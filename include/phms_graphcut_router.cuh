#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include "phms_adaptive_cards.h"
constexpr unsigned GC_BLOOM = 399887, GC_ROW = 65, GC_MAXL = 512, GC_PACKET = 67,
                   GC_INVALID = 0xffffffffu;
__device__ inline unsigned gc_hash(unsigned x, bool second)
{
    unsigned long long h = second ? 0x84222325ull : 0xcbf29ce4ull,
                       m = second ? 0x1B3ull : 0x01000193ull;
    for (unsigned b = 0; b < 4; ++b)
        h = (h ^ ((x >> (b * 8)) & 255)) * m;
    return h % GC_BLOOM;
}
__device__ inline void gc_offer(unsigned id, float d, unsigned* ids, float* ds)
{
    if (id == GC_INVALID || id == ids[0] || id == ids[1])
        return;
    if (d < ds[0])
    {
        ids[1] = ids[0];
        ds[1] = ds[0];
        ids[0] = id;
        ds[0] = d;
    }
    else if (d < ds[1])
    {
        ids[1] = id;
        ds[1] = d;
    }
}
__global__ void gc_norm(const float* x, float* norm, unsigned rows, unsigned d)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
        return;
    float s = 0;
    for (unsigned k = 0; k < d; ++k)
        s = fmaf(x[uint64_t(i) * d + k], x[uint64_t(i) * d + k], s);
    norm[i] = s;
}
__global__ void gc_route(const float* q, const float* centers, const unsigned* owners,
                         const float* old, const unsigned* oldentries, unsigned* mask,
                         unsigned* seed, unsigned d, unsigned nc, const float* dot = nullptr,
                         const float* norm = nullptr, const float* qnorm = nullptr,
                         unsigned* local_seeds = nullptr, unsigned cards = 2,
                         const unsigned* fine_entries = nullptr, bool adaptive = false,
                         PhmsAdaptiveCards policy = PhmsAdaptiveCards(),
                         unsigned* entry_masks = nullptr)
{
    unsigned t = threadIdx.x;
    unsigned qi = blockIdx.x;
    using Sort = cub::BlockRadixSort<float, 128, 8, unsigned>;
    __shared__ typename Sort::TempStorage temp;
    __shared__ unsigned sorted[1024], mins[4];
    __shared__ float olddist[256], oldmin[4];
    if (t < 4)
        mins[t] = __float_as_uint(FLT_MAX);
    __syncthreads();
    float keys[8];
    unsigned values[8];
    for (unsigned i = 0; i < 8; ++i)
    {
        unsigned j = t * 8 + i;
        float s = FLT_MAX;
        if (j < nc)
        {
            if (dot)
                s = fmaxf(0.0f, (qnorm[qi] + norm[j]) + dot[uint64_t(qi) * (nc + 256) + j]);
            else
            {
                s = 0;
                for (unsigned k = 0; k < d; ++k)
                {
                    float a = q[uint64_t(qi) * d + k] - centers[uint64_t(j) * d + k];
                    s = fmaf(a, a, s);
                }
            }
            atomicMin(mins + owners[j], __float_as_uint(s));
        }
        keys[i] = s;
        values[i] = j;
    }
    Sort(temp).Sort(keys, values);
    for (unsigned i = 0; i < 8; ++i)
        sorted[t * 8 + i] = values[i];
    for (unsigned j = t; j < 256; j += 128)
    {
        float s = 0;
        if (dot)
            s = fmaxf(0.0f, (qnorm[qi] + norm[nc + j]) + dot[uint64_t(qi) * (nc + 256) + nc + j]);
        else
            for (unsigned k = 0; k < d; ++k)
            {
                float a = q[uint64_t(qi) * d + k] - old[uint64_t(j) * d + k];
                s = fmaf(a, a, s);
            }
        olddist[j] = s;
    }
    __syncthreads();
    if (dot)
    {
        if (t == 0)
        {
            for (unsigned g = 0; g < 4; ++g)
                oldmin[g] = FLT_MAX;
            for (unsigned j = 0; j < 256; ++j)
                oldmin[oldentries[2 * j]] = fminf(oldmin[oldentries[2 * j]], olddist[j]);
        }
        __syncthreads();
        for (unsigned j = t; j < 256; j += 128)
        {
            float bound =
                512 * FLT_EPSILON *
                (qnorm[qi] + norm[nc + j] + fabsf(dot[uint64_t(qi) * (nc + 256) + nc + j]) + 1);
            if (olddist[j] <= oldmin[oldentries[2 * j]] + bound)
            {
                float s = 0;
                for (unsigned k = 0; k < d; ++k)
                {
                    float a = q[uint64_t(qi) * d + k] - old[uint64_t(j) * d + k];
                    s = fmaf(a, a, s);
                }
                olddist[j] = s;
            }
            else
                olddist[j] = FLT_MAX;
        }
        __syncthreads();
    }
    if (t == 0)
    {
        unsigned votes[4] = {};
        unsigned rank[4] = {0, 1, 2, 3};
        unsigned entry[4] = {};
        float score[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
        for (unsigned i = 0; i < 16; ++i)
            ++votes[owners[sorted[i]]];
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = i + 1; j < 4; ++j)
            {
                unsigned a = rank[i];
                unsigned b = rank[j];
                if (votes[b] > votes[a] ||
                    (votes[b] == votes[a] && (mins[b] < mins[a] || (mins[b] == mins[a] && b < a))))
                {
                    rank[i] = b;
                    rank[j] = a;
                }
            }
        unsigned m = 0;
        for (unsigned i = 0; i < cards; ++i)
            m |= 1u << rank[i];
        if (adaptive)
        {
            float distances[4];
            for (unsigned g = 0; g < 4; ++g)
                distances[g] = __uint_as_float(mins[g]);
            m = phms_adaptive_mask(rank, votes, distances, policy);
        }
        mask[qi] = m;
        for (unsigned j = 0; j < 256; ++j)
        {
            unsigned g = oldentries[j * 2];
            if (olddist[j] < score[g])
            {
                score[g] = olddist[j];
                entry[g] = oldentries[j * 2 + 1];
            }
        }
        unsigned entrymask = (1u << rank[0]);
        if (__popc(m) > 1)
            entrymask |= 1u << rank[1];
        if (entry_masks)
            entry_masks[qi] = entrymask;
        unsigned home = 0;
        float best = FLT_MAX;
        for (unsigned g = 0; g < 4; ++g)
            if ((entrymask & (1u << g)) && score[g] < best)
            {
                best = score[g];
                home = g;
            }
        seed[qi] = entry[home];
        if (fine_entries)
            for (unsigned i = 0; i < nc; ++i)
            {
                unsigned j = sorted[i];
                if (entrymask & (1u << owners[j]))
                {
                    seed[qi] = fine_entries[2 * j + 1];
                    break;
                }
            }
        if (local_seeds)
            for (unsigned g = 0; g < 4; ++g)
                local_seeds[qi * 4 + g] = entry[g];
    }
}

struct GcRegionChoice
{
    float distance;
    unsigned owner;
    unsigned index;
    unsigned id;
};
struct GcRegionMin
{
    __host__ __device__ GcRegionChoice operator()(const GcRegionChoice& a,
                                                  const GcRegionChoice& b) const
    {
        if (a.distance != b.distance)
            return a.distance < b.distance ? a : b;
        if (a.owner != b.owner)
            return a.owner < b.owner ? a : b;
        return a.index < b.index ? a : b;
    }
};
__global__ void gc_region_entry(const float* q, const float* centers, const unsigned* entries,
                                const unsigned* entry_masks, unsigned* seeds, unsigned dimension,
                                unsigned count, const float* dot = nullptr,
                                const float* center_norm = nullptr,
                                const float* query_norm = nullptr)
{
    const unsigned qi = blockIdx.x;
    const unsigned t = threadIdx.x;
    const unsigned mask = entry_masks[qi];
    GcRegionChoice best = {FLT_MAX, GC_INVALID, GC_INVALID, GC_INVALID};
    GcRegionMin minimum;
    using RangeReduce = cub::BlockReduce<float, 128>;
    __shared__ typename RangeReduce::TempStorage range_temp;
    __shared__ float cutoff;
    if (dot)
    {
        float upper = FLT_MAX;
        for (unsigned c = t; c < count; c += 128)
            if (mask & (1u << entries[2 * c]))
            {
                float product = dot[uint64_t(qi) * count + c];
                float estimate = fmaxf(0.0f, (query_norm[qi] + center_norm[c]) + product);
                float error =
                    512 * FLT_EPSILON * (query_norm[qi] + center_norm[c] + fabsf(product) + 1);
                upper = fminf(upper, estimate + error);
            }
        float all_upper = RangeReduce(range_temp).Reduce(upper, cub::Min());
        if (t == 0)
            cutoff = all_upper;
        __syncthreads();
    }
    for (unsigned c = t; c < count; c += 128)
    {
        unsigned owner = entries[2 * c];
        if (!(mask & (1u << owner)))
            continue;
        if (dot)
        {
            float product = dot[uint64_t(qi) * count + c];
            float estimate = fmaxf(0.0f, (query_norm[qi] + center_norm[c]) + product);
            float error =
                512 * FLT_EPSILON * (query_norm[qi] + center_norm[c] + fabsf(product) + 1);
            if (fmaxf(0.0f, estimate - error) > cutoff)
                continue;
        }
        float distance = 0;
        for (unsigned d = 0; d < dimension; ++d)
        {
            float a = q[uint64_t(qi) * dimension + d] - centers[uint64_t(c) * dimension + d];
            distance = fmaf(a, a, distance);
        }
        best = minimum(best, GcRegionChoice{distance, owner, c, entries[2 * c + 1]});
    }
    using Reduce = cub::BlockReduce<GcRegionChoice, 128>;
    __shared__ typename Reduce::TempStorage temp;
    auto winner = Reduce(temp).Reduce(best, minimum);
    if (t == 0)
        seeds[qi] = winner.id;
}
