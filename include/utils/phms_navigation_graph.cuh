#pragma once
#include "phms_graphcut_router.cuh"
#include "phms_navigation_model.h"
constexpr unsigned GC_NAV_QUERY_CACHE = 4096;
inline size_t gc_nav_shared_bytes(unsigned count, unsigned dimension)
{
    return (size_t((count + 31) / 32) + (dimension <= GC_NAV_QUERY_CACHE ? dimension : 0)) *
           sizeof(unsigned);
}
__device__ __forceinline__ float gc_nav_quad_fp32(const float* query, const float* point,
                                                  unsigned dimension, unsigned lane,
                                                  unsigned quad_mask)
{
    float distance = 0;
    for (unsigned k = lane; k < dimension; k += 4)
    {
        float a = query[k] - point[k];
        distance = fmaf(a, a, distance);
    }
    distance += __shfl_xor_sync(quad_mask, distance, 1, 4);
    distance += __shfl_xor_sync(quad_mask, distance, 2, 4);
    return distance;
}
__global__ void gc_navigation_graph(const float* queries, const float* points,
                                    const unsigned* entries, const unsigned* adj,
                                    const unsigned* starts, unsigned* mask, unsigned* seed,
                                    unsigned* statistics, unsigned count, unsigned degree,
                                    unsigned dimension, unsigned beam, unsigned budget,
                                    unsigned cards, bool adaptive, PhmsAdaptiveCards policy,
                                    PhmsNavStop stop = PhmsNavStop())
{
    unsigned t = threadIdx.x;
    unsigned qid = blockIdx.x;
    extern __shared__ unsigned seen[];
    float* cached_query = reinterpret_cast<float*>(seen + (count + 31) / 32);
    const float* query =
        dimension <= GC_NAV_QUERY_CACHE ? cached_query : queries + uint64_t(qid) * dimension;
    __shared__ float distances[128], pending_distance[64];
    __shared__ unsigned ids[128], done[128], pending[64], size, parent, evals, hops;
    PhmsNavStopState stop_state;
    for (unsigned w = t; w < (count + 31) / 32; w += blockDim.x)
        seen[w] = 0;
    if (dimension <= GC_NAV_QUERY_CACHE)
        for (unsigned k = t; k < dimension; k += blockDim.x)
            cached_query[k] = queries[uint64_t(qid) * dimension + k];
    if (t == 0)
    {
        size = 0;
        evals = 0;
        hops = 0;
        parent = GC_INVALID;
    }
    __syncthreads();
    for (unsigned step = 0; step <= budget; ++step)
    {
        unsigned width = step ? degree : 4;
        if (t == 0)
        {
            for (unsigned j = 0; j < width; ++j)
            {
                unsigned v = step ? adj[uint64_t(parent) * degree + j] : starts[j];
                unsigned bit = 1u << (v % 32);
                bool fresh = !(seen[v / 32] & bit);
                seen[v / 32] |= bit;
                pending[j] = fresh ? v : GC_INVALID;
                evals += fresh;
            }
        }
        __syncthreads();
        if (t < width)
        {
            unsigned v = pending[t];
            float sum = FLT_MAX;
            if (v != GC_INVALID)
            {
                sum = 0;
                for (unsigned k = 0; k < dimension; ++k)
                {
                    float a = query[k] - points[uint64_t(v) * dimension + k];
                    sum = fmaf(a, a, sum);
                }
            }
            pending_distance[t] = sum;
        }
        __syncthreads();
        if (t == 0)
        {
            for (unsigned j = 0; j < width; ++j)
            {
                unsigned v = pending[j];
                if (v == GC_INVALID)
                    continue;
                float distance = pending_distance[j];
                unsigned pos = 0;
                while (pos < size &&
                       (distances[pos] < distance || (distances[pos] == distance && ids[pos] < v)))
                    ++pos;
                if (pos >= beam)
                    continue;
                unsigned last = size < beam ? size++ : beam - 1;
                for (unsigned k = last; k > pos; --k)
                {
                    distances[k] = distances[k - 1];
                    ids[k] = ids[k - 1];
                    done[k] = done[k - 1];
                }
                distances[pos] = distance;
                ids[pos] = v;
                done[pos] = 0;
            }
            parent = GC_INVALID;
            bool halt = false;
            if (stop.minimum && size >= 16 && hops >= stop.minimum && hops % stop.interval == 0)
            {
                uint64_t signature = 1469598103934665603ull;
                for (unsigned i = 0; i < 16; ++i)
                    signature = phms_nav_signature_add(signature, ids[i]);
                float frontier = FLT_MAX;
                for (unsigned i = 0; i < size; ++i)
                    if (!done[i])
                    {
                        frontier = distances[i];
                        break;
                    }
                halt = phms_nav_should_stop(stop, hops, signature, frontier, distances[15],
                                            stop_state);
            }
            if (!halt && step < budget)
                for (unsigned i = 0; i < size; ++i)
                    if (!done[i])
                    {
                        done[i] = 1;
                        parent = ids[i];
                        ++hops;
                        break;
                    }
        }
        __syncthreads();
        if (parent == GC_INVALID)
            break;
    }
    if (t == 0)
    {
        unsigned votes[8] = {};
        unsigned rank[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        float mins[8] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
        for (unsigned i = 0; i < size; ++i)
        {
            unsigned g = entries[2 * ids[i]];
            mins[g] = fminf(mins[g], distances[i]);
            if (i < 16)
                ++votes[g];
        }
        for (unsigned i = 0; i < PHMS_GC_GROUPS; ++i)
            for (unsigned j = i + 1; j < PHMS_GC_GROUPS; ++j)
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
            m = phms_adaptive_mask(rank, votes, mins, policy);
        unsigned entrymask = 1u << rank[0];
        if (__popc(m) > 1)
            entrymask |= 1u << rank[1];
        GcRegionChoice best = {FLT_MAX, GC_INVALID, GC_INVALID, GC_INVALID};
        GcRegionMin minimum;
        for (unsigned i = 0; i < size; ++i)
        {
            unsigned v = ids[i];
            unsigned g = entries[2 * v];
            if (entrymask & (1u << g))
                best = minimum(best, GcRegionChoice{distances[i], g, v, entries[2 * v + 1]});
        }
        mask[qid] = m;
        seed[qid] = best.id;
        if (statistics)
        {
            statistics[2 * qid] = evals;
            statistics[2 * qid + 1] = hops;
        }
    }
}

__device__ __forceinline__ void gc_nav_sync()
{
    if (blockDim.x == 32)
        __syncwarp();
    else
        __syncthreads();
}
__device__ __forceinline__ unsigned gc_nav_dot4(unsigned a, unsigned b, unsigned sum)
{
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, sum);
#else
    for (unsigned k = 0; k < 4; ++k)
        sum += ((a >> (8 * k)) & 255) * ((b >> (8 * k)) & 255);
    return sum;
#endif
}
__global__ void gc_navigation_graph_parallel(
    const float* queries, const float* points, const unsigned* entries, const unsigned* adj,
    const unsigned* starts, unsigned* mask, unsigned* seed, unsigned* statistics, unsigned count,
    unsigned degree, unsigned dimension, unsigned beam, unsigned budget, unsigned cards,
    bool adaptive, PhmsAdaptiveCards policy, PhmsNavStop stop = PhmsNavStop(),
    bool cooperative = false, bool integer_points = false, const unsigned* packed_points = nullptr,
    const unsigned* point_norm = nullptr, unsigned fp32_lanes = 1, unsigned probe_cards = 0,
    const unsigned* point_membership = nullptr, float replica_target = 0.75f,
    unsigned entry_policy = 0, const unsigned* fixed_mask = nullptr,
    float* seed_distances = nullptr, unsigned* seed_indices = nullptr)
{
    const unsigned t = threadIdx.x;
    const unsigned qid = blockIdx.x;
    extern __shared__ unsigned seen[];
    float* cached_query = reinterpret_cast<float*>(seen + (count + 31) / 32);
    const float* query =
        dimension <= GC_NAV_QUERY_CACHE ? cached_query : queries + uint64_t(qid) * dimension;
    __shared__ unsigned long long keys[128], next[128], pending[64];
    __shared__ unsigned done[128], next_done[128], parent, evals, hops;
    __shared__ int integer_query;
    __shared__ unsigned query_words[32], query_norm;
    PhmsNavStopState stop_state;
    constexpr unsigned long long invalid = 0xffffffffffffffffull;
    const bool want_integer = cooperative || packed_points;
    for (unsigned w = t; w < (count + 31) / 32; w += blockDim.x)
        seen[w] = 0;
    bool safe = integer_points && dimension <= 128;
    for (unsigned j = t; j < dimension; j += blockDim.x)
    {
        float x = queries[uint64_t(qid) * dimension + j];
        if (dimension <= GC_NAV_QUERY_CACHE)
            cached_query[j] = x;
        if (want_integer)
            safe = safe && (x >= 0 && x <= 255 && x == truncf(x));
    }
    if (want_integer)
    {
        int all_safe = __syncthreads_and(safe);
        if (t == 0)
            integer_query = all_safe;
    }
    else if (t == 0)
        integer_query = 0;
    for (unsigned j = t; j < beam; j += blockDim.x)
    {
        keys[j] = invalid;
        done[j] = 0;
    }
    if (t == 0)
    {
        evals = 0;
        hops = 0;
        parent = GC_INVALID;
    }
    gc_nav_sync();
    if (integer_query && packed_points)
    {
        if (t == 0)
            query_norm = 0;
        gc_nav_sync();
        for (unsigned j = t; j < (dimension + 3) / 4; j += blockDim.x)
        {
            unsigned word = 0;
            unsigned norm = 0;
            for (unsigned k = 0; k < 4; ++k)
                if (j * 4 + k < dimension)
                {
                    unsigned value = static_cast<unsigned>(query[j * 4 + k]);
                    word |= value << (k * 8);
                    norm += value * value;
                }
            query_words[j] = word;
            atomicAdd(&query_norm, norm);
        }
        gc_nav_sync();
    }
    for (unsigned step = 0; step <= budget; ++step)
    {
        const unsigned width = step ? degree : 4;
        if (integer_query && packed_points)
        {
            const unsigned words = (dimension + 3) / 4;
            for (unsigned j = t; j < width; j += blockDim.x)
            {
                unsigned v = step ? adj[uint64_t(parent) * degree + j] : starts[j],
                         bit = 1u << (v % 32);
                bool fresh = !(atomicOr(seen + v / 32, bit) & bit);
                unsigned long long key = invalid;
                if (fresh)
                {
                    unsigned product = 0;
                    for (unsigned k = 0; k < words; ++k)
                        product = gc_nav_dot4(query_words[k],
                                              packed_points[uint64_t(v) * words + k], product);
                    float distance = static_cast<float>(query_norm + point_norm[v] - 2 * product);
                    key = (static_cast<unsigned long long>(__float_as_uint(distance)) << 32) | v;
                    atomicAdd(&evals, 1u);
                }
                pending[j] = key;
            }
        }
        else if (integer_query || fp32_lanes == 4)
        {
            const unsigned lane = t % 4;
            const unsigned group = t / 4;
            for (unsigned j = group; j < width; j += blockDim.x / 4)
            {
                unsigned v = step ? adj[uint64_t(parent) * degree + j] : starts[j];
                const unsigned quad_mask = 0xfu << ((t % 32) & ~3u);
                unsigned fresh = 0;
                if (lane == 0)
                {
                    unsigned bit = 1u << (v % 32);
                    fresh = !(atomicOr(seen + v / 32, bit) & bit);
                }
                fresh = __shfl_sync(quad_mask, fresh, (t % 32) & ~3u);
                float distance = 0;
                if (fresh)
                    distance = gc_nav_quad_fp32(query, points + uint64_t(v) * dimension, dimension,
                                                lane, quad_mask);
                if (lane == 0)
                {
                    pending[j] =
                        fresh
                            ? ((static_cast<unsigned long long>(__float_as_uint(distance)) << 32) |
                               v)
                            : invalid;
                    if (fresh)
                        atomicAdd(&evals, 1u);
                }
            }
        }
        else
            for (unsigned j = t; j < width; j += blockDim.x)
            {
                unsigned v = step ? adj[uint64_t(parent) * degree + j] : starts[j];
                unsigned bit = 1u << (v % 32);
                bool fresh = !(atomicOr(seen + v / 32, bit) & bit);
                unsigned long long key = invalid;
                if (fresh)
                {
                    float distance = 0;
                    for (unsigned k = 0; k < dimension; ++k)
                    {
                        float a = query[k] - points[uint64_t(v) * dimension + k];
                        distance = fmaf(a, a, distance);
                    }
                    key = (static_cast<unsigned long long>(__float_as_uint(distance)) << 32) | v;
                    atomicAdd(&evals, 1u);
                }
                pending[j] = key;
            }
        for (unsigned j = t; j < beam; j += blockDim.x)
        {
            next[j] = invalid;
            next_done[j] = 0;
        }
        gc_nav_sync();
        for (unsigned i = t; i < beam; i += blockDim.x)
            if (keys[i] != invalid)
            {
                unsigned rank = i;
                for (unsigned j = 0; j < width; ++j)
                    rank += pending[j] < keys[i];
                if (rank < beam)
                {
                    next[rank] = keys[i];
                    next_done[rank] = done[i];
                }
            }
        for (unsigned i = t; i < width; i += blockDim.x)
            if (pending[i] != invalid)
            {
                unsigned lo = 0;
                unsigned hi = beam;
                auto key = pending[i];
                while (lo < hi)
                {
                    unsigned mid = (lo + hi) / 2;
                    if (keys[mid] < key)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                unsigned rank = lo;
                for (unsigned j = 0; j < width; ++j)
                    rank += pending[j] < key;
                if (rank < beam)
                {
                    next[rank] = key;
                    next_done[rank] = 0;
                }
            }
        gc_nav_sync();
        for (unsigned j = t; j < beam; j += blockDim.x)
        {
            keys[j] = next[j];
            done[j] = next_done[j];
        }
        gc_nav_sync();
        if (t == 0)
        {
            parent = GC_INVALID;
            bool halt = false;
            if (stop.minimum && keys[15] != invalid && hops >= stop.minimum &&
                hops % stop.interval == 0)
            {
                uint64_t signature = 1469598103934665603ull;
                for (unsigned i = 0; i < 16; ++i)
                    signature = phms_nav_signature_add(signature, static_cast<unsigned>(keys[i]));
                float frontier = FLT_MAX;
                for (unsigned i = 0; i < beam; ++i)
                    if (keys[i] != invalid && !done[i])
                    {
                        frontier = __uint_as_float(keys[i] >> 32);
                        break;
                    }
                halt = phms_nav_should_stop(stop, hops, signature, frontier,
                                            __uint_as_float(keys[15] >> 32), stop_state);
            }
            if (!halt && step < budget)
                for (unsigned i = 0; i < beam; ++i)
                    if (keys[i] != invalid && !done[i])
                    {
                        done[i] = 1;
                        parent = static_cast<unsigned>(keys[i]);
                        ++hops;
                        break;
                    }
        }
        gc_nav_sync();
        if (parent == GC_INVALID)
            break;
    }
    if (t == 0)
    {
        unsigned votes[8] = {};
        unsigned rank[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        float mins[8] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
        for (unsigned i = 0; i < beam && keys[i] != invalid; ++i)
        {
            unsigned v = static_cast<unsigned>(keys[i]);
            unsigned g = entries[2 * v];
            float d = __uint_as_float(keys[i] >> 32);
            mins[g] = fminf(mins[g], d);
            if (i < 16)
                ++votes[g];
        }
        for (unsigned i = 0; i < PHMS_GC_GROUPS; ++i)
            for (unsigned j = i + 1; j < PHMS_GC_GROUPS; ++j)
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
            m = phms_adaptive_mask(rank, votes, mins, policy);
        unsigned entrymask = 1u << rank[0];
        if (__popc(m) > 1)
            entrymask |= 1u << rank[1];
        if (entry_policy)
            entrymask = m;
        GcRegionChoice best = {FLT_MAX, GC_INVALID, GC_INVALID, GC_INVALID};
        GcRegionMin minimum;
        if (fixed_mask)
        {
            m = fixed_mask[qid];
            entrymask = m;
            const unsigned index = seed_indices[qid];
            best = GcRegionChoice{seed_distances[qid], entries[2 * index], index, seed[qid]};
        }
        for (unsigned i = 0; i < beam && keys[i] != invalid; ++i)
        {
            unsigned v = static_cast<unsigned>(keys[i]), g = entries[2 * v];
            if (entrymask & (1u << g))
                best = minimum(
                    best, GcRegionChoice{__uint_as_float(keys[i] >> 32), g, v, entries[2 * v + 1]});
        }
        if (probe_cards)
        {
            m = 0;
            for (unsigned i = 0; i < probe_cards; ++i)
                m |= 1u << rank[i];
        }
        if (point_membership)
        {
            unsigned near_masks[16] = {};
            unsigned n = 0;
            unsigned required = 0;
            float replica_mins[8] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
                                     FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
            for (unsigned i = 0; i < beam && keys[i] != invalid; ++i)
            {
                unsigned v = static_cast<unsigned>(keys[i]);
                unsigned membership = point_membership[v];
                float dist = __uint_as_float(keys[i] >> 32);
                for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                    if (membership & (1u << g))
                        replica_mins[g] = fminf(replica_mins[g], dist);
                if (i < 16)
                {
                    near_masks[n++] = membership;
                    if (policy.distance_ratio > 0 &&
                        dist <= __uint_as_float(keys[0] >> 32) * policy.distance_ratio)
                        required |= 1u << i;
                }
            }
            PhmsAdaptiveCards copy_policy = policy;
            copy_policy.coverage = replica_target;
            m = phms_replica_mask(near_masks, n, replica_mins, point_membership[best.index],
                                  copy_policy, required);
        }
        if (seed_distances)
            seed_distances[qid] = best.distance;
        if (seed_indices)
            seed_indices[qid] = best.index;
        mask[qid] = m;
        seed[qid] = best.id;
        if (statistics)
        {
            statistics[2 * qid] = evals;
            statistics[2 * qid + 1] = hops;
        }
    }
}
