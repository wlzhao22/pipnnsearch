#pragma once

#ifdef __CUDACC__
#define PHMS_ROUTE_HD __host__ __device__
#else
#define PHMS_ROUTE_HD
#endif
struct PhmsAdaptiveCards
{
    unsigned minimum = 2;
    unsigned maximum = 4;
    float coverage = 0.9375f;
    float distance_ratio = 1.10f;
};
PHMS_ROUTE_HD inline unsigned phms_adaptive_mask(const unsigned* rank, const unsigned* votes,
                                                 const float* distance, PhmsAdaptiveCards p)
{
    unsigned count = p.minimum;
    unsigned total = 0;
    for (unsigned i = 0; i < count; ++i)
        total += votes[rank[i]];
    while (count < p.maximum && float(total) < 16.0f * p.coverage)
        total += votes[rank[count++]];
    if (p.distance_ratio > 0)
    {
        float best = distance[0];
        for (unsigned g = 1; g < PHMS_GC_GROUPS; ++g)
            if (distance[g] < best)
                best = distance[g];
        for (unsigned i = count; i < p.maximum; ++i)
            if (distance[rank[i]] <= best * p.distance_ratio)
                count = i + 1;
    }
    unsigned mask = 0;
    for (unsigned i = 0; i < count; ++i)
        mask |= 1u << rank[i];
    return mask;
}
PHMS_ROUTE_HD inline unsigned phms_route_bitcount(unsigned x)
{
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
PHMS_ROUTE_HD inline unsigned phms_replica_mask(const unsigned* point_masks, unsigned n,
                                                const float* mins, unsigned entry_mask,
                                                PhmsAdaptiveCards policy, unsigned required = 0)
{
    unsigned coverage[8] = {};
    unsigned selected = 0;
    unsigned covered = 0;
    unsigned count = 0;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
            if (point_masks[i] & (1u << g))
                coverage[g] |= 1u << i;
    while (count < policy.maximum)
    {
        if (count >= policy.minimum &&
            float(phms_route_bitcount(covered)) >= float(n) * policy.coverage &&
            !(required & ~covered))
            break;
        unsigned best = PHMS_GC_GROUPS;
        unsigned gain = 0;
        for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
            if (!(selected & (1u << g)) && (count || (entry_mask & (1u << g))))
            {
                unsigned value = phms_route_bitcount(coverage[g] & ~covered);
                if (best == PHMS_GC_GROUPS || value > gain ||
                    (value == gain &&
                     (mins[g] < mins[best] || (mins[g] == mins[best] && g < best))))
                {
                    best = g;
                    gain = value;
                }
            }
        if (best == PHMS_GC_GROUPS)
            break;
        selected |= 1u << best;
        covered |= coverage[best];
        ++count;
    }
    return selected;
}
#undef PHMS_ROUTE_HD
