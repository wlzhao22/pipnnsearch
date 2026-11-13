#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
struct PhmsGcHeader
{
    uint64_t magic;
    uint32_t version;
    uint32_t nodes;
    uint32_t chunks;
    uint32_t card;
    uint64_t words;
    uint64_t local_nodes;
};
static_assert(sizeof(PhmsGcHeader) == 40, "GraphCut disk schema");
inline void phms_gc_validate(const PhmsGcHeader& h, uint64_t bytes, unsigned card, unsigned nodes,
                             unsigned chunks)
{
    if (h.magic != 0x3143515043474e42ull || h.version != 1 || h.card != card || h.card >= 8 ||
        h.nodes != nodes || h.chunks != chunks || h.words != (uint64_t(nodes) + 63) / 64 ||
        h.local_nodes > nodes || bytes != 40 + h.words * 12 + h.local_nodes * chunks)
        throw std::runtime_error("GraphCut PQ schema/index mismatch");
}
inline uint64_t phms_gc_rank(unsigned id, const uint64_t* bits, const unsigned* prefix)
{
    const unsigned w = id / 64;
    const unsigned b = id % 64;
    if (!(bits[w] & (uint64_t(1) << b)))
        throw std::runtime_error("nonmember rank lookup");
    return uint64_t(prefix[w]) + __builtin_popcountll(bits[w] & ((uint64_t(1) << b) - 1));
}
inline std::vector<unsigned>
phms_gc_seed_order(const std::vector<unsigned>& seeds,
                   const std::unordered_map<unsigned, unsigned>& categories)
{
    std::vector<unsigned> counts(categories.size());
    std::vector<unsigned> category(seeds.size());
    std::vector<unsigned> order(seeds.size());
    for (unsigned q = 0; q < seeds.size(); ++q)
    {
        auto it = categories.find(seeds[q]);
        if (it == categories.end() || it->second >= counts.size())
            throw std::runtime_error("unknown seed category");
        category[q] = it->second;
        ++counts[category[q]];
    }
    unsigned offset = 0;
    for (unsigned& n : counts)
    {
        unsigned count = n;
        n = offset;
        offset += count;
    }
    for (unsigned q = 0; q < seeds.size(); ++q)
        order[counts[category[q]]++] = q;
    return order;
}
struct PhmsGcBatch
{
    unsigned mask;
    std::vector<unsigned> ids;
};
inline unsigned phms_gc_coalesce_masks(std::vector<unsigned>& masks, unsigned minimum_queries,
                                       unsigned mode = 0, unsigned maximum_cards = PHMS_GC_GROUPS)
{
    if (mode > 4)
        throw std::runtime_error("invalid coalesce mode");
    if (!maximum_cards || maximum_cards > PHMS_GC_GROUPS)
        throw std::runtime_error("invalid coalesce card limit");
    const unsigned support_min =
        mode >= 3 ? std::max(1u, minimum_queries / (mode == 3 ? 2 : 4)) : minimum_queries;
    std::array<unsigned, 256> counts{};
    const unsigned full = (1u << PHMS_GC_GROUPS) - 1;
    for (unsigned m : masks)
    {
        if (!m || m > full)
            throw std::runtime_error("invalid route mask");
        ++counts[m];
    }
    if (maximum_cards < PHMS_GC_GROUPS)
    {
        for (unsigned m : masks)
            if (unsigned(__builtin_popcount(m)) > maximum_cards)
                throw std::runtime_error("input route exceeds coalesce card limit");
        if (!minimum_queries)
            return 0;
        std::array<unsigned, 256> destination{};
        std::array<bool, 256> assigned{};
        std::vector<unsigned> pending;
        for (unsigned m = 1; m <= full; ++m)
            if (counts[m])
            {
                destination[m] = m;
                if (__builtin_popcount(m) >= 2 && counts[m] < minimum_queries)
                    pending.push_back(m);
                else
                    assigned[m] = true;
            }
        std::stable_sort(pending.begin(), pending.end(),
                         [&](unsigned a, unsigned b)
                         { return counts[a] != counts[b] ? counts[a] > counts[b] : a < b; });
        for (unsigned m : pending)
        {
            if (assigned[m])
                continue;
            unsigned best = m;
            unsigned best_support = 0;
            for (unsigned candidate = 1; candidate <= full; ++candidate)
            {
                if ((candidate & m) != m || unsigned(__builtin_popcount(candidate)) > maximum_cards)
                    continue;
                unsigned support = counts[candidate];
                if (mode >= 2)
                    for (unsigned r : pending)
                        if (r != candidate && !assigned[r] && (r & candidate) == r)
                            support += counts[r];
                if (support < support_min)
                    continue;
                if (!best_support || __builtin_popcount(candidate) < __builtin_popcount(best) ||
                    (__builtin_popcount(candidate) == __builtin_popcount(best) &&
                     (support > best_support || (support == best_support && candidate < best))))
                {
                    best = candidate;
                    best_support = support;
                }
            }
            if (mode >= 2 && best_support)
            {
                for (unsigned r : pending)
                    if (!assigned[r] && (r & best) == r)
                    {
                        destination[r] = best;
                        assigned[r] = true;
                    }
            }
            else
            {
                destination[m] = best;
                assigned[m] = true;
            }
        }
        unsigned promoted = 0;
        for (unsigned& m : masks)
            if (destination[m] != m)
            {
                m = destination[m];
                ++promoted;
            }
        return promoted;
    }
    if (!mode)
    {
        unsigned promoted = 0;
        for (unsigned& m : masks)
            if (m != full && __builtin_popcount(m) >= 2 && counts[m] < minimum_queries)
            {
                m = full;
                ++promoted;
            }
        return promoted;
    }
    std::array<unsigned, 256> destination{}, active{};
    std::array<bool, 256> assigned{};
    std::vector<unsigned> pending;
    for (unsigned m = 1; m <= full; ++m)
        if (counts[m])
        {
            destination[m] = m;
            if (m != full && __builtin_popcount(m) >= 2 && counts[m] < minimum_queries)
                pending.push_back(m);
            else
            {
                active[m] = counts[m];
                assigned[m] = true;
            }
        }
    std::stable_sort(pending.begin(), pending.end(),
                     [&](unsigned a, unsigned b)
                     { return counts[a] != counts[b] ? counts[a] > counts[b] : a < b; });
    for (unsigned m : pending)
    {
        if (assigned[m])
            continue;
        unsigned best = full;
        unsigned best_support = active[full];
        auto consider = [&](unsigned candidate, unsigned support)
        {
            if ((candidate & m) != m || support < support_min)
                return;
            const int size = __builtin_popcount(candidate);
            const int best_size = __builtin_popcount(best);
            if (size < best_size ||
                (size == best_size &&
                 (support > best_support || (support == best_support && candidate < best))))
            {
                best = candidate;
                best_support = support;
            }
        };
        if (mode)
            for (unsigned candidate = 1; candidate <= full; ++candidate)
                if (active[candidate])
                    consider(candidate, active[candidate]);
        if (mode >= 2)
        {
            for (unsigned other : pending)
            {
                if (assigned[other])
                    continue;
                const unsigned candidate = m | other;
                if (__builtin_popcount(candidate) > 4)
                    continue;
                unsigned support = active[candidate];
                for (unsigned r : pending)
                    if (!assigned[r] && (r & candidate) == r)
                        support += counts[r];
                consider(candidate, support);
            }
        }
        if (mode >= 2 && best != full)
        {
            for (unsigned r : pending)
                if (!assigned[r] && (r & best) == r)
                {
                    destination[r] = best;
                    assigned[r] = true;
                    active[best] += counts[r];
                }
        }
        else
        {
            destination[m] = best;
            assigned[m] = true;
            active[best] += counts[m];
        }
    }
    unsigned promoted = 0;
    for (unsigned& m : masks)
        if (destination[m] != m)
        {
            m = destination[m];
            ++promoted;
        }
    return promoted;
}
inline std::vector<PhmsGcBatch> phms_gc_balanced_batches(const std::vector<unsigned>& masks,
                                                         unsigned workers)
{
    if (!workers || masks.size() < workers)
        throw std::runtime_error("invalid worker/query count");
    std::array<std::vector<unsigned>, 256> groups;
    for (unsigned q = 0; q < masks.size(); ++q)
    {
        if (!masks[q] || masks[q] >= 256)
            throw std::runtime_error("invalid route mask");
        groups[masks[q]].push_back(q);
    }
    unsigned nonempty = 0;
    unsigned maximum = 0;
    for (auto& g : groups)
    {
        nonempty += !g.empty();
        maximum = std::max(maximum, unsigned(g.size()));
    }
    unsigned lo = 1;
    unsigned hi = maximum;
    const unsigned target = std::max(workers, nonempty);
    while (lo < hi)
    {
        unsigned cap = lo + (hi - lo) / 2;
        unsigned needed = 0;
        for (auto& g : groups)
            needed += (g.size() + cap - 1) / cap;
        if (needed <= target)
            hi = cap;
        else
            lo = cap + 1;
    }
    std::array<unsigned, 256> seats{};
    unsigned used = 0;
    for (unsigned m = 1; m < 256; ++m)
        if (!groups[m].empty())
        {
            seats[m] = (groups[m].size() + lo - 1) / lo;
            used += seats[m];
        }
    while (used < target)
    {
        unsigned best = 0;
        for (unsigned m = 1; m < 256; ++m)
            if (seats[m] < groups[m].size() &&
                (!best || uint64_t(groups[m].size()) * seats[best] >
                              uint64_t(groups[best].size()) * seats[m]))
                best = m;
        if (!best)
            break;
        ++seats[best];
        ++used;
    }
    std::vector<PhmsGcBatch> jobs;
    for (unsigned m = 1; m < 256; ++m)
        if (seats[m])
        {
            unsigned begin = 0;
            for (unsigned i = 0; i < seats[m]; ++i)
            {
                unsigned count = groups[m].size() / seats[m] + (i < groups[m].size() % seats[m]);
                jobs.push_back({m, {groups[m].begin() + begin, groups[m].begin() + begin + count}});
                begin += count;
            }
        }
    std::stable_sort(jobs.begin(), jobs.end(),
                     [](const PhmsGcBatch& a, const PhmsGcBatch& b)
                     { return a.ids.size() > b.ids.size(); });
    return jobs;
}
inline std::vector<PhmsGcBatch> phms_gc_cost_batches(const std::vector<unsigned>& masks,
                                                     unsigned workers, unsigned mode)
{
    if (!workers || masks.size() < workers || mode < 1 || mode > 2)
        throw std::runtime_error("invalid cost batch options");
    std::array<std::vector<unsigned>, 256> groups;
    for (unsigned q = 0; q < masks.size(); ++q)
    {
        unsigned m = masks[q];
        if (!m || m >= (1u << PHMS_GC_GROUPS))
            throw std::runtime_error("invalid cost batch mask");
        groups[m].push_back(q);
    }
    std::array<unsigned, 256> seats{};
    unsigned used = 0;
    auto cost = [&](unsigned m, unsigned s)
    {
        unsigned cards = __builtin_popcount(m);
        return uint64_t((groups[m].size() + s - 1) / s) * (4 + mode * (cards - 1)) +
               (mode == 2 ? 128ull * (cards - 1) : 0ull);
    };
    for (unsigned m = 1; m < 256; ++m)
        if (!groups[m].empty())
        {
            seats[m] = 1;
            ++used;
        }
    const unsigned target = std::max(workers, used);
    while (used < target)
    {
        unsigned best = 0;
        for (unsigned m = 1; m < 256; ++m)
            if (seats[m] && seats[m] < groups[m].size())
                if (!best || cost(m, seats[m]) > cost(best, seats[best]))
                    best = m;
        if (!best)
            break;
        ++seats[best];
        ++used;
    }
    std::vector<PhmsGcBatch> out;
    for (unsigned m = 1; m < 256; ++m)
        if (seats[m])
        {
            unsigned begin = 0;
            for (unsigned s = 0; s < seats[m]; ++s)
            {
                unsigned n = groups[m].size() / seats[m] + (s < groups[m].size() % seats[m]);
                out.push_back({m, {groups[m].begin() + begin, groups[m].begin() + begin + n}});
                begin += n;
            }
        }
    std::stable_sort(out.begin(), out.end(),
                     [&](const PhmsGcBatch& a, const PhmsGcBatch& b)
                     {
                         auto score = [&](const PhmsGcBatch& j)
                         {
                             unsigned k = __builtin_popcount(j.mask);
                             return uint64_t(j.ids.size()) * (4 + mode * (k - 1)) +
                                    (mode == 2 ? 128ull * (k - 1) : 0ull);
                         };
                         return score(a) > score(b);
                     });
    return out;
}
inline std::vector<PhmsGcBatch> phms_gc_batches(const std::vector<unsigned>& masks,
                                                unsigned workers, unsigned max_queries = 0)
{
    if (!workers || masks.size() < workers)
        throw std::runtime_error("invalid worker/query count");
    std::array<std::vector<unsigned>, 256> groups;
    for (unsigned q = 0; q < masks.size(); ++q)
    {
        if (!masks[q] || masks[q] >= 256)
            throw std::runtime_error("invalid route mask");
        groups[masks[q]].push_back(q);
    }
    std::vector<PhmsGcBatch> jobs;
    for (unsigned m = 1; m < 256; ++m)
        if (!groups[m].empty())
            jobs.push_back({m, std::move(groups[m])});
    if (max_queries)
    {
        std::vector<PhmsGcBatch> chunks;
        for (auto& job : jobs)
            for (size_t begin = 0; begin < job.ids.size(); begin += max_queries)
            {
                size_t end = std::min(job.ids.size(), begin + max_queries);
                chunks.push_back({job.mask, {job.ids.begin() + begin, job.ids.begin() + end}});
            }
        jobs = std::move(chunks);
    }
    while (jobs.size() < workers)
    {
        auto it = std::max_element(jobs.begin(), jobs.end(),
                                   [](const PhmsGcBatch& a, const PhmsGcBatch& b)
                                   { return a.ids.size() < b.ids.size(); });
        if (it->ids.size() < 2)
            break;
        PhmsGcBatch tail{it->mask, {it->ids.begin() + it->ids.size() / 2, it->ids.end()}};
        it->ids.resize(it->ids.size() / 2);
        jobs.push_back(std::move(tail));
    }
    std::stable_sort(jobs.begin(), jobs.end(),
                     [](const PhmsGcBatch& a, const PhmsGcBatch& b)
                     { return a.ids.size() > b.ids.size(); });
    return jobs;
}
