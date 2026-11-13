#pragma once
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include "phms_data.h"

#include <cstdint>

#ifdef __CUDACC__
#define PHMS_NAV_HD __host__ __device__
#else
#define PHMS_NAV_HD
#endif
struct PhmsNavStop
{
    unsigned minimum = 0;
    unsigned interval = 4;
    unsigned stable_checks = 2;
    float frontier_ratio = 1.0f;
};
struct PhmsNavStopState
{
    uint64_t previous = 0;
    unsigned stable = 0;
    bool have_previous = false;
};
PHMS_NAV_HD inline uint64_t phms_nav_signature_add(uint64_t hash, unsigned id)
{
    return (hash ^ id) * 1099511628211ull;
}
PHMS_NAV_HD inline bool phms_nav_should_stop(const PhmsNavStop& policy, unsigned hops,
                                             uint64_t signature, float frontier, float cutoff,
                                             PhmsNavStopState& state)
{
    if (!policy.minimum || hops < policy.minimum || hops % policy.interval)
        return false;
    if (state.have_previous && state.previous == signature)
        ++state.stable;
    else
        state.stable = 0;
    state.previous = signature;
    state.have_previous = true;
    return state.stable >= policy.stable_checks && frontier > cutoff * policy.frontier_ratio;
}
#undef PHMS_NAV_HD

inline PhmsNavStop navigation_stop_from_env()
{
    PhmsNavStop stop;
    if (!getenv("PHMS_V6_NAV_EARLY_STOP") || !atoi(getenv("PHMS_V6_NAV_EARLY_STOP")))
        return stop;
    stop.minimum = getenv("PHMS_V6_NAV_STOP_MIN") ? std::stoul(getenv("PHMS_V6_NAV_STOP_MIN")) : 16;
    stop.interval =
        getenv("PHMS_V6_NAV_STOP_INTERVAL") ? std::stoul(getenv("PHMS_V6_NAV_STOP_INTERVAL")) : 4;
    stop.stable_checks =
        getenv("PHMS_V6_NAV_STOP_STABLE") ? std::stoul(getenv("PHMS_V6_NAV_STOP_STABLE")) : 2;
    stop.frontier_ratio =
        getenv("PHMS_V6_NAV_STOP_RATIO") ? std::stof(getenv("PHMS_V6_NAV_STOP_RATIO")) : 1.f;
    need(stop.minimum >= 16 && stop.interval > 0 && stop.stable_checks > 0 &&
             std::isfinite(stop.frontier_ratio) && stop.frontier_ratio >= 1.f,
         "invalid navigation stopping policy");
    return stop;
}

struct NavigationGraph
{
    Map adjacency;
    Map starts;
    unsigned count;
    unsigned degree;
    NavigationGraph(const std::string& dir, unsigned n)
        : adjacency(dir + "/adjacency.ubin"), starts(dir + "/starts.ubin"), count(n)
    {
        degree = adjacency.at<unsigned>(4);
        need(adjacency.at<unsigned>(0) == n && degree >= 2 && degree <= 64 &&
                 adjacency.bytes == 8 + uint64_t(n) * degree * 4,
             "bad navigation edges");
        need(starts.bytes == 24 && starts.at<unsigned>(0) == 1 && starts.at<unsigned>(4) == 4,
             "bad navigation starts");
        for (unsigned i = 0; i < 4; ++i)
            need(start(i) < n, "invalid navigation start");
        for (unsigned u = 0; u < n; ++u)
        {
            std::unordered_set<unsigned> unique;
            for (unsigned j = 0; j < degree; ++j)
            {
                auto v = edges(u)[j];
                need(v < n && v != u && unique.insert(v).second, "bad navigation neighbor");
            }
        }
    }
    unsigned start(unsigned i) const
    {
        return starts.at<unsigned>(8 + i * 4);
    }
    const unsigned* edges(unsigned u) const
    {
        return (const unsigned*)(adjacency.data + 8) + uint64_t(u) * degree;
    }
};
