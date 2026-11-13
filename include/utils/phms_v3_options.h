#ifndef PHMS_V3_OPTIONS_H
#define PHMS_V3_OPTIONS_H
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace phms_v3
{
#ifdef __CUDACC__
#define PHMS_V3_HD __host__ __device__
#else
#define PHMS_V3_HD
#endif
template <unsigned K>
PHMS_V3_HD inline void offer_parent(unsigned id, float distance, unsigned (&ids)[K],
                                    float (&distances)[K])
{
    if (!(distance < 3.402823E+38f))
        return;
    for (unsigned i = 0; i < K; ++i)
        if (ids[i] == id)
        {
            if (distance < distances[i])
                distances[i] = distance;
            return;
        }
    for (unsigned i = 0; i < K; ++i)
        if (distance < distances[i])
        {
            for (unsigned j = K - 1; j > i; --j)
            {
                ids[j] = ids[j - 1];
                distances[j] = distances[j - 1];
            }
            ids[i] = id;
            distances[i] = distance;
            return;
        }
}
#undef PHMS_V3_HD
inline unsigned unsigned_value(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("v3 budget must be an unsigned decimal integer");
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<unsigned>::max())
        throw std::runtime_error("v3 budget overflow");
    return static_cast<unsigned>(value);
}
inline unsigned checked_budget(unsigned requested, unsigned worklist)
{
    if (!requested)
        return worklist + 50u;
    if (requested < 2 || requested > worklist + 50u)
        throw std::runtime_error("v3 budget must be 2..L+50, or 0 for the original budget");
    return requested;
}
inline unsigned expansion_budget(unsigned worklist)
{
    const char* value = std::getenv("PHMS_V3_EXPANSION_BUDGET");
    return checked_budget(value && *value ? unsigned_value(value) : 0, worklist);
}
inline std::vector<unsigned> budget_schedule()
{
    const char* value = std::getenv("PHMS_V3_BUDGET_SCHEDULE");
    std::vector<unsigned> result;
    if (!value || !*value)
        return result;
    const std::string text(value);
    size_t begin = 0;
    for (;;)
    {
        const size_t end = text.find(',', begin);
        const unsigned item = unsigned_value(text.substr(begin, end - begin));
        if (item == 1)
            throw std::runtime_error("v3 budget cannot be 1");
        result.push_back(item);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return result;
}
inline bool warp_sort_enabled()
{
    const char* value = std::getenv("PHMS_V3_WARP_SORT");
    return value && std::atoi(value) != 0;
}
inline unsigned parent_pipeline()
{
    const char* value = std::getenv("PHMS_V3_PARENT_PIPELINE");
    const unsigned mode = value && *value ? unsigned_value(value) : 0;
    if (mode > 5)
        throw std::runtime_error("v3 parent pipeline must be 0..5");
    return mode;
}
inline bool rotate_submit()
{
    const char* value = std::getenv("PHMS_V3_ROTATE_SUBMIT");
    const unsigned mode = value && *value ? unsigned_value(value) : 0;
    if (mode > 1)
        throw std::runtime_error("v3 rotate submit must be 0 or 1");
    return mode != 0;
}
inline unsigned final_candidates()
{
    const char* value = std::getenv("PHMS_V3_FINAL_CANDIDATES");
    const unsigned n = value && *value ? unsigned_value(value) : 0;
    if (n > 2)
        throw std::runtime_error("v3 final candidates must be0..2 per GPU");
    return n;
}
inline int append_candidate(unsigned* track, unsigned queries, unsigned q, unsigned capacity,
                            unsigned& count, unsigned id, unsigned dataset_size)
{
    if (id >= dataset_size)
        return -1;
    for (unsigned i = 0; i < count; ++i)
        if (track[(size_t)i * queries + q] == id)
            return 0;
    if (count >= capacity)
        return -1;
    track[(size_t)count++ * queries + q] = id;
    return 1;
}
} // namespace phms_v3
#endif
