#include "phms.h"
#include "phms_timeline.h"
#ifdef PHMS_TIMELINE
#include <cuda_profiler_api.h>
#endif
#include "phms_graphcut_batches.h"
#include "phms_graphcut_router.cuh"
#include "phms_navigation_graph.cuh"
#include "phms_data.h"
#include "phms_navigation_model.h"
#include "phms_v3_options.h"
#include <cublas_v2.h>
#include <memory>
#include <thread>
#include <atomic>
#include <omp.h>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <condition_variable>
#include <mutex>
static void require_idle_or_self()
{
    FILE* f = popen("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits", "r");
    need(f, "cannot inspect GPU processes");
    char line[256];
    bool other = false;
    while (fgets(line, sizeof(line), f))
    {
        std::string s(line);
        auto a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos)
            continue;
        auto b = s.find_last_not_of(" \t\r\n");
        s = s.substr(a, b - a + 1);
        if (s.find_first_not_of("0123456789") != std::string::npos)
        {
            other = true;
            continue;
        }
        if (std::strtoull(s.c_str(), nullptr, 10) != (unsigned long long)getpid())
            other = true;
    }
    int status = pclose(f);
    need(status == 0 && !other,
         "GPU state unavailable or another compute process exists; benchmark refused");
}

static void ck(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void cb(cublasStatus_t e)
{
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failed");
}
template <class T> struct Buffer
{
    T* p = nullptr;
    size_t n = 0;
    int gpu = 0;
    bool pinned = false;
    void alloc(size_t count, bool host = false)
    {
        n = count;
        pinned = host;
        ck(cudaGetDevice(&gpu));
        if (host)
            ck(cudaMallocHost(&p, n * sizeof(T)));
        else
            ck(cudaMalloc(&p, n * sizeof(T)));
    }
    void copy(const T* h)
    {
        ck(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Buffer()
    {
        if (p)
        {
            cudaSetDevice(gpu);
            if (pinned)
                cudaFreeHost(p);
            else
                cudaFree(p);
        }
    }
};
struct Job
{
    unsigned budget = 0;
    PhmsGcBatch batch;
    std::vector<int> gpus;
    std::vector<std::unique_ptr<PHMSSearch<float>>> objects;
    Buffer<float> queries;
    Buffer<unsigned> seeds;
    std::vector<result_ann_t> ids;
    std::vector<float> dist;
    std::vector<std::vector<unsigned>> tracks;
    std::vector<std::vector<std::vector<unsigned>>> shard_tracks;
    std::atomic<unsigned> remaining{0};
};
struct NavigationReplica
{
    int gpu = 0;
    cudaStream_t stream = nullptr;
    Buffer<float> points;
    Buffer<unsigned> entries;
    Buffer<unsigned> adjacency;
    Buffer<unsigned> starts;
    Buffer<unsigned> packed;
    Buffer<unsigned> norm;
    Buffer<unsigned> masks;
    Buffer<unsigned> seeds;
    Buffer<unsigned> statistics;
    Buffer<unsigned> membership;
    Buffer<unsigned> entry_masks;
    Buffer<unsigned> entry_indices;
    Buffer<unsigned> entry_statistics;
    Buffer<float> entry_distances;
    ~NavigationReplica()
    {
        if (stream)
        {
            cudaSetDevice(gpu);
            cudaStreamDestroy(stream);
        }
    }
};
static double elapsed(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
int main(int argc, char** argv)
{
    try
    {
        need(argc == 11, "usage: v5_graphcut_search PREFIX QUERY GT NQ L WORKERS THREADS "
                         "FIXED_CARDS MODEL1024 OLD256");
        need(getenv("PHMS_GRAPHCUT_LAYOUT") && *getenv("PHMS_GRAPHCUT_LAYOUT"),
             "set PHMS_GRAPHCUT_LAYOUT");
        for (const char* name : {"PHMS_COMPACT_SHARD_UPLOAD", "PHMS_PIPELINE_PARENT_PREP"})
            need(!getenv(name) || !atoi(getenv(name)),
                 std::string("unsupported GraphCut option: ") + name);
        unsigned nq = std::stoul(argv[4]), L = std::stoul(argv[5]), workers = std::stoul(argv[6]),
                 threads = std::stoul(argv[7]), cards = std::stoul(argv[8]);
        need(nq >= workers && workers && threads && cards >= 1 && cards <= PHMS_GC_GROUPS &&
                 L >= 10 && L <= 512,
             "invalid options");
        const unsigned final_candidates = phms_v3::final_candidates();
        const bool independent =
            getenv("PHMS_INDEPENDENT_SEARCH") && atoi(getenv("PHMS_INDEPENDENT_SEARCH"));
#ifndef PHMS_INDEPENDENT_EXPERIMENT
        need(!independent, "independent mode requires the separate experimental core");
#endif
        need(!independent || !final_candidates,
             "independent experiment uses expanded parents only");
        std::array<unsigned, 8> budgets;
        budgets.fill(phms_v3::expansion_budget(L));
        const bool adaptive_budget =
            getenv("PHMS_V5_BUDGET_BY_CARDS") && *getenv("PHMS_V5_BUDGET_BY_CARDS");
        if (adaptive_budget)
        {
            std::stringstream input(getenv("PHMS_V5_BUDGET_BY_CARDS"));
            std::string value;
            unsigned i = 0;
            while (std::getline(input, value, ','))
            {
                need(i < PHMS_GC_GROUPS, "budget schedule requires exactly four values");
                unsigned b = phms_v3::unsigned_value(value);
                need(b >= 2, "explicit per-card budget must be >=2");
                budgets[i++] = phms_v3::checked_budget(b, L);
            }
            need(i == PHMS_GC_GROUPS, "budget schedule requires exactly four values");
        }
        require_idle_or_self();
        int devices = 0;
        ck(cudaGetDeviceCount(&devices));
        need(devices == PHMS_GC_GROUPS, "exactly two visible GPUs required");
        static std::unique_ptr<Data> resident_data;
        static std::string resident_key;
        const std::string key = std::string(argv[1]) + "\n" + argv[2] + "\n" + argv[3];
        if (!resident_data)
        {
            resident_key = key;
            resident_data.reset(new Data(argv[1], argv[2], argv[3]));
        }
        need(key == resident_key, "cannot change dataset in a resident session");
        Data& d = *resident_data;
        need(nq <= d.nq, "too many queries");
        const char* nav_path = getenv("PHMS_V6_NAV_MODEL");
        const bool navigation = nav_path && *nav_path;
        need(navigation, "Scaling driver requires real-point graph navigation");
        Map centers(std::string(navigation ? nav_path : argv[9]) + "/centroids.fbin"),
            entries(std::string(navigation ? nav_path : argv[9]) + "/entries.ubin");
        Map oldcenters(std::string(navigation ? nav_path : argv[10]) + "/centroids.fbin"),
            oldentries(std::string(navigation ? nav_path : argv[10]) + "/entries.ubin");
        const unsigned fine_count = centers.at<unsigned>(0), old_count = navigation ? 0 : 256;
        need((navigation ? (fine_count >= 16 && fine_count <= 65536) : fine_count == 1024) &&
                 centers.at<unsigned>(4) == d.d &&
                 centers.bytes == 8 + uint64_t(fine_count) * d.d * 4,
             "bad router points");
        need(entries.bytes == 8 + uint64_t(fine_count) * 8 &&
                 entries.at<unsigned>(0) == fine_count && entries.at<unsigned>(4) == 2,
             "bad router owners");
        if (!navigation)
        {
            need(oldcenters.at<unsigned>(0) == 256 && oldcenters.at<unsigned>(4) == d.d &&
                     oldcenters.bytes == 8 + 256ull * d.d * 4,
                 "bad entry centers");
            need(oldentries.bytes == 8 + 256ull * 8 && oldentries.at<unsigned>(0) == 256 &&
                     oldentries.at<unsigned>(4) == 2,
                 "bad entry IDs");
        }
        auto* c = (const float*)(centers.data + 8);
        auto* oc = (const float*)(oldcenters.data + 8);
        auto* e = (const unsigned*)(entries.data + 8);
        auto* oe = (const unsigned*)(oldentries.data + 8);
        std::vector<unsigned> owners(fine_count);
        std::array<unsigned, 8> entry_count{};
        for (unsigned j = 0; j < fine_count; ++j)
        {
            need(e[2 * j] < PHMS_GC_GROUPS && e[2 * j + 1] < d.n, "invalid router owner");
            owners[j] = e[2 * j];
        }
        for (unsigned j = 0; j < old_count; ++j)
        {
            need(oe[2 * j] < PHMS_GC_GROUPS && oe[2 * j + 1] < d.n, "invalid entry");
            ++entry_count[oe[2 * j]];
        }
        if (!navigation)
            for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                need(entry_count[g] > 0, "missing entry shard");
        std::vector<float> combined(c, c + uint64_t(fine_count) * d.d);
        if (!navigation)
            combined.insert(combined.end(), oc, oc + 256ull * d.d);
        std::unique_ptr<NavigationGraph> nav;
        unsigned nav_beam =
            getenv("PHMS_V6_NAV_BEAM") ? std::stoul(getenv("PHMS_V6_NAV_BEAM")) : 64;
        unsigned nav_budget =
            getenv("PHMS_V6_NAV_BUDGET") ? std::stoul(getenv("PHMS_V6_NAV_BUDGET")) : 64;
        const bool nav_parallel =
            !getenv("PHMS_V6_NAV_PARALLEL") || atoi(getenv("PHMS_V6_NAV_PARALLEL"));
        const unsigned nav_threads =
            getenv("PHMS_V6_NAV_THREADS") ? std::stoul(getenv("PHMS_V6_NAV_THREADS")) : 128;
        const PhmsNavStop nav_stop = navigation_stop_from_env();
        const bool nav_cooperative =
            getenv("PHMS_V6_NAV_COOPERATIVE") && atoi(getenv("PHMS_V6_NAV_COOPERATIVE"));
        const bool nav_packed = getenv("PHMS_V6_NAV_PACKED") && atoi(getenv("PHMS_V6_NAV_PACKED"));
        const unsigned nav_fp32_lanes =
            getenv("PHMS_V6_NAV_FP32_LANES") ? std::stoul(getenv("PHMS_V6_NAV_FP32_LANES")) : 1;
        need(nav_fp32_lanes == 1 || nav_fp32_lanes == 4, "navigation FP32 lanes must be 1 or 4");
        need(nav_fp32_lanes == 1 || (!nav_cooperative && !nav_packed),
             "generic FP32 mode must not use integer-specific paths");
        const unsigned nav_devices =
            getenv("PHMS_V6_NAV_DEVICES") ? std::stoul(getenv("PHMS_V6_NAV_DEVICES")) : 1;
        const unsigned entry_mode =
            getenv("PHMS_NAV_ENTRY_MODE") ? std::stoul(getenv("PHMS_NAV_ENTRY_MODE")) : 0;
        const unsigned entry_beam =
            getenv("PHMS_NAV_ENTRY_BEAM") ? std::stoul(getenv("PHMS_NAV_ENTRY_BEAM")) : 128;
        const unsigned entry_budget =
            getenv("PHMS_NAV_ENTRY_BUDGET") ? std::stoul(getenv("PHMS_NAV_ENTRY_BUDGET")) : 128;
        need(entry_mode <= 2 && entry_beam >= 16 && entry_beam <= 128 && entry_budget > 0,
             "invalid entry refinement settings");
        need(nav_devices == 1 || nav_devices == PHMS_GC_GROUPS,
             "navigation devices must be 1 or 2");
        bool nav_integer_points = d.d <= 128;
        need(nav_threads == 32 || nav_threads == 128, "navigation threads must be 32 or 128");
        if (navigation)
        {
            need(d.d > 0 && nav_beam >= 16 && nav_beam <= 128 && nav_budget > 0,
                 "unsupported navigation beam/dimension/budget");
            need(nav_parallel || nav_fp32_lanes == 1,
                 "FP32 cooperative mode requires parallel navigation");
            nav.reset(new NavigationGraph(nav_path, fine_count));
            const char* part_path = getenv("PHMS_V6_REGION_PARTS");
            need(part_path && *part_path, "set PHMS_V6_REGION_PARTS");
            Map parts(part_path);
            need(parts.bytes == d.n, "bad navigation partition");
            for (unsigned j = 0; j < fine_count; ++j)
            {
                need(parts.data[e[2 * j + 1]] == e[2 * j],
                     "navigation owner differs from partition");
                need(!memcmp(c + uint64_t(j) * d.d,
                             d.graph.data + uint64_t(e[2 * j + 1]) * d.stride, d.d * 4),
                     "navigation point is not its actual base vector");
                for (unsigned k = 0; k < d.d; ++k)
                {
                    float x = c[uint64_t(j) * d.d + k];
                    need(std::isfinite(x), "nonfinite navigation point");
                    nav_integer_points =
                        nav_integer_points && (x >= 0 && x <= 255 && x == std::trunc(x));
                }
            }
        }
        const bool fine_entry = getenv("PHMS_V4_FINE_ENTRY") && atoi(getenv("PHMS_V4_FINE_ENTRY"));
        const char* region_path = getenv("PHMS_V6_REGION_ENTRY_MODEL");
        const bool region_enabled = region_path && *region_path;
        const bool entry_ab = getenv("PHMS_V6_ENTRY_AB") && atoi(getenv("PHMS_V6_ENTRY_AB"));
        const bool entry_sgemm =
            getenv("PHMS_V6_ENTRY_SGEMM") && atoi(getenv("PHMS_V6_ENTRY_SGEMM"));
        need(!navigation || (!region_enabled && !entry_ab && !fine_entry),
             "graph navigation cannot be combined with KMeans entry experiments");
        need(!entry_ab || region_enabled, "entry A/B requires a region entry model");
        need(!region_enabled || !fine_entry,
             "region entry and old fine-entry experiment are mutually exclusive");
        bool use_region_entry = region_enabled;
        std::unique_ptr<Map> region_centers;
        std::unique_ptr<Map> region_entries;
        std::unique_ptr<Map> region_parts;
        unsigned region_count = 0;
        if (region_enabled)
        {
            region_centers.reset(new Map(std::string(region_path) + "/centroids.fbin"));
            region_entries.reset(new Map(std::string(region_path) + "/entries.ubin"));
            const char* part_path = getenv("PHMS_V6_REGION_PARTS");
            need(part_path && *part_path,
                 "set PHMS_V6_REGION_PARTS for entry ownership validation");
            region_parts.reset(new Map(part_path));
            need(region_parts->bytes == d.n, "region partition size mismatch");
            region_count = region_centers->at<unsigned>(0);
            need(region_count == 4096 && region_centers->at<unsigned>(4) == d.d &&
                     region_centers->bytes == 8 + uint64_t(region_count) * d.d * 4,
                 "bad region entry centers");
            need(region_entries->at<unsigned>(0) == region_count &&
                     region_entries->at<unsigned>(4) == 2 &&
                     region_entries->bytes == 8 + uint64_t(region_count) * 8,
                 "bad region entry metadata");
            const auto* re = (const unsigned*)(region_entries->data + 8);
            const auto* rc = (const float*)(region_centers->data + 8);
            for (unsigned j = 0; j < region_count; ++j)
            {
                need(re[j * 2] < 4 && re[j * 2 + 1] < d.n &&
                         region_parts->data[re[j * 2 + 1]] == re[j * 2],
                     "entry model/partition mismatch");
                const float* x = (const float*)(d.graph.data + uint64_t(re[j * 2 + 1]) * d.stride);
                for (unsigned k = 0; k < d.d; ++k)
                    need(std::isfinite(rc[uint64_t(j) * d.d + k]) &&
                             rc[uint64_t(j) * d.d + k] == x[k],
                         "representative is not its actual node vector");
            }
        }
        const bool resident =
            getenv("PHMS_V4_RESIDENT_QUERIES") && atoi(getenv("PHMS_V4_RESIDENT_QUERIES"));
        need(!entry_mode || (navigation && nav_parallel && resident &&
                             nav_devices == PHMS_GC_GROUPS && !independent),
             "entry refinement requires distributed parallel navigation and resident queries");
        need(nav_devices == 1 || (navigation && resident && nav_parallel && nq >= PHMS_GC_GROUPS),
             "four-device navigation requires graph mode, resident queries and parallel kernel");
        const bool parallel_gather =
            getenv("PHMS_V4_PARALLEL_GATHER") && atoi(getenv("PHMS_V4_PARALLEL_GATHER"));
        const bool adaptive = getenv("PHMS_V5_ADAPTIVE") && atoi(getenv("PHMS_V5_ADAPTIVE"));
        const unsigned route_probe =
            getenv("PHMS_ROUTE_PROBE_CARDS") ? std::stoul(getenv("PHMS_ROUTE_PROBE_CARDS")) : 0;
        need(route_probe <= PHMS_GC_GROUPS && (!route_probe || (navigation && nav_parallel)),
             "invalid route probe");
        const bool replica_route =
            getenv("PHMS_REPLICA_AWARE_ROUTE") && atoi(getenv("PHMS_REPLICA_AWARE_ROUTE"));
        const float replica_target =
            getenv("PHMS_REPLICA_COVERAGE") ? std::stof(getenv("PHMS_REPLICA_COVERAGE")) : 0.75f;
        need(std::isfinite(replica_target) && replica_target > 0 && replica_target <= 1,
             "invalid replica coverage");
        need(!replica_route || (navigation && nav_parallel && !route_probe),
             "replica routing requires parallel graph navigation, no fixed-card probe");
        need(!entry_mode || (!route_probe && !replica_route && !entry_ab),
             "entry refinement requires unchanged primary routing policy");
        unsigned coalesce_min = 0;
        unsigned coalesced_queries = 0;
        unsigned coalesce_mode = 0;
        if (getenv("PHMS_V5_COALESCE_MIN_QUERIES"))
            coalesce_min = std::stoul(getenv("PHMS_V5_COALESCE_MIN_QUERIES"));
        if (getenv("PHMS_GC_COALESCE_MODE"))
            coalesce_mode = std::stoul(getenv("PHMS_GC_COALESCE_MODE"));
        need(coalesce_mode <= 4, "invalid coalesce mode");
        PhmsAdaptiveCards policy;
        if (getenv("PHMS_V5_MIN_CARDS"))
            policy.minimum = std::stoul(getenv("PHMS_V5_MIN_CARDS"));
        if (getenv("PHMS_V5_MAX_CARDS"))
            policy.maximum = std::stoul(getenv("PHMS_V5_MAX_CARDS"));
        if (getenv("PHMS_V5_VOTE_COVERAGE"))
            policy.coverage = std::stof(getenv("PHMS_V5_VOTE_COVERAGE"));
        if (getenv("PHMS_V5_DISTANCE_RATIO"))
            policy.distance_ratio = std::stof(getenv("PHMS_V5_DISTANCE_RATIO"));
        need(policy.minimum >= 1 && policy.minimum <= policy.maximum &&
                 policy.maximum <= PHMS_GC_GROUPS && std::isfinite(policy.coverage) &&
                 policy.coverage > 0 && policy.coverage <= 1 &&
                 std::isfinite(policy.distance_ratio) &&
                 (policy.distance_ratio == 0 || policy.distance_ratio >= 1),
             "invalid adaptive policy");
        need(!coalesce_min || adaptive, "mask coalescing requires adaptive routing");
        ck(cudaSetDevice(0));
        Buffer<float> dq;
        Buffer<float> dc;
        Buffer<float> doc;
        Buffer<float> dot;
        Buffer<float> norm;
        Buffer<float> qnorm;
        Buffer<unsigned> downers;
        Buffer<unsigned> doe;
        Buffer<unsigned> dfe;
        Buffer<unsigned> dm;
        Buffer<unsigned> ds;
        Buffer<float> hostq;
        Buffer<unsigned> hm;
        Buffer<unsigned> hs;
        Buffer<float> dregion;
        Buffer<float> dregion_norm;
        Buffer<float> dregion_dot;
        Buffer<unsigned> dregion_entries;
        Buffer<unsigned> dentry_masks;
        Buffer<unsigned> dnav_adj;
        Buffer<unsigned> dnav_starts;
        Buffer<unsigned> dnav_stats;
        Buffer<unsigned> dnav_membership;
        std::vector<unsigned> nav_membership;
        if (replica_route)
        {
            nav_membership.assign(fine_count, 0);
            const size_t bytes = 40 + ((uint64_t(d.n) + 63) / 64) * 8;
            for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
            {
                std::string path = std::string(getenv("PHMS_GRAPHCUT_LAYOUT")) + "/pq" +
                                   std::to_string(g) + ".bin";
                int fd = open(path.c_str(), O_RDONLY);
                need(fd >= 0, "open PQ membership failed");
                const uint8_t* map =
                    static_cast<const uint8_t*>(mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0));
                if (map == MAP_FAILED)
                {
                    close(fd);
                    throw std::runtime_error("map PQ membership failed");
                }
                const auto* bits = reinterpret_cast<const uint64_t*>(map + 40);
                for (unsigned p = 0; p < fine_count; ++p)
                {
                    unsigned id = e[2 * p + 1];
                    if (bits[id / 64] & (1ull << (id % 64)))
                        nav_membership[p] |= 1u << g;
                }
                munmap(const_cast<uint8_t*>(map), bytes);
                close(fd);
            }
            for (unsigned p = 0; p < fine_count; ++p)
                need(nav_membership[p] & (1u << e[2 * p]), "primary navigation PQ missing");
            dnav_membership.alloc(fine_count);
            dnav_membership.copy(nav_membership.data());
        }
        Buffer<unsigned> dnav_packed;
        Buffer<unsigned> dnav_norm;
        std::vector<unsigned> nav_host_packed;
        std::vector<unsigned> nav_host_norm;
        if (navigation)
        {
            dnav_adj.alloc(uint64_t(fine_count) * nav->degree);
            dnav_adj.copy(nav->edges(0));
            dnav_starts.alloc(4);
            dnav_starts.copy((const unsigned*)(nav->starts.data + 8));
            dnav_stats.alloc(nq * 2ull);
        }
        if (navigation && nav_packed && nav_integer_points)
        {
            unsigned words = (d.d + 3) / 4;
            nav_host_packed.resize(uint64_t(fine_count) * words);
            nav_host_norm.resize(fine_count);
            for (unsigned j = 0; j < fine_count; ++j)
                for (unsigned k = 0; k < d.d; ++k)
                {
                    unsigned value = static_cast<unsigned>(c[uint64_t(j) * d.d + k]);
                    nav_host_packed[uint64_t(j) * words + k / 4] |= value << ((k % 4) * 8);
                    nav_host_norm[j] += value * value;
                }
            dnav_packed.alloc(nav_host_packed.size());
            dnav_packed.copy(nav_host_packed.data());
            dnav_norm.alloc(nav_host_norm.size());
            dnav_norm.copy(nav_host_norm.data());
        }
        constexpr unsigned entry_tile = 512;
        if (region_enabled)
        {
            dregion.alloc(uint64_t(region_count) * d.d);
            dregion.copy((const float*)(region_centers->data + 8));
            dregion_entries.alloc(uint64_t(region_count) * 2);
            dregion_entries.copy((const unsigned*)(region_entries->data + 8));
            dentry_masks.alloc(nq);
            if (entry_sgemm)
            {
                dregion_norm.alloc(region_count);
                dregion_dot.alloc(uint64_t(region_count) * std::min(entry_tile, nq));
                gc_norm<<<(region_count + 127) / 128, 128>>>(dregion.p, dregion_norm.p,
                                                             region_count, d.d);
                ck(cudaGetLastError());
            }
        }
        hostq.alloc(uint64_t(nq) * d.d, true);
        memcpy(hostq.p, d.query, hostq.n * 4);
        hm.alloc(nq, true);
        hs.alloc(nq, true);
        dq.alloc(hostq.n);
        dc.alloc(combined.size());
        dc.copy(combined.data());
        dm.alloc(nq);
        ds.alloc(nq);
        dfe.alloc(uint64_t(fine_count) * 2);
        dfe.copy(e);
        if (!navigation)
        {
            doc.alloc(256ull * d.d);
            doc.copy(oc);
            downers.alloc(1024);
            downers.copy(owners.data());
            doe.alloc(512);
            doe.copy(oe);
        }
        std::array<Buffer<float>, 8> resident_queries;
        auto preload_start = std::chrono::steady_clock::now();
        if (resident)
        {
            dq.copy(hostq.p);
            for (int g = 1; g < PHMS_GC_GROUPS; ++g)
            {
                ck(cudaSetDevice(g));
                resident_queries[g].alloc(hostq.n);
                resident_queries[g].copy(hostq.p);
            }
            ck(cudaSetDevice(0));
        }
        const double preload_ms = elapsed(preload_start);
        std::array<std::unique_ptr<NavigationReplica>, 8> nav_replicas;
        uint64_t nav_extra_model_bytes = 0;
        if (navigation && nav_devices == PHMS_GC_GROUPS)
        {
            for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
            {
                ck(cudaSetDevice(g));
                nav_replicas[g].reset(new NavigationReplica);
                auto& r = *nav_replicas[g];
                r.gpu = g;
                ck(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
                if (entry_mode == 2)
                {
                    unsigned count =
                        uint64_t(g + 1) * nq / PHMS_GC_GROUPS - uint64_t(g) * nq / PHMS_GC_GROUPS;
                    r.entry_masks.alloc(count);
                    r.entry_indices.alloc(count);
                    r.entry_distances.alloc(count);
                    r.entry_statistics.alloc(uint64_t(count) * 2);
                }
                if (!g)
                    continue;
                r.points.alloc(uint64_t(fine_count) * d.d);
                r.points.copy(c);
                r.entries.alloc(uint64_t(fine_count) * 2);
                r.entries.copy(e);
                r.adjacency.alloc(uint64_t(fine_count) * nav->degree);
                r.adjacency.copy(nav->edges(0));
                r.starts.alloc(4);
                r.starts.copy((const unsigned*)(nav->starts.data + 8));
                if (dnav_packed.p)
                {
                    r.packed.alloc(nav_host_packed.size());
                    r.packed.copy(nav_host_packed.data());
                    r.norm.alloc(nav_host_norm.size());
                    r.norm.copy(nav_host_norm.data());
                }
                if (replica_route)
                {
                    r.membership.alloc(fine_count);
                    r.membership.copy(nav_membership.data());
                }
                unsigned chunk =
                    uint64_t(g + 1) * nq / PHMS_GC_GROUPS - uint64_t(g) * nq / PHMS_GC_GROUPS;
                r.masks.alloc(chunk);
                r.seeds.alloc(chunk);
                r.statistics.alloc(uint64_t(chunk) * 2);
                nav_extra_model_bytes += (r.points.n + r.entries.n + r.adjacency.n + r.starts.n +
                                          r.packed.n + r.norm.n + r.membership.n) *
                                         4;
            }
            ck(cudaSetDevice(0));
        }
        if (!navigation)
        {
            dot.alloc(1280ull * nq);
            norm.alloc(1280);
            qnorm.alloc(nq);
            gc_norm<<<10, 128>>>(dc.p, norm.p, 1280, d.d);
            ck(cudaGetLastError());
        }
        cublasHandle_t blas;
        cb(cublasCreate(&blas));
        cb(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        std::vector<unsigned> masks(nq);
        std::vector<unsigned> seeds(nq);
        std::vector<unsigned> entry_base_seeds(nq);
        Buffer<unsigned> entry_host_masks;
        if (entry_mode == 2)
            entry_host_masks.alloc(nq, true);
        double entry_refine_ms = 0;
        double route_h2d_ms = 0;
        auto route = [&]()
        {
            ck(cudaSetDevice(0));
            route_h2d_ms = 0;
            entry_refine_ms = 0;
            if (!resident)
            {
                auto upload_start = std::chrono::steady_clock::now();
                dq.copy(hostq.p);
                route_h2d_ms = elapsed(upload_start);
            }
            if (navigation && nav_devices == PHMS_GC_GROUPS)
            {
                for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                {
                    ck(cudaSetDevice(g));
                    auto& r = *nav_replicas[g];
                    unsigned begin = uint64_t(g) * nq / PHMS_GC_GROUPS,
                             chunk = uint64_t(g + 1) * nq / PHMS_GC_GROUPS - begin;
                    const float* query = (g ? resident_queries[g].p : dq.p) + uint64_t(begin) * d.d;
                    unsigned* device_masks = g ? r.masks.p : dm.p;
                    unsigned* device_seeds = g ? r.seeds.p : ds.p;
                    gc_navigation_graph_parallel<<<
                        chunk, nav_threads, gc_nav_shared_bytes(fine_count, d.d), r.stream>>>(
                        query, g ? r.points.p : dc.p, g ? r.entries.p : dfe.p,
                        g ? r.adjacency.p : dnav_adj.p, g ? r.starts.p : dnav_starts.p,
                        device_masks, device_seeds, g ? r.statistics.p : dnav_stats.p, fine_count,
                        nav->degree, d.d, nav_beam, nav_budget, cards, adaptive, policy, nav_stop,
                        nav_cooperative, nav_integer_points, g ? r.packed.p : dnav_packed.p,
                        g ? r.norm.p : dnav_norm.p, nav_fp32_lanes, route_probe,
                        g ? r.membership.p : dnav_membership.p, replica_target,
                        entry_mode == 1 ? 1 : 0, nullptr,
                        entry_mode == 2 ? r.entry_distances.p : nullptr,
                        entry_mode == 2 ? r.entry_indices.p : nullptr);
                    ck(cudaGetLastError());
                    ck(cudaMemcpyAsync(hm.p + begin, device_masks, chunk * 4,
                                       cudaMemcpyDeviceToHost, r.stream));
                    ck(cudaMemcpyAsync(hs.p + begin, device_seeds, chunk * 4,
                                       cudaMemcpyDeviceToHost, r.stream));
                }
                for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                {
                    ck(cudaSetDevice(g));
                    ck(cudaStreamSynchronize(nav_replicas[g]->stream));
                }
                ck(cudaSetDevice(0));
            }
            else if (navigation)
            {
                if (nav_parallel)
                    gc_navigation_graph_parallel<<<nq, nav_threads,
                                                   gc_nav_shared_bytes(fine_count, d.d)>>>(
                        dq.p, dc.p, dfe.p, dnav_adj.p, dnav_starts.p, dm.p, ds.p, dnav_stats.p,
                        fine_count, nav->degree, d.d, nav_beam, nav_budget, cards, adaptive, policy,
                        nav_stop, nav_cooperative, nav_integer_points, dnav_packed.p, dnav_norm.p,
                        nav_fp32_lanes, route_probe, dnav_membership.p, replica_target);
                else
                    gc_navigation_graph<<<nq, 128, gc_nav_shared_bytes(fine_count, d.d)>>>(
                        dq.p, dc.p, dfe.p, dnav_adj.p, dnav_starts.p, dm.p, ds.p, dnav_stats.p,
                        fine_count, nav->degree, d.d, nav_beam, nav_budget, cards, adaptive, policy,
                        nav_stop);
                ck(cudaGetLastError());
            }
            else
            {
                gc_norm<<<(nq + 127) / 128, 128>>>(dq.p, qnorm.p, nq, d.d);
                float alpha = -2;
                float beta = 0;
                cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, 1280, nq, d.d, &alpha, dc.p, d.d,
                               dq.p, d.d, &beta, dot.p, 1280));
                gc_route<<<nq, 128>>>(dq.p, dc.p, downers.p, doc.p, doe.p, dm.p, ds.p, d.d, 1024,
                                      dot.p, norm.p, qnorm.p, nullptr, cards,
                                      fine_entry ? dfe.p : nullptr, adaptive, policy,
                                      region_enabled ? dentry_masks.p : nullptr);
                ck(cudaGetLastError());
                if (use_region_entry)
                {
                    if (entry_sgemm)
                    {
                        for (unsigned begin = 0; begin < nq; begin += entry_tile)
                        {
                            unsigned count = std::min(entry_tile, nq - begin);
                            cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, region_count, count, d.d,
                                           &alpha, dregion.p, d.d, dq.p + uint64_t(begin) * d.d,
                                           d.d, &beta, dregion_dot.p, region_count));
                            gc_region_entry<<<count, 128>>>(
                                dq.p + uint64_t(begin) * d.d, dregion.p, dregion_entries.p,
                                dentry_masks.p + begin, ds.p + begin, d.d, region_count,
                                dregion_dot.p, dregion_norm.p, qnorm.p + begin);
                            ck(cudaGetLastError());
                        }
                    }
                    else
                    {
                        gc_region_entry<<<nq, 128>>>(dq.p, dregion.p, dregion_entries.p,
                                                     dentry_masks.p, ds.p, d.d, region_count);
                        ck(cudaGetLastError());
                    }
                }
            }
            if (nav_devices == 1)
            {
                ck(cudaMemcpy(hm.p, dm.p, nq * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(hs.p, ds.p, nq * 4, cudaMemcpyDeviceToHost));
            }
            masks.assign(hm.p, hm.p + nq);
            seeds.assign(hs.p, hs.p + nq);
            for (unsigned q = 0; q < nq; ++q)
            {
                unsigned count = __builtin_popcount(masks[q]);
                need(masks[q] > 0 && masks[q] < (1u << PHMS_GC_GROUPS) && seeds[q] < d.n &&
                         (adaptive ? (count >= policy.minimum && count <= policy.maximum)
                                   : count == cards),
                     "invalid GPU route");
            }
            coalesced_queries = phms_gc_coalesce_masks(masks, coalesce_min, coalesce_mode,
                                                       adaptive ? policy.maximum : PHMS_GC_GROUPS);
            if (adaptive)
                for (unsigned m : masks)
                    need(unsigned(__builtin_popcount(m)) <= policy.maximum,
                         "coalescing exceeded routing card limit");
            if (entry_mode == 2)
            {
                auto refine_start = std::chrono::steady_clock::now();
                entry_base_seeds = seeds;
                memcpy(entry_host_masks.p, masks.data(), nq * 4ull);
                for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                {
                    ck(cudaSetDevice(g));
                    auto& r = *nav_replicas[g];
                    unsigned begin = uint64_t(g) * nq / PHMS_GC_GROUPS,
                             chunk = uint64_t(g + 1) * nq / PHMS_GC_GROUPS - begin;
                    ck(cudaMemcpyAsync(r.entry_masks.p, entry_host_masks.p + begin, chunk * 4,
                                       cudaMemcpyHostToDevice, r.stream));
                    const float* query = (g ? resident_queries[g].p : dq.p) + uint64_t(begin) * d.d;
                    gc_navigation_graph_parallel<<<
                        chunk, nav_threads, gc_nav_shared_bytes(fine_count, d.d), r.stream>>>(
                        query, g ? r.points.p : dc.p, g ? r.entries.p : dfe.p,
                        g ? r.adjacency.p : dnav_adj.p, g ? r.starts.p : dnav_starts.p,
                        r.entry_masks.p, g ? r.seeds.p : ds.p, r.entry_statistics.p, fine_count,
                        nav->degree, d.d, entry_beam, entry_budget, cards, adaptive, policy,
                        PhmsNavStop(), nav_cooperative, nav_integer_points,
                        g ? r.packed.p : dnav_packed.p, g ? r.norm.p : dnav_norm.p, nav_fp32_lanes,
                        0, nullptr, replica_target, 1, r.entry_masks.p, r.entry_distances.p,
                        r.entry_indices.p);
                    ck(cudaGetLastError());
                    ck(cudaMemcpyAsync(hs.p + begin, g ? r.seeds.p : ds.p, chunk * 4,
                                       cudaMemcpyDeviceToHost, r.stream));
                }
                for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
                {
                    ck(cudaSetDevice(g));
                    ck(cudaStreamSynchronize(nav_replicas[g]->stream));
                }
                ck(cudaSetDevice(0));
                seeds.assign(hs.p, hs.p + nq);
                entry_refine_ms = elapsed(refine_start);
            }
        };
        unsigned max_query_batch = 0;
        if (getenv("PHMS_V5_MAX_QUERY_BATCH"))
            max_query_batch = std::stoul(getenv("PHMS_V5_MAX_QUERY_BATCH"));
        const bool balanced_batches =
            getenv("PHMS_V5_BALANCED_BATCHES") && atoi(getenv("PHMS_V5_BALANCED_BATCHES"));
        unsigned cost_batches =
            getenv("PHMS_GC_COST_BATCHES") ? std::stoul(getenv("PHMS_GC_COST_BATCHES")) : 0;
        need(cost_batches <= 2 && (!cost_batches || balanced_batches),
             "cost batching requires balanced batches");
        const bool seed_locality =
            getenv("PHMS_V5_SORT_BY_SEED") && atoi(getenv("PHMS_V5_SORT_BY_SEED"));
        need(!entry_ab || !seed_locality,
             "entry A/B requires frozen batch order (seed sorting disabled)");
        const bool seed_counting =
            !getenv("PHMS_V5_SEED_COUNTING") || atoi(getenv("PHMS_V5_SEED_COUNTING"));
        const bool park_workers =
            getenv("PHMS_V5_PARK_WORKERS") && atoi(getenv("PHMS_V5_PARK_WORKERS"));
        const unsigned parallel_submit_min = getenv("PHMS_PARALLEL_GPU_MIN_CARDS")
                                                 ? std::stoul(getenv("PHMS_PARALLEL_GPU_MIN_CARDS"))
                                                 : 3;
        std::vector<unsigned> seed_ids;
        for (unsigned j = 0; j < old_count; ++j)
            seed_ids.push_back(oe[2 * j + 1]);
        if (navigation)
            for (unsigned j = 0; j < fine_count; ++j)
                seed_ids.push_back(e[2 * j + 1]);
        if (fine_entry)
            for (unsigned j = 0; j < 1024; ++j)
                seed_ids.push_back(e[2 * j + 1]);
        if (region_enabled)
        {
            const auto* re = (const unsigned*)(region_entries->data + 8);
            for (unsigned j = 0; j < region_count; ++j)
                seed_ids.push_back(re[j * 2 + 1]);
        }
        std::sort(seed_ids.begin(), seed_ids.end());
        seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()), seed_ids.end());
        std::unordered_map<unsigned, unsigned> seed_categories;
        for (unsigned i = 0; i < seed_ids.size(); ++i)
            seed_categories.emplace(seed_ids[i], i);
        std::unique_ptr<Map> query_regions;
        if (getenv("PHMS_QUERY_REGIONS") && *getenv("PHMS_QUERY_REGIONS"))
        {
            need(seed_locality && seed_counting, "region grouping requires counting seed locality");
            query_regions.reset(new Map(getenv("PHMS_QUERY_REGIONS")));
            need(query_regions->bytes == uint64_t(d.n) * sizeof(uint16_t),
                 "invalid query-region map size");
            for (auto& entry : seed_categories)
            {
                entry.second = reinterpret_cast<const uint16_t*>(query_regions->data)[entry.first];
                need(entry.second < seed_categories.size(),
                     "region category exceeds counting capacity");
            }
        }
        need(!balanced_batches || !max_query_batch,
             "balanced batches and explicit cap are separate experiments");
        route();
        const auto frozen_masks = masks;
        const auto frozen_seeds = seeds;
        uint64_t mask_hash = 14695981039346656037ull;
        unsigned entry_changed = 0;
        for (unsigned q = 0; q < nq; ++q)
        {
            entry_changed += entry_mode == 2 && seeds[q] != entry_base_seeds[q];
            for (unsigned b = 0; b < 4; ++b)
            {
                mask_hash ^= (masks[q] >> (8 * b)) & 255;
                mask_hash *= 1099511628211ull;
            }
        }
        std::cout << "[ENTRY REFINE] mode=" << entry_mode << " mask_fnv64=" << mask_hash
                  << " changed_vs_stage1=" << entry_changed << " beam=" << entry_beam
                  << " budget=" << entry_budget << '\n';
        uint64_t entry_hash = 14695981039346656037ull;
        for (unsigned id : seeds)
            for (unsigned byte = 0; byte < 4; ++byte)
            {
                entry_hash ^= (id >> (8 * byte)) & 255;
                entry_hash *= 1099511628211ull;
            }
        std::cout << "[V6 ENTRY HASH] fnv64=" << entry_hash << " route_probe_cards=" << route_probe
                  << '\n';
        std::vector<unsigned> base_seeds;
        if (entry_ab)
        {
            use_region_entry = false;
            route();
            need(masks == frozen_masks, "entry experiment changed card masks");
            base_seeds = seeds;
            use_region_entry = true;
            route();
            need(seeds == frozen_seeds, "region entry is nondeterministic");
        }
        auto organize = [&]()
        {
            std::vector<unsigned> query_order(nq);
            std::vector<unsigned> ordered_masks(nq);
            std::iota(query_order.begin(), query_order.end(), 0u);
            if (seed_locality)
            {
                if (seed_counting)
                    query_order = phms_gc_seed_order(seeds, seed_categories);
                else
                    std::stable_sort(query_order.begin(), query_order.end(),
                                     [&](unsigned a, unsigned b) { return seeds[a] < seeds[b]; });
            }
            for (unsigned i = 0; i < nq; ++i)
                ordered_masks[i] = masks[query_order[i]];
            auto out = cost_batches ? phms_gc_cost_batches(ordered_masks, workers, cost_batches)
                       : balanced_batches
                           ? phms_gc_balanced_batches(ordered_masks, workers)
                           : phms_gc_batches(ordered_masks, workers, max_query_batch);
            for (auto& batch : out)
                for (unsigned& id : batch.ids)
                    id = query_order[id];
            return out;
        };
        auto batches = organize();
        std::vector<unsigned> query_seen(nq);
        for (auto& batch : batches)
            for (unsigned q : batch.ids)
            {
                need(q < nq && batch.mask == masks[q], "bad ordered batch");
                ++query_seen[q];
            }
        for (unsigned n : query_seen)
            need(n == 1, "query lost or duplicated during organization");
        std::array<unsigned, 9> card_counts{};
        uint64_t card_tasks = 0;
        for (unsigned m : masks)
        {
            unsigned count = __builtin_popcount(m);
            ++card_counts[count];
            card_tasks += count;
        }
        const double average_cards = double(card_tasks) / nq;
        std::vector<std::unique_ptr<Job>> jobs;
        const std::string nt = std::to_string(threads);
        setenv("PHMS_CPU_THREADS", nt.c_str(), 1);
        setenv("OMP_DYNAMIC", "FALSE", 1);
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        for (auto& batch : batches)
        {
            std::unique_ptr<Job> j(new Job);
            j->batch = std::move(batch);
            j->budget = budgets[__builtin_popcount(j->batch.mask) - 1];
            j->queries.alloc(uint64_t(j->batch.ids.size()) * d.d, true);
            j->seeds.alloc(j->batch.ids.size(), true);
            j->ids.resize(j->batch.ids.size() * 10ull);
            j->dist.resize(j->ids.size());
            for (int g = 0; g < PHMS_GC_GROUPS; ++g)
                if (j->batch.mask & (1u << g))
                {
                    ck(cudaSetDevice(g));
                    std::unique_ptr<PHMSSearch<float>> o(new PHMSSearch<float>);
                    need(o->phms_load_sharded(argv[1], 0, d.n), "GraphCut index load failed");
                    o->phms_set_searchparams(10, L, ENUM_DIST_L2);
                    o->phms_set_expansion_budget(j->budget);
                    o->phms_alloc(j->batch.ids.size());
                    o->phms_prepare_ngpu_graph(j->batch.ids.size());
                    j->gpus.push_back(g);
                    j->objects.push_back(std::move(o));
                }
            if (independent)
                j->shard_tracks.resize(j->gpus.size());
            jobs.push_back(std::move(j));
        }
        auto gather_job = [&](Job& j)
        {
            for (unsigned i = 0; i < j.batch.ids.size(); ++i)
            {
                unsigned q = j.batch.ids[i];
                if (!resident)
                    memcpy(j.queries.p + uint64_t(i) * d.d, hostq.p + uint64_t(q) * d.d, d.d * 4);
                j.seeds.p[i] = seeds[q];
            }
        };
        auto gather = [&]()
        {
            for (auto& j : jobs)
                gather_job(*j);
        };
        gather();
        std::vector<result_ann_t> output(nq * 10ull);
        std::vector<float> distances(nq * 10ull);
        std::vector<std::vector<unsigned>> assignments(workers);
        std::vector<uint64_t> load(workers);
        std::vector<std::pair<unsigned, unsigned>> tasks;
        for (unsigned i = 0; i < jobs.size(); ++i)
            for (unsigned s = 0; s < (independent ? jobs[i]->gpus.size() : 1); ++s)
            {
                unsigned t = tasks.size();
                tasks.emplace_back(i, s);
                unsigned w = std::min_element(load.begin(), load.end()) - load.begin();
                assignments[w].push_back(t);
                load[w] += jobs[i]->batch.ids.size() * (independent ? 1 : jobs[i]->gpus.size()) *
                           jobs[i]->budget;
            }
        const char* diagnostic_dir = getenv("PHMS_QG_DIAGNOSTICS");
        const bool diagnostic = diagnostic_dir && *diagnostic_dir;
        unsigned diagnostic_run = 0;
        bool collect_tracks = true;
        auto merge_independent = [&](Job& j)
        {
            j.tracks.assign(j.batch.ids.size(), {});
#pragma omp parallel for num_threads(threads) schedule(static)
            for (unsigned row = 0; row < j.batch.ids.size(); ++row)
            {
                auto& track = j.tracks[row];
                std::unordered_set<unsigned> seen;
                for (auto& shard : j.shard_tracks)
                    for (unsigned id : shard[row])
                        if (seen.insert(id).second)
                            track.push_back(id);
                const float* q = d.query + uint64_t(j.batch.ids[row]) * d.d;
                std::vector<Candidate> candidates;
                candidates.reserve(track.size());
                for (unsigned id : track)
                    candidates.push_back(
                        {id, phms_simd_l2(reinterpret_cast<const float*>(d.graph.data +
                                                                         uint64_t(id) * d.stride),
                                          q, d.d)});
                unsigned k = std::min<unsigned>(10, candidates.size());
                std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                                  less_candidate);
                for (unsigned i = 0; i < k; ++i)
                {
                    j.ids[row * 10ull + i] = candidates[i].id;
                    j.dist[row * 10ull + i] = candidates[i].distance;
                }
                for (unsigned i = k; i < 10; ++i)
                {
                    j.ids[row * 10ull + i] = UINT32_MAX - i;
                    j.dist[row * 10ull + i] = INFINITY;
                }
            }
        };
        auto execute = [&](unsigned w)
        {
            for (unsigned task : assignments[w])
            {
                auto work = tasks[task];
                auto& j = *jobs[work.first];
                std::vector<PHMSSearch<float>*> objects;
                std::string label;
                if (phms_trace_enabled())
                    label = "JOB=" + std::to_string(work.first) +
                            " mask=" + std::to_string(j.batch.mask) +
                            " nq=" + std::to_string(j.batch.ids.size()) +
                            " worker=" + std::to_string(w);
                PhmsTraceRange job_trace(label.c_str());
                if (independent)
                {
                    unsigned s = work.second;
                    auto& o = j.objects[s];
                    int g = j.gpus[s];
                    if (resident)
                        o->phms_select_resident_queries(g ? resident_queries[g].p : dq.p, nq,
                                                        j.batch.ids.data(), j.batch.ids.size(),
                                                        hostq.p);
                    std::vector<PHMSSearch<float>*> single{o.get()};
                    PHMSSearch<float>::ngpu_phms_query(single, j.queries.p, j.batch.ids.size(),
                                                       nullptr, nullptr, {d.n}, j.seeds.p,
                                                       &j.shard_tracks[s]);
                    if (j.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        merge_independent(j);
                    continue;
                }
                for (unsigned i = 0; i < j.objects.size(); ++i)
                {
                    auto& o = j.objects[i];
                    int g = j.gpus[i];
                    if (resident)
                        o->phms_select_resident_queries(g ? resident_queries[g].p : dq.p, nq,
                                                        j.batch.ids.data(), j.batch.ids.size(),
                                                        hostq.p);
                    objects.push_back(o.get());
                }
                std::vector<unsigned> ends(objects.size(), d.n);
                PHMSSearch<float>::ngpu_phms_query(objects, j.queries.p, j.batch.ids.size(),
                                                   j.ids.data(), j.dist.data(), ends, j.seeds.p,
                                                   collect_tracks ? &j.tracks : nullptr);
                if (route_probe)
                    for (unsigned row = 0; row < j.batch.ids.size(); ++row)
                        for (unsigned k = std::min<size_t>(10, j.tracks[row].size()); k < 10; ++k)
                        {
                            j.ids[row * 10ull + k] = UINT32_MAX - k;
                            j.dist[row * 10ull + k] = INFINITY;
                        }
            }
        };
        auto request = [&](bool timed)
        {
            require_idle_or_self();
            for (auto& j : jobs)
                j->remaining.store(j->gpus.size());
            std::atomic<unsigned> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> team;
            std::mutex start_mutex;
            std::condition_variable start_condition;
            std::vector<double> worker_ms(workers);
            std::vector<double> worker_gather_ms(workers);
            auto run_worker = [&](unsigned w)
            {
                auto begin = std::chrono::steady_clock::now();
                if (timed && parallel_gather && !independent)
                {
                    for (unsigned task : assignments[w])
                        gather_job(*jobs[tasks[task].first]);
                    worker_gather_ms[w] = elapsed(begin);
                }
                execute(w);
                worker_ms[w] = elapsed(begin);
            };
            auto warm = [&](unsigned w)
            {
                omp_set_num_threads(threads);
                for (unsigned task : assignments[w])
                {
                    auto work = tasks[task];
                    auto& j = *jobs[work.first];
                    for (unsigned s = 0; s < j.gpus.size(); ++s)
                        if (!independent || s == work.second)
                        {
                            ck(cudaSetDevice(j.gpus[s]));
                            ck(cudaFree(nullptr));
                        }
                }
#pragma omp parallel num_threads(threads)
                {
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    if (getenv("PHMS_PARALLEL_GPU_SUBMIT") &&
                        atoi(getenv("PHMS_PARALLEL_GPU_SUBMIT")))
                    {
                        unsigned tid = omp_get_thread_num();
                        unsigned nth = omp_get_num_threads();
                        for (unsigned task : assignments[w])
                        {
                            auto& job = *jobs[tasks[task].first];
                            if (job.gpus.size() >= parallel_submit_min)
                                for (unsigned s = tid; s < job.gpus.size(); s += nth)
                                {
                                    ck(cudaSetDevice(job.gpus[s]));
                                    ck(cudaFree(nullptr));
                                }
                        }
                    }
                }
            };
            for (unsigned w = 1; w < workers; ++w)
                team.emplace_back(
                    [&, w]()
                    {
                        warm(w);
                        ++ready;
                        if (park_workers)
                        {
                            std::unique_lock<std::mutex> lock(start_mutex);
                            start_condition.wait(lock, [&]()
                                                 { return go.load(std::memory_order_acquire); });
                        }
                        else
                            while (!go.load(std::memory_order_acquire))
                            {
                            }
                        run_worker(w);
                    });
            warm(0);
            while (ready.load() != workers - 1)
            {
            }
            auto start = std::chrono::steady_clock::now();
            PhmsTraceRange request_trace("REQUEST_WALL");
            double route_ms = 0;
            double gather_ms = 0;
            double organization_ms = 0;
            if (timed)
            {
                auto t = std::chrono::steady_clock::now();
                route();
                need(masks == frozen_masks &&
                         seeds == (entry_ab && !use_region_entry ? base_seeds : frozen_seeds),
                     "route changed from prepared workspaces");
                route_ms = elapsed(t);
                t = std::chrono::steady_clock::now();
                auto layout = organize();
                need(layout.size() == jobs.size(), "job layout changed");
                for (unsigned i = 0; i < layout.size(); ++i)
                    need(layout[i].mask == jobs[i]->batch.mask &&
                             layout[i].ids == jobs[i]->batch.ids,
                         "query ordering changed");
                organization_ms = elapsed(t);
                if (!parallel_gather || independent)
                {
                    t = std::chrono::steady_clock::now();
                    gather();
                    gather_ms = elapsed(t);
                }
            }
            if (park_workers)
            {
                {
                    std::lock_guard<std::mutex> lock(start_mutex);
                    go.store(true, std::memory_order_release);
                }
                start_condition.notify_all();
            }
            else
                go.store(true, std::memory_order_release);
            run_worker(0);
            for (auto& t : team)
                t.join();
            auto scatter = std::chrono::steady_clock::now();
            for (auto& j : jobs)
                for (unsigned i = 0; i < j->batch.ids.size(); ++i)
                {
                    unsigned q = j->batch.ids[i];
                    memcpy(output.data() + uint64_t(q) * 10, j->ids.data() + uint64_t(i) * 10,
                           10 * sizeof(result_ann_t));
                    memcpy(distances.data() + uint64_t(q) * 10, j->dist.data() + uint64_t(i) * 10,
                           40);
                }
            double scatter_ms = elapsed(scatter);
            double wall = elapsed(start);
            request_trace.end();
            uint64_t hits = 0;
            std::array<uint64_t, 9> hits_by_cards{};
            for (unsigned q = 0; q < nq; ++q)
            {
                std::unordered_set<result_ann_t> unique;
                for (unsigned k = 0; k < 10; ++k)
                {
                    auto id = output[uint64_t(q) * 10 + k];
                    need(unique.insert(id).second, "duplicate output");
                    for (unsigned t = 0; t < 10; ++t)
                    {
                        bool hit = id == d.gt[uint64_t(q) * d.gtk + t];
                        hits += hit;
                        hits_by_cards[__builtin_popcount(masks[q])] += hit;
                    }
                }
            }
            uint64_t id_hash = 14695981039346656037ull;
            uint64_t dist_hash = id_hash;
            for (size_t i = 0; i < output.size(); ++i)
            {
                uint64_t id = output[i];
                unsigned bits = 0;
                memcpy(&bits, &distances[i], 4);
                for (unsigned b = 0; b < 8; ++b)
                {
                    id_hash ^= (id >> (8 * b)) & 255;
                    id_hash *= 1099511628211ull;
                }
                for (unsigned b = 0; b < 4; ++b)
                {
                    dist_hash ^= (bits >> (8 * b)) & 255;
                    dist_hash *= 1099511628211ull;
                }
            }
            if (diagnostic)
            {
                const std::string prefix =
                    std::string(diagnostic_dir) + "/run" + std::to_string(diagnostic_run++);
                auto dump = [&](const std::string& suffix, const void* data, size_t bytes)
                {
                    std::ofstream file(prefix + suffix, std::ios::binary);
                    need(bool(file), "diagnostic output open");
                    file.write(reinterpret_cast<const char*>(data), bytes);
                    need(bool(file), "diagnostic output write");
                };
                dump(".ids.u64", output.data(), output.size() * sizeof(result_ann_t));
                dump(".dist.f32", distances.data(), distances.size() * sizeof(float));
                dump(".masks.u32", masks.data(), masks.size() * sizeof(unsigned));
                dump(".raw_masks.u32", hm.p, uint64_t(nq) * sizeof(unsigned));
                dump(".seeds.u32", seeds.data(), seeds.size() * sizeof(unsigned));
                unsigned width = 0;
                for (auto& j : jobs)
                    width = std::max<unsigned>(width,
                                               j->budget * (independent ? j->gpus.size() : 1) + 1);
                std::vector<unsigned> tracks(uint64_t(nq) * width, UINT32_MAX);
                for (auto& j : jobs)
                    for (unsigned i = 0; i < j->batch.ids.size(); ++i)
                    {
                        unsigned q = j->batch.ids[i];
                        auto& track = j->tracks[i];
                        need(track.size() < width, "diagnostic track overflow");
                        tracks[uint64_t(q) * width] = track.size();
                        std::copy(track.begin(), track.end(),
                                  tracks.begin() + uint64_t(q) * width + 1);
                    }
                dump(".tracks.u32", tracks.data(), tracks.size() * sizeof(unsigned));
                std::ofstream meta(prefix + ".json");
                meta << "{\"queries\":" << nq << ",\"track_width\":" << width
                     << ",\"timed\":" << timed << ",\"L\":" << L << ",\"GPUs\":" << PHMS_GC_GROUPS
                     << ",\"Bloom\":" << PHMS_V3_BF_ENTRIES << "}";
            }
            std::cout << "[V5 OUTPUT] timed=" << timed << " ids_fnv64=" << id_hash
                      << " distances_fnv64=" << dist_hash << " region_entry=" << use_region_entry
                      << '\n';
            if (collect_tracks)
            {
                uint64_t expanded = 0;
                uint64_t short_queries = 0;
                for (auto& j : jobs)
                    for (auto& track : j->tracks)
                    {
                        expanded += track.size();
                        short_queries += track.size() < 10;
                    }
                std::cout << "[ROUTE WORK] timed=" << timed << " parent_candidates=" << expanded
                          << " mean=" << double(expanded) / nq << " short_queries=" << short_queries
                          << '\n';
            }
            if (timed)
                std::cout << std::setprecision(10) << L << ',' << nq * 1000. / wall << ','
                          << double(hits) / (nq * 10ull) << " wall_ms=" << wall
                          << " route_ms=" << route_ms << " gather_ms=" << gather_ms
                          << " scatter_ms=" << scatter_ms
                          << " min_cards=" << (adaptive ? policy.minimum : cards)
                          << " max_cards=" << (adaptive ? policy.maximum : cards)
                          << " route_h2d_ms=" << route_h2d_ms
                          << " compute_reference_ms=" << wall - route_h2d_ms
                          << " compute_reference_qps=" << nq * 1000. / (wall - route_h2d_ms)
                          << " fine_entry=" << fine_entry << " region_entry=" << use_region_entry
                          << " region_entry_points=" << (use_region_entry ? region_count : 0)
                          << " region_entry_sgemm=" << (use_region_entry && entry_sgemm)
                          << " navigation_graph=" << navigation
                          << " navigation_beam=" << (navigation ? nav_beam : 0)
                          << " navigation_budget=" << (navigation ? nav_budget : 0)
                          << " navigation_parallel=" << nav_parallel
                          << " navigation_threads=" << nav_threads
                          << " navigation_stop_min=" << nav_stop.minimum
                          << " navigation_stop_interval=" << nav_stop.interval
                          << " navigation_stop_stable=" << nav_stop.stable_checks
                          << " navigation_stop_ratio=" << nav_stop.frontier_ratio
                          << " navigation_cooperative=" << nav_cooperative
                          << " navigation_integer_points=" << nav_integer_points
                          << " navigation_packed=" << (dnav_packed.p != nullptr)
                          << " navigation_packed_bytes=" << (dnav_packed.n + dnav_norm.n) * 4
                          << " navigation_devices=" << nav_devices
                          << " extra_navigation_model_bytes=" << nav_extra_model_bytes
                          << " entry_mode=" << entry_mode << " entry_refine_ms=" << entry_refine_ms
                          << " entry_beam=" << entry_beam << " entry_budget=" << entry_budget
                          << " navigation_fp32_lanes=" << nav_fp32_lanes
                          << " resident_queries=" << resident
                          << " parallel_gather=" << parallel_gather << " adaptive=" << adaptive
                          << " average_cards=" << average_cards << " bloom=" << PHMS_V3_BF_ENTRIES
                          << " max_query_batch=" << max_query_batch
                          << " balanced_batches=" << balanced_batches
                          << " cost_batches=" << cost_batches
                          << " region_grouping=" << bool(query_regions)
                          << " seed_locality=" << seed_locality
                          << " seed_counting=" << seed_counting << " park_workers=" << park_workers
                          << " independent_search=" << independent
                          << " route_probe_cards=" << route_probe
                          << " replica_aware_route=" << replica_route
                          << " organization_ms=" << organization_ms
                          << " expansion_budget=" << phms_v3::expansion_budget(L)
                          << " final_candidates_per_shard=" << final_candidates
                          << " adaptive_budget=" << adaptive_budget
                          << " coalesce_min=" << coalesce_min
                          << " coalesced_queries=" << coalesced_queries << " worker_max_ms="
                          << *std::max_element(worker_ms.begin(), worker_ms.end())
                          << " worker_gather_max_ms="
                          << *std::max_element(worker_gather_ms.begin(), worker_gather_ms.end())
                          << std::endl;
            if (timed)
                for (unsigned c = 1; c <= PHMS_GC_GROUPS; ++c)
                    if (card_counts[c])
                        std::cout << "[V5 RECALL BY CARDS] cards=" << c
                                  << " queries=" << card_counts[c] << " budget=" << budgets[c - 1]
                                  << " recall="
                                  << double(hits_by_cards[c]) / (card_counts[c] * 10ull) << '\n';
        };
        std::cout << "[V4] workers=" << workers << " threads_per_worker=" << threads
                  << " jobs=" << jobs.size() << " fixed_cards=" << (adaptive ? 0 : cards)
                  << " CPU_full_graph=true PQ_replication_unchanged=true\n";
        std::cout << "[V5 ROUTE] adaptive=" << adaptive << " min_cards=" << policy.minimum
                  << " max_cards=" << policy.maximum << " vote_coverage=" << policy.coverage
                  << " distance_ratio=" << policy.distance_ratio
                  << " average_cards=" << average_cards << " bloom=" << PHMS_V3_BF_ENTRIES << "\n";
        std::cout << "[V5 BATCH] max_query_batch=" << max_query_batch << " workers=" << workers
                  << " actual_jobs=" << jobs.size() << '\n';
        std::cout << "[V5 COALESCE] minimum_queries=" << coalesce_min
                  << " promoted_queries=" << coalesced_queries << " mode=" << coalesce_mode << '\n';
        std::cout << "[V5 ORGANIZATION] balanced_batches=" << balanced_batches
                  << " seed_locality=" << seed_locality << '\n';
        for (unsigned c = 1; c <= PHMS_GC_GROUPS; ++c)
            std::cout << "[V5 ROUTE HIST] cards=" << c << " queries=" << card_counts[c] << '\n';
        for (unsigned c = 1; c <= PHMS_GC_GROUPS; ++c)
            std::cout << "[V5 BUDGET] cards=" << c << " budget=" << budgets[c - 1] << '\n';
        std::cout << "[V4 TIMING] primary=request wall; resident_queries=" << resident
                  << " preload_ms=" << preload_ms << " additional_resident_GPU_bytes="
                  << (resident ? uint64_t(PHMS_GC_GROUPS - 1) * hostq.n * 4 : 0)
                  << "; resident mode preloads Query ONLY (like v6), timed selection/gather, LUT "
                     "rebuild, waits, rerank and scatter retained; fine_entry="
                  << fine_entry << "\n";
        std::array<uint64_t, 8> gpu_queries{};
        for (unsigned i = 0; i < jobs.size(); ++i)
        {
            auto& j = *jobs[i];
            std::cout << "[V4 JOB] " << i << " mask=" << j.batch.mask
                      << " queries=" << j.batch.ids.size() << " GPUs=";
            for (int g : j.gpus)
            {
                gpu_queries[g] += j.batch.ids.size();
                std::cout << g << ' ';
            }
            std::cout << '\n';
        }
        for (unsigned g = 0; g < PHMS_GC_GROUPS; ++g)
            std::cout << "[V4 LOAD] GPU=" << g << " query_tasks=" << gpu_queries[g] << '\n';
        auto memory_snapshot = [&](const char* suffix)
        {
            const char* prefix = getenv("PHMS_MEMORY_SNAPSHOT_PREFIX");
            if (prefix && *prefix)
                for (const char* name : {"smaps_rollup", "numa_maps"})
                {
                    std::ifstream input(std::string("/proc/self/") + name);
                    std::ofstream out(std::string(prefix) + "." + suffix + "." + name + ".txt");
                    need(bool(input) && bool(out), "memory snapshot failed");
                    out << input.rdbuf();
                }
        };
        memory_snapshot("before");
        setenv("PHMS_TIMELINE_ACTIVE", "0", 1);
        request(false);
        if (entry_ab)
        {
            use_region_entry = false;
            route();
            need(masks == frozen_masks && seeds == base_seeds, "baseline routing changed");
            gather();
            request(false);
        }
        collect_tracks =
            diagnostic || route_probe ||
            (getenv("PHMS_ROUTE_PROBE_STATS") && atoi(getenv("PHMS_ROUTE_PROBE_STATS")));
        unsigned repeats = 5;
        if (getenv("PHMS_V4_REPEATS"))
            repeats = std::stoul(getenv("PHMS_V4_REPEATS"));
        need(repeats > 0, "zero repeats");
        for (unsigned r = 0; r < repeats; ++r)
        {
#ifdef PHMS_TIMELINE
            bool capture =
                getenv("PHMS_TRACE_REPEAT") && r == std::stoul(getenv("PHMS_TRACE_REPEAT"));
            if (capture)
            {
                setenv("PHMS_TIMELINE_ACTIVE", "1", 1);
                ck(cudaSetDevice(0));
                ck(cudaProfilerStart());
            }
#endif
            if (entry_ab)
            {
                for (unsigned i = 0; i < 2; ++i)
                {
                    use_region_entry = ((r % 2) ? i == 0 : i == 1);
                    request(true);
                }
            }
            else
                request(true);
#ifdef PHMS_TIMELINE
            if (capture)
            {
                ck(cudaSetDevice(0));
                ck(cudaProfilerStop());
                setenv("PHMS_TIMELINE_ACTIVE", "0", 1);
            }
#endif
        }
        memory_snapshot("after");
        for (int g = 0; g < PHMS_GC_GROUPS; ++g)
        {
            ck(cudaSetDevice(g));
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ck(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::cout << "[V4 MEMORY] GPU=" << g
                      << " device_used_bytes_after=" << total_bytes - free_bytes << '\n';
        }
        std::ifstream status("/proc/self/status");
        std::string memory_line;
        while (std::getline(status, memory_line))
            if (memory_line.compare(0, 6, "VmHWM:") == 0 ||
                memory_line.compare(0, 6, "VmRSS:") == 0 ||
                memory_line.compare(0, 13, "HugetlbPages:") == 0)
                std::cout << "[V4 MEMORY] " << memory_line << '\n';
        for (auto& j : jobs)
            for (unsigned i = 0; i < j->objects.size(); ++i)
            {
                ck(cudaSetDevice(j->gpus[i]));
                j->objects[i]->phms_free();
            }
        ck(cudaSetDevice(0));
        cb(cublasDestroy(blas));
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[V4 error] " << e.what() << std::endl;
        return 1;
    }
}
