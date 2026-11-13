#ifndef PHMS_SHARD_ADJACENCY_H
#define PHMS_SHARD_ADJACENCY_H
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr size_t PHMS_ADJ_HEADER_BYTES = 4096;
struct PhmsAdjHeader
{
    char magic[8];
    uint32_t version;
    uint32_t complete;
    uint32_t nodes;
    uint32_t degree;
    uint32_t shards;
    uint32_t dimension;
    uint32_t element_bytes;
    uint32_t record_words;
    uint64_t entry_bytes;
    uint64_t source_bytes;
    uint64_t source_device;
    uint64_t source_inode;
    int64_t source_mtime_sec;
    int64_t source_mtime_nsec;
    uint64_t file_bytes;
    uint32_t ends[8];
};
static_assert(sizeof(PhmsAdjHeader) <= PHMS_ADJ_HEADER_BYTES, "sidecar header size");

inline unsigned phms_adj_record_words(unsigned r, unsigned g)
{
    return (2 + 2 * g + r + 1) & ~1u;
}
inline unsigned phms_adj_owner(unsigned id, unsigned g, const uint32_t* ends)
{
    unsigned owner = 0;
    while (owner + 1 < g && id >= ends[owner])
        ++owner;
    return owner;
}
inline void phms_adj_group(const uint32_t* input, unsigned n, unsigned r, unsigned g,
                           const uint32_t* ends, uint32_t* record)
{
    if (r > 64 || n > r || g < 1 || g > 7)
        throw std::runtime_error("invalid shard record shape");
    auto* prefix = reinterpret_cast<uint8_t*>(record);
    std::memset(record, 0, phms_adj_record_words(r, g) * sizeof(uint32_t));
    for (unsigned i = 0; i < n; ++i)
    {
        if (input[i] >= ends[g - 1])
            throw std::runtime_error("invalid graph neighbour ID");
        ++prefix[phms_adj_owner(input[i], g, ends) + 1];
    }
    for (unsigned s = 0; s < g; ++s)
        prefix[s + 1] += prefix[s];
    unsigned cursor[8] = {};
    for (unsigned s = 0; s < g; ++s)
        cursor[s] = prefix[s];
    for (unsigned i = 0; i < n; ++i)
    {
        const unsigned s = phms_adj_owner(input[i], g, ends);
        record[2 + 2 * g + cursor[s]++] = input[i];
        record[2 + 2 * s + (i >= 32)] |= 1u << (i % 32);
    }
}

class PhmsShardAdjacency
{
    int fd_ = -1;
    void* mapping_ = MAP_FAILED;
    size_t bytes_ = 0;

  public:
    PhmsAdjHeader header{};
    explicit PhmsShardAdjacency(const char* path, const std::string& source)
    {
        fd_ = open(path, O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("cannot open offline shard adjacency");
        struct stat fs
        {
        }, src{};
        bool ok = fstat(fd_, &fs) == 0 && stat(source.c_str(), &src) == 0 &&
                  pread(fd_, &header, sizeof(header), 0) == (ssize_t)sizeof(header);
        ok = ok && std::memcmp(header.magic, "PHMSADJ1", 8) == 0 && header.version == 1 &&
             header.complete == 1 && header.shards >= 1 && header.shards <= 7 &&
             header.degree <= 64 &&
             header.record_words == phms_adj_record_words(header.degree, header.shards) &&
             header.file_bytes ==
                 PHMS_ADJ_HEADER_BYTES + (uint64_t)header.nodes * header.record_words * 4 &&
             (uint64_t)fs.st_size == header.file_bytes &&
             (uint64_t)src.st_size == header.source_bytes &&
             (uint64_t)src.st_dev == header.source_device &&
             (uint64_t)src.st_ino == header.source_inode &&
             src.st_mtim.tv_sec == header.source_mtime_sec &&
             src.st_mtim.tv_nsec == header.source_mtime_nsec;
        for (unsigned g = 0; ok && g < header.shards; ++g)
            ok = header.ends[g] == (uint64_t)header.nodes * (g + 1) / header.shards;
        if (!ok)
        {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("offline adjacency header/source identity mismatch");
        }
        bytes_ = fs.st_size;
        mapping_ = mmap(nullptr, bytes_, PROT_READ, MAP_SHARED | MAP_POPULATE, fd_, 0);
        if (mapping_ == MAP_FAILED)
        {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("offline adjacency mmap failed");
        }
        volatile uint64_t sum = 0;
        const volatile uint8_t* p = static_cast<const volatile uint8_t*>(mapping_);
        for (size_t offset = 0; offset < bytes_; offset += 4096)
            sum += p[offset];
        (void)sum;
    }
    PhmsShardAdjacency(const PhmsShardAdjacency&) = delete;
    PhmsShardAdjacency& operator=(const PhmsShardAdjacency&) = delete;
    ~PhmsShardAdjacency()
    {
        if (mapping_ != MAP_FAILED)
            munmap(mapping_, bytes_);
        if (fd_ >= 0)
            close(fd_);
    }
    const uint32_t* record(unsigned node) const
    {
        return reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(mapping_) +
                                                 PHMS_ADJ_HEADER_BYTES) +
               (uint64_t)node * header.record_words;
    }
};
#endif
