#ifndef UTILS_H_
#define UTILS_H_

#include <unistd.h>
#include <stdio.h>
#include <chrono>
#include <fcntl.h>

#include <cuda_runtime.h>

#include <iostream>

#define gpuErrchk(ans)                                                                             \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

off_t caclulate_filesize(const char* chFileName)
{
    int fd = -1;

    if ((fd = open(chFileName, O_RDONLY, (mode_t)0)) == -1)
    {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }

    long cur_pos = 0L;
    off_t file_size = 0L;

    cur_pos = lseek(fd, 0, SEEK_CUR);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, cur_pos, SEEK_CUR);
    close(fd);
    return file_size;
}

unsigned long long log_message(const char* message)
{
    const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    const std::time_t ctimenow_obj = std::chrono::system_clock::to_time_t(now);
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    struct tm* local = localtime(&ctimenow_obj);
    std::cout << local->tm_mday << ":" << local->tm_mon << ":" << local->tm_year << ":"
              << local->tm_hour << ":" << local->tm_min << ":" << local->tm_sec << " [" << millis
              << "] : " << message << std::endl;
    return millis;
}

#include <sys/time.h>
#include <cuda_runtime.h>

struct CPUTimer
{
    int err;
    double start;
    double stop;
    struct timeval Tv;

    CPUTimer()
    {
        start = 0.0;
        stop = 0.0;
        err = 0;
    }

    void Start()
    {
        err = gettimeofday(&Tv, NULL);
        if (!err)
            start = Tv.tv_sec + Tv.tv_usec * 1.0e-6;
    }

    void Stop()
    {
        err = gettimeofday(&Tv, NULL);
        if (!err)
            stop = Tv.tv_sec + Tv.tv_usec * 1.0e-6;
    }

    double Elapsed()
    {
        double elapsed;
        elapsed = stop - start;
        return elapsed;
    }
};

struct GPUTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t streamToRecord;
    bool m_bDisable;

    GPUTimer(cudaStream_t& streamInParam, bool bDisable = true) : m_bDisable(bDisable)
    {
        if (m_bDisable)
            return;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        streamToRecord = streamInParam;
    }

    ~GPUTimer()
    {
        if (m_bDisable)
            return;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        if (m_bDisable)
            return;

        cudaEventRecord(start, streamToRecord);
    }

    void Stop()
    {
        if (m_bDisable)
            return;

        cudaEventRecord(stop, streamToRecord);
    }

    float Elapsed()
    {
        if (m_bDisable)
            return 0;

        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

#endif
