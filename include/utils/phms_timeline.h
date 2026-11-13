#pragma once
#include <cstdlib>
#ifdef PHMS_TIMELINE
#include <nvToolsExt.h>
#endif
inline bool phms_trace_enabled()
{
#ifdef PHMS_TIMELINE
    const char* value = std::getenv("PHMS_TIMELINE_ACTIVE");
    return value && std::atoi(value) != 0;
#else
    return false;
#endif
}
class PhmsTraceRange
{
    bool enabled_ = false;
    bool open_ = false;

  public:
    explicit PhmsTraceRange(const char* name) : enabled_(phms_trace_enabled())
    {
        set(name);
    }
    void end()
    {
#ifdef PHMS_TIMELINE
        if (open_)
            nvtxRangePop();
#endif
        open_ = false;
    }
    void set(const char* name)
    {
        end();
#ifdef PHMS_TIMELINE
        if (enabled_)
        {
            nvtxRangePushA(name);
            open_ = true;
        }
#endif
    }
    ~PhmsTraceRange()
    {
        end();
    }
    PhmsTraceRange(const PhmsTraceRange&) = delete;
};
