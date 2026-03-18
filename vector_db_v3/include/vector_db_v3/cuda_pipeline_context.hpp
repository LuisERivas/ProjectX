#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "vector_db_v3/status.hpp"

namespace vector_db_v3::kmeans {

enum class GpuResidencyMode {
    Off = 0,
    Stage = 1,
    Auto = 2,
};

struct ResidencyStats {
    std::string residency_mode = "off";
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t bytes_reused = 0;
    std::uint64_t bytes_h2d_saved_est = 0;
    std::uint64_t alloc_calls = 0;
};

class CudaPipelineContext {
public:
    ~CudaPipelineContext();

    CudaPipelineContext(const CudaPipelineContext&) = delete;
    CudaPipelineContext& operator=(const CudaPipelineContext&) = delete;

    static std::shared_ptr<CudaPipelineContext> create_from_env();
    static GpuResidencyMode mode_from_env();
    static std::string mode_to_string(GpuResidencyMode mode);

    bool enabled() const;
    GpuResidencyMode mode() const;

    Status acquire_buffer(const std::string& key, std::size_t bytes, void** ptr_out, bool* reused_out);
    Status copy_h2d(void* dst, const void* src, std::size_t bytes, const char* label);
    Status copy_h2d_if_changed(
        const std::string& key,
        std::uint64_t host_token,
        void* dst,
        const void* src,
        std::size_t bytes,
        const char* label,
        bool* skipped_out);
    Status copy_d2h(void* dst, const void* src, std::size_t bytes, const char* label);
    Status memset(void* ptr, int value, std::size_t bytes, const char* label);
    Status sync(const char* label);

    void reset_stage();
    void release_all();
    ResidencyStats stats() const;

private:
    CudaPipelineContext(GpuResidencyMode mode, std::size_t max_bytes, bool enabled);

    GpuResidencyMode mode_ = GpuResidencyMode::Off;
    std::size_t max_bytes_ = 0U;
    bool enabled_ = false;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vector_db_v3::kmeans
