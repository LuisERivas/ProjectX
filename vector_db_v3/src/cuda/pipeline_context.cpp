#include "vector_db_v3/cuda_pipeline_context.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#if VECTOR_DB_V3_CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace vector_db_v3::kmeans {

namespace {

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::size_t parse_max_bytes() {
    constexpr std::size_t kDefaultMaxBytes = static_cast<std::size_t>(4ULL) * 1024ULL * 1024ULL * 1024ULL;
    const char* raw = std::getenv("VECTOR_DB_V3_GPU_RESIDENCY_MAX_BYTES");
    if (raw == nullptr || *raw == '\0') {
        return kDefaultMaxBytes;
    }
    try {
        return static_cast<std::size_t>(std::stoull(raw));
    } catch (...) {
        return kDefaultMaxBytes;
    }
}

}  // namespace

struct CudaPipelineContext::Impl {
    struct BufferSlot {
        void* ptr = nullptr;
        std::size_t capacity_bytes = 0U;
        bool used_this_stage = false;
        std::uint64_t last_host_token = 0ULL;
        std::size_t last_host_bytes = 0U;
    };

    std::unordered_map<std::string, BufferSlot> slots;
    mutable std::mutex mu;
    std::size_t total_allocated_bytes = 0U;
    ResidencyStats stats{};
};

CudaPipelineContext::CudaPipelineContext(GpuResidencyMode mode, std::size_t max_bytes, bool enabled)
    : mode_(mode), max_bytes_(max_bytes), enabled_(enabled), impl_(std::make_unique<Impl>()) {
    impl_->stats.residency_mode = mode_to_string(mode);
}

CudaPipelineContext::~CudaPipelineContext() {
    release_all();
}

std::shared_ptr<CudaPipelineContext> CudaPipelineContext::create_from_env() {
    const GpuResidencyMode mode = mode_from_env();
    const std::size_t max_bytes = parse_max_bytes();
    const bool force_off = env_truthy(std::getenv("VECTOR_DB_V3_GPU_RESIDENCY_FORCE_DISABLE"));

#if VECTOR_DB_V3_CUDA_ENABLED
    const bool enabled = !force_off && (mode == GpuResidencyMode::Stage || mode == GpuResidencyMode::Auto);
#else
    const bool enabled = false;
#endif
    return std::shared_ptr<CudaPipelineContext>(new CudaPipelineContext(mode, max_bytes, enabled));
}

GpuResidencyMode CudaPipelineContext::mode_from_env() {
    const char* raw = std::getenv("VECTOR_DB_V3_GPU_RESIDENCY_MODE");
    if (raw == nullptr || *raw == '\0') {
        return GpuResidencyMode::Auto;
    }
    std::string mode(raw);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode == "off") {
        return GpuResidencyMode::Off;
    }
    if (mode == "stage") {
        return GpuResidencyMode::Stage;
    }
    return GpuResidencyMode::Auto;
}

std::string CudaPipelineContext::mode_to_string(GpuResidencyMode mode) {
    switch (mode) {
        case GpuResidencyMode::Stage:
            return "stage";
        case GpuResidencyMode::Auto:
            return "auto";
        case GpuResidencyMode::Off:
        default:
            return "off";
    }
}

bool CudaPipelineContext::enabled() const {
    return enabled_;
}

GpuResidencyMode CudaPipelineContext::mode() const {
    return mode_;
}

Status CudaPipelineContext::acquire_buffer(
    const std::string& key,
    std::size_t bytes,
    void** ptr_out,
    bool* reused_out) {
    if (ptr_out == nullptr) {
        return Status::Error("gpu residency context: ptr_out is null");
    }
    *ptr_out = nullptr;
    if (reused_out != nullptr) {
        *reused_out = false;
    }
    if (bytes == 0U) {
        return Status::Ok();
    }
    if (!enabled_) {
        return Status::Error("gpu residency context: disabled");
    }

#if VECTOR_DB_V3_CUDA_ENABLED
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto& slot = impl_->slots[key];
    if (slot.ptr != nullptr && slot.capacity_bytes >= bytes) {
        slot.used_this_stage = true;
        if (reused_out != nullptr) {
            *reused_out = true;
        }
        impl_->stats.cache_hits += 1ULL;
        impl_->stats.bytes_reused += static_cast<std::uint64_t>(bytes);
        *ptr_out = slot.ptr;
        return Status::Ok();
    }

    impl_->stats.cache_misses += 1ULL;
    const std::size_t old_capacity = slot.capacity_bytes;
    const std::size_t grow_to = std::max(bytes, old_capacity == 0U ? bytes : old_capacity * 2U);

    auto attempt_alloc = [&](std::size_t request_bytes) -> Status {
        if (impl_->total_allocated_bytes + request_bytes > max_bytes_) {
            return Status::Error("gpu residency context: max bytes exceeded");
        }
        void* new_ptr = nullptr;
        const cudaError_t alloc_st = cudaMalloc(&new_ptr, request_bytes);
        if (alloc_st != cudaSuccess) {
            return Status::Error(std::string("gpu residency context: cudaMalloc failed: ") + cudaGetErrorString(alloc_st));
        }
        impl_->stats.alloc_calls += 1ULL;
        if (slot.ptr != nullptr) {
            cudaFree(slot.ptr);
            impl_->total_allocated_bytes -= slot.capacity_bytes;
        }
        slot.ptr = new_ptr;
        slot.capacity_bytes = request_bytes;
        slot.used_this_stage = true;
        slot.last_host_token = 0ULL;
        slot.last_host_bytes = 0U;
        impl_->total_allocated_bytes += request_bytes;
        *ptr_out = slot.ptr;
        return Status::Ok();
    };

    Status st = attempt_alloc(grow_to);
    if (st.ok) {
        return st;
    }

    // Retry once after releasing currently-unused slots.
    for (auto it = impl_->slots.begin(); it != impl_->slots.end();) {
        auto& candidate = it->second;
        if (candidate.used_this_stage || candidate.ptr == nullptr) {
            ++it;
            continue;
        }
        cudaFree(candidate.ptr);
        impl_->total_allocated_bytes -= candidate.capacity_bytes;
        it = impl_->slots.erase(it);
    }
    st = attempt_alloc(bytes);
    if (!st.ok) {
        return Status::Error(st.message + "; release_unused_then_retry_failed");
    }
    return st;
#else
    (void)key;
    return Status::Error("gpu residency context: cuda_not_compiled");
#endif
}

Status CudaPipelineContext::copy_h2d(void* dst, const void* src, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
#if VECTOR_DB_V3_CUDA_ENABLED
    const cudaError_t st = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memcpy h2d failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
#else
    (void)dst;
    (void)src;
    (void)label;
    return Status::Error("cuda memcpy h2d failed: cuda_not_compiled");
#endif
}

Status CudaPipelineContext::copy_h2d_if_changed(
    const std::string& key,
    std::uint64_t host_token,
    void* dst,
    const void* src,
    std::size_t bytes,
    const char* label,
    bool* skipped_out) {
    if (skipped_out != nullptr) {
        *skipped_out = false;
    }
    if (!enabled_ || bytes == 0U) {
        return copy_h2d(dst, src, bytes, label);
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        auto it = impl_->slots.find(key);
        if (it != impl_->slots.end()) {
            auto& slot = it->second;
            if (slot.ptr == dst && slot.last_host_token == host_token && slot.last_host_bytes == bytes) {
                impl_->stats.bytes_h2d_saved_est += static_cast<std::uint64_t>(bytes);
                if (skipped_out != nullptr) {
                    *skipped_out = true;
                }
                return Status::Ok();
            }
        }
    }
    const Status st = copy_h2d(dst, src, bytes, label);
    if (!st.ok) {
        return st;
    }
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->slots.find(key);
    if (it != impl_->slots.end() && it->second.ptr == dst) {
        it->second.last_host_token = host_token;
        it->second.last_host_bytes = bytes;
    }
    return Status::Ok();
}

Status CudaPipelineContext::copy_d2h(void* dst, const void* src, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
#if VECTOR_DB_V3_CUDA_ENABLED
    const cudaError_t st = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memcpy d2h failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
#else
    (void)dst;
    (void)src;
    (void)label;
    return Status::Error("cuda memcpy d2h failed: cuda_not_compiled");
#endif
}

Status CudaPipelineContext::memset(void* ptr, int value, std::size_t bytes, const char* label) {
    if (bytes == 0U) {
        return Status::Ok();
    }
#if VECTOR_DB_V3_CUDA_ENABLED
    const cudaError_t st = cudaMemset(ptr, value, bytes);
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda memset failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
#else
    (void)ptr;
    (void)value;
    (void)label;
    return Status::Error("cuda memset failed: cuda_not_compiled");
#endif
}

Status CudaPipelineContext::sync(const char* label) {
#if VECTOR_DB_V3_CUDA_ENABLED
    const cudaError_t st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        return Status::Error(std::string("cuda sync failed (") + label + "): " + cudaGetErrorString(st));
    }
    return Status::Ok();
#else
    (void)label;
    return Status::Error("cuda sync failed: cuda_not_compiled");
#endif
}

void CudaPipelineContext::reset_stage() {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->mu);
    for (auto& kv : impl_->slots) {
        kv.second.used_this_stage = false;
    }
}

void CudaPipelineContext::release_all() {
    if (!impl_) {
        return;
    }
#if VECTOR_DB_V3_CUDA_ENABLED
    std::lock_guard<std::mutex> lock(impl_->mu);
    for (auto& kv : impl_->slots) {
        if (kv.second.ptr != nullptr) {
            cudaFree(kv.second.ptr);
            kv.second.ptr = nullptr;
            kv.second.capacity_bytes = 0U;
        }
    }
    impl_->slots.clear();
    impl_->total_allocated_bytes = 0U;
#endif
}

ResidencyStats CudaPipelineContext::stats() const {
    if (!impl_) {
        return ResidencyStats{};
    }
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->stats;
}

}  // namespace vector_db_v3::kmeans
