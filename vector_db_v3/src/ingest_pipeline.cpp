#include "vector_db_v3/ingest_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#if defined(VECTOR_DB_V3_CUDA_ENABLED) && VECTOR_DB_V3_CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace vector_db_v3::ingest {

namespace {

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

std::size_t env_fail_after_batches() {
    const char* value = std::getenv("VECTOR_DB_V3_INGEST_FAIL_AFTER_BATCHES");
    if (value == nullptr || *value == '\0') {
        return 0U;
    }
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (...) {
        return 0U;
    }
}

std::size_t env_size_t(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    try {
        const std::size_t parsed = static_cast<std::size_t>(std::stoull(value));
        return parsed > 0U ? parsed : fallback;
    } catch (...) {
        return fallback;
    }
}

double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

bool try_enable_pinned_mode(bool requested) {
    if (!requested) {
        return false;
    }
#if defined(VECTOR_DB_V3_CUDA_ENABLED) && VECTOR_DB_V3_CUDA_ENABLED
    void* ptr = nullptr;
    const cudaError_t st = cudaHostAlloc(&ptr, 4096U, cudaHostAllocDefault);
    if (st != cudaSuccess) {
        return false;
    }
    cudaFreeHost(ptr);
    return true;
#else
    return false;
#endif
}

Status run_sync(
    VectorStore* store,
    const PipelineOptions& options,
    const BatchProducer& producer,
    PipelineStats* stats_out) {
    const std::size_t fail_after_batches = env_fail_after_batches();
    const std::size_t reserve_capacity = options.reserved_batch_capacity > 0U
        ? options.reserved_batch_capacity
        : options.batch_size;
    bool eof = false;
    RecordBatch batch;
    if (reserve_capacity > 0U) {
        batch.reserve(reserve_capacity);
    }
    while (!eof) {
        batch.clear();
        const auto producer_start = std::chrono::steady_clock::now();
        const Status p = producer(&batch, &eof);
        if (!p.ok) {
            return p;
        }
        if (stats_out != nullptr) {
            stats_out->producer_wait_ms += elapsed_ms_since(producer_start);
        }
        if (!batch.empty()) {
            const auto commit_start = std::chrono::steady_clock::now();
            const Status s = store->insert_batch_with_options(batch, true);
            if (!s.ok) {
                return s;
            }
            if (stats_out != nullptr) {
                stats_out->records_committed += batch.size();
                stats_out->batches_committed += 1U;
                stats_out->commit_apply_ms += elapsed_ms_since(commit_start);
            }
            if (fail_after_batches > 0U &&
                stats_out != nullptr &&
                stats_out->batches_committed >= fail_after_batches) {
                return Status::Error("ingest pipeline forced failure after batch threshold");
            }
        }
    }
    return store->flush_ingest_state();
}

Status run_async(
    VectorStore* store,
    const PipelineOptions& options,
    const BatchProducer& producer,
    PipelineStats* stats_out) {
    const std::size_t max_queue = std::max<std::size_t>(1U, options.queue_capacity_batches);
    const std::size_t fail_after_batches = env_fail_after_batches();
    const std::size_t reserve_capacity = options.reserved_batch_capacity > 0U
        ? options.reserved_batch_capacity
        : options.batch_size;
    std::deque<std::unique_ptr<RecordBatch>> queue;
    std::deque<std::unique_ptr<RecordBatch>> free_batches;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    std::atomic<bool> cancelled{false};
    bool producer_done = false;
    Status producer_status = Status::Ok();
    Status consumer_status = Status::Ok();
    double producer_wait_ms = 0.0;
    double consumer_wait_ms = 0.0;
    std::size_t peak_queue_depth = 0U;

    std::thread producer_thread([&]() {
        bool eof = false;
        std::unique_ptr<RecordBatch> local_batch;
        while (!eof && !cancelled.load()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (!free_batches.empty()) {
                    local_batch = std::move(free_batches.front());
                    free_batches.pop_front();
                }
            }
            if (!local_batch) {
                local_batch = std::make_unique<RecordBatch>();
                if (reserve_capacity > 0U) {
                    local_batch->reserve(reserve_capacity);
                }
            } else {
                local_batch->clear();
            }

            const auto produce_start = std::chrono::steady_clock::now();
            Status p = producer(local_batch.get(), &eof);
            if (!p.ok) {
                std::lock_guard<std::mutex> lk(mtx);
                producer_status = p;
                producer_done = true;
                cv_not_empty.notify_all();
                cv_not_full.notify_all();
                return;
            }
            if (stats_out != nullptr) {
                producer_wait_ms += elapsed_ms_since(produce_start);
            }

            if (local_batch->empty() && eof) {
                break;
            }
            if (!local_batch->empty()) {
                std::unique_lock<std::mutex> lk(mtx);
                const auto wait_start = std::chrono::steady_clock::now();
                cv_not_full.wait(lk, [&]() {
                    return queue.size() < max_queue || cancelled.load();
                });
                if (stats_out != nullptr) {
                    producer_wait_ms += elapsed_ms_since(wait_start);
                }
                if (cancelled.load()) {
                    break;
                }
                queue.push_back(std::move(local_batch));
                peak_queue_depth = std::max<std::size_t>(peak_queue_depth, queue.size());
                cv_not_empty.notify_one();
            }
        }
        std::lock_guard<std::mutex> lk(mtx);
        producer_done = true;
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    });

    while (true) {
        std::unique_ptr<RecordBatch> batch;
        {
            std::unique_lock<std::mutex> lk(mtx);
            const auto wait_start = std::chrono::steady_clock::now();
            cv_not_empty.wait(lk, [&]() {
                return !queue.empty() || producer_done;
            });
            if (stats_out != nullptr) {
                consumer_wait_ms += elapsed_ms_since(wait_start);
            }
            if (queue.empty() && producer_done) {
                break;
            }
            if (!queue.empty()) {
                batch = std::move(queue.front());
                queue.pop_front();
                cv_not_full.notify_one();
            }
        }
        if (batch && !batch->empty()) {
            const auto commit_start = std::chrono::steady_clock::now();
            consumer_status = store->insert_batch_with_options(*batch, true);
            if (!consumer_status.ok) {
                cancelled.store(true);
                cv_not_full.notify_all();
                cv_not_empty.notify_all();
                break;
            }
            if (stats_out != nullptr) {
                stats_out->records_committed += batch->size();
                stats_out->batches_committed += 1U;
                stats_out->commit_apply_ms += elapsed_ms_since(commit_start);
            }
            batch->clear();
            {
                std::lock_guard<std::mutex> lk(mtx);
                free_batches.push_back(std::move(batch));
            }
            if (fail_after_batches > 0U &&
                stats_out != nullptr &&
                stats_out->batches_committed >= fail_after_batches) {
                consumer_status = Status::Error("ingest pipeline forced failure after batch threshold");
                cancelled.store(true);
                cv_not_full.notify_all();
                cv_not_empty.notify_all();
                break;
            }
        }
    }

    if (producer_thread.joinable()) {
        producer_thread.join();
    }
    if (!consumer_status.ok) {
        return consumer_status;
    }
    if (!producer_status.ok) {
        return producer_status;
    }
    if (stats_out != nullptr) {
        stats_out->producer_wait_ms += producer_wait_ms;
        stats_out->consumer_wait_ms += consumer_wait_ms;
        stats_out->peak_queue_depth = peak_queue_depth;
    }
    return store->flush_ingest_state();
}

}  // namespace

Status run_pipeline(
    VectorStore* store,
    const PipelineOptions& options,
    const BatchProducer& producer,
    PipelineStats* stats_out) {
    if (store == nullptr) {
        return Status::Error("ingest pipeline: store is null");
    }
    if (!producer) {
        return Status::Error("ingest pipeline: producer is null");
    }
    if (stats_out != nullptr) {
        *stats_out = PipelineStats{};
    }

    const std::string async_policy = [&]() {
        const char* raw = std::getenv("VECTOR_DB_V3_INGEST_ASYNC_POLICY");
        if (raw == nullptr || *raw == '\0') {
            return std::string("explicit");
        }
        std::string value(raw);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }();
    bool async_enabled = options.async_enabled ||
        env_truthy(std::getenv("VECTOR_DB_V3_INGEST_ASYNC_MODE"));
    if (async_policy == "off") {
        async_enabled = false;
    } else if (async_policy == "on") {
        async_enabled = true;
    }
    const bool pinned_requested = options.request_pinned ||
        env_truthy(std::getenv("VECTOR_DB_V3_INGEST_PINNED"));
    const bool pinned_enabled = try_enable_pinned_mode(pinned_requested);
    PipelineOptions effective_options = options;
    effective_options.queue_capacity_batches = env_size_t(
        "VECTOR_DB_V3_INGEST_QUEUE_CAPACITY",
        std::max<std::size_t>(1U, options.queue_capacity_batches));
    effective_options.reserved_batch_capacity = env_size_t(
        "VECTOR_DB_V3_INGEST_PRODUCER_CHUNK",
        options.reserved_batch_capacity > 0U ? options.reserved_batch_capacity : options.batch_size);

    if (stats_out != nullptr) {
        stats_out->async_enabled = async_enabled;
        stats_out->pinned_enabled = pinned_enabled;
        stats_out->pinned_mode = pinned_enabled ? "pinned" : "standard";
    }

    if (!async_enabled) {
        return run_sync(store, effective_options, producer, stats_out);
    }
    return run_async(store, effective_options, producer, stats_out);
}

}  // namespace vector_db_v3::ingest
