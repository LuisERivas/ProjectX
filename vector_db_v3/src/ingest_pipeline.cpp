#include "vector_db_v3/ingest_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
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
    const BatchProducer& producer,
    PipelineStats* stats_out) {
    const std::size_t fail_after_batches = env_fail_after_batches();
    bool eof = false;
    while (!eof) {
        RecordBatch batch;
        const Status p = producer(&batch, &eof);
        if (!p.ok) {
            return p;
        }
        if (!batch.empty()) {
            const Status s = store->insert_batch_with_options(batch, true);
            if (!s.ok) {
                return s;
            }
            if (stats_out != nullptr) {
                stats_out->records_committed += batch.size();
                stats_out->batches_committed += 1U;
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
    std::deque<RecordBatch> queue;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    std::atomic<bool> cancelled{false};
    bool producer_done = false;
    Status producer_status = Status::Ok();
    Status consumer_status = Status::Ok();

    std::thread producer_thread([&]() {
        bool eof = false;
        while (!eof && !cancelled.load()) {
            RecordBatch batch;
            Status p = producer(&batch, &eof);
            if (!p.ok) {
                std::lock_guard<std::mutex> lk(mtx);
                producer_status = p;
                producer_done = true;
                cv_not_empty.notify_all();
                cv_not_full.notify_all();
                return;
            }
            if (batch.empty() && eof) {
                break;
            }
            if (!batch.empty()) {
                std::unique_lock<std::mutex> lk(mtx);
                cv_not_full.wait(lk, [&]() {
                    return queue.size() < max_queue || cancelled.load();
                });
                if (cancelled.load()) {
                    break;
                }
                queue.push_back(std::move(batch));
                cv_not_empty.notify_one();
            }
        }
        std::lock_guard<std::mutex> lk(mtx);
        producer_done = true;
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    });

    while (true) {
        RecordBatch batch;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_not_empty.wait(lk, [&]() {
                return !queue.empty() || producer_done;
            });
            if (queue.empty() && producer_done) {
                break;
            }
            if (!queue.empty()) {
                batch = std::move(queue.front());
                queue.pop_front();
                cv_not_full.notify_one();
            }
        }
        if (!batch.empty()) {
            consumer_status = store->insert_batch_with_options(batch, true);
            if (!consumer_status.ok) {
                cancelled.store(true);
                cv_not_full.notify_all();
                cv_not_empty.notify_all();
                break;
            }
            if (stats_out != nullptr) {
                stats_out->records_committed += batch.size();
                stats_out->batches_committed += 1U;
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

    const bool async_enabled = options.async_enabled ||
        env_truthy(std::getenv("VECTOR_DB_V3_INGEST_ASYNC_MODE"));
    const bool pinned_requested = options.request_pinned ||
        env_truthy(std::getenv("VECTOR_DB_V3_INGEST_PINNED"));
    const bool pinned_enabled = try_enable_pinned_mode(pinned_requested);

    if (stats_out != nullptr) {
        stats_out->async_enabled = async_enabled;
        stats_out->pinned_enabled = pinned_enabled;
        stats_out->pinned_mode = pinned_enabled ? "pinned" : "standard";
    }

    if (!async_enabled) {
        return run_sync(store, producer, stats_out);
    }
    return run_async(store, options, producer, stats_out);
}

}  // namespace vector_db_v3::ingest
