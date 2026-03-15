#include "vector_db_v3/codec/io.hpp"

#include <atomic>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace vector_db_v3::codec {

std::filesystem::path temp_path_for(const std::filesystem::path& target) {
    static std::atomic<std::uint64_t> counter{0};
    const auto n = counter.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream name;
    name << target.filename().string() << ".tmp." << static_cast<unsigned long long>(n);
    return target.parent_path() / name.str();
}

Status read_file_bytes(const std::filesystem::path& path, std::vector<std::uint8_t>* out) {
    if (out == nullptr) {
        return Status::Error("read_file_bytes: out is null");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status::Error("read_file_bytes: cannot open " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    if (sz < 0) {
        return Status::Error("read_file_bytes: tellg failed " + path.string());
    }
    out->assign(static_cast<std::size_t>(sz), 0U);
    in.seekg(0, std::ios::beg);
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(out->data()), sz);
    }
    if (!in && sz > 0) {
        return Status::Error("read_file_bytes: read failed " + path.string());
    }
    return Status::Ok();
}

Status validate_exact_size_multiple(
    std::size_t bytes,
    std::size_t record_size,
    const std::string& artifact_name) {
    if (record_size == 0U) {
        return Status::Error("record_size is zero for " + artifact_name);
    }
    if ((bytes % record_size) != 0U) {
        return Status::Error(
            artifact_name + ": byte size " + std::to_string(bytes) +
            " is not a multiple of record size " + std::to_string(record_size));
    }
    return Status::Ok();
}

namespace {

Status sync_file(const fs::path& p) {
#ifdef _WIN32
    const int fd = _open(p.string().c_str(), _O_RDONLY | _O_BINARY);
    if (fd < 0) {
        return Status::Error("sync_file: open failed for " + p.string());
    }
    if (_commit(fd) != 0) {
        _close(fd);
        return Status::Error("sync_file: _commit failed for " + p.string());
    }
    _close(fd);
#else
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status::Error("sync_file: open failed for " + p.string());
    }
    if (::fsync(fd) != 0) {
        ::close(fd);
        return Status::Error("sync_file: fsync failed for " + p.string());
    }
    ::close(fd);
#endif
    return Status::Ok();
}

Status replace_atomic(const fs::path& temp, const fs::path& target) {
#ifdef _WIN32
    const std::wstring temp_w = temp.wstring();
    const std::wstring target_w = target.wstring();
    if (fs::exists(target)) {
        if (!::ReplaceFileW(
                target_w.c_str(),
                temp_w.c_str(),
                nullptr,
                REPLACEFILE_IGNORE_MERGE_ERRORS,
                nullptr,
                nullptr)) {
            return Status::Error("replace_atomic: ReplaceFileW failed for " + target.string());
        }
    } else {
        if (!::MoveFileExW(
                temp_w.c_str(),
                target_w.c_str(),
                MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH)) {
            return Status::Error("replace_atomic: MoveFileExW failed for " + target.string());
        }
    }
#else
    std::error_code ec;
    fs::rename(temp, target, ec);
    if (ec) {
        return Status::Error("replace_atomic: rename failed for " + target.string() + ": " + ec.message());
    }
#endif
    return Status::Ok();
}

}  // namespace

Status write_atomic_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    try {
        fs::create_directories(path.parent_path());
        const fs::path temp = temp_path_for(path);
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return Status::Error("write_atomic_bytes: cannot open temp file " + temp.string());
            }
            if (!bytes.empty()) {
                out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            }
            out.flush();
            if (!out) {
                return Status::Error("write_atomic_bytes: write/flush failed for " + temp.string());
            }
        }
        const Status sync = sync_file(temp);
        if (!sync.ok) {
            std::error_code ignore;
            fs::remove(temp, ignore);
            return sync;
        }
        const Status rep = replace_atomic(temp, path);
        if (!rep.ok) {
            std::error_code ignore;
            fs::remove(temp, ignore);
            return rep;
        }
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::Error(std::string("write_atomic_bytes: exception: ") + e.what());
    } catch (...) {
        return Status::Error("write_atomic_bytes: unknown exception");
    }
}

}  // namespace vector_db_v3::codec
