#pragma once

#include <filesystem>

namespace vector_db_v3::paths {

inline std::filesystem::path manifest(const std::filesystem::path& data_dir) {
    return data_dir / "manifest.json";
}

inline std::filesystem::path wal(const std::filesystem::path& data_dir) {
    return data_dir / "wal.log";
}

inline std::filesystem::path segments_dir(const std::filesystem::path& data_dir) {
    return data_dir / "segments";
}

inline std::filesystem::path clusters_current_dir(const std::filesystem::path& data_dir) {
    return data_dir / "clusters" / "current";
}

}  // namespace vector_db_v3::paths
