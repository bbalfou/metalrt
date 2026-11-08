#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace metalrt_examples {
struct Image {
    size_t width = 0, height = 0;
    // Top row first, RGBA8 in the image's sRGB display space, opaque alpha.
    std::vector<uint8_t> rgba;
};
Image load_image(const std::string& path);
void write_rgb_png(const std::string& path, size_t width, size_t height, const float* rgb);
inline uint32_t positive_count(const std::string& value) {
    size_t end = 0;
    const auto count = std::stoull(value, &end);
    if (value.empty() || value[0] == '-' || end != value.size() || !count || count > UINT32_MAX)
        throw std::invalid_argument("expected a positive uint32 count");
    return static_cast<uint32_t>(count);
}
} // namespace metalrt_examples
