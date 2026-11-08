#include "runtime_internal.h"

namespace metalrt {
size_t pixel_size(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8Unorm:
        return 1;
    case PixelFormat::RGBA8Unorm:
        return 4;
    case PixelFormat::R32Float:
        return 4;
    case PixelFormat::RGBA32Float:
        return 16;
    }
    throw std::invalid_argument("unknown texture pixel format");
}
Texture2D Device::upload_texture(TextureDescriptor desc, const void* pixels, size_t size,
                                 size_t row_bytes) const {
    require(static_cast<bool>(*this), "empty device");
    require(desc.width && desc.height && desc.width <= 16384 && desc.height <= 16384,
            "texture dimensions must be between 1 and 16384");
    const auto row = checked_mul(desc.width, pixel_size(desc.format));
    if (!row_bytes)
        row_bytes = row;
    require(row_bytes >= row, "texture row stride is smaller than its pixels");
    const auto required = checked_add(checked_mul(desc.height - 1, row_bytes), row);
    require(pixels && size >= required, "texture source pixels are null or truncated");
    auto result = impl_->upload_texture(desc, pixels, row_bytes);
    if (!result)
        throw std::runtime_error("backend texture upload failed");
    result->device_owner = impl_;
    ++impl_->texture_allocations;
    impl_->texture_uploaded_bytes += checked_mul(row, desc.height);
    return Texture2D{std::move(result)};
}
Sampler Device::sampler(SamplerDescriptor desc) const {
    require(static_cast<bool>(*this), "empty device");
    require(desc.filter == Filter::Nearest || desc.filter == Filter::Linear, "unknown filter");
    const auto valid_address = [](AddressMode mode) {
        return mode == AddressMode::ClampToEdge || mode == AddressMode::Repeat;
    };
    require(valid_address(desc.address_u) && valid_address(desc.address_v), "unknown address mode");
    auto result = impl_->make_sampler(desc);
    if (!result)
        throw std::runtime_error("backend sampler allocation failed");
    result->device_owner = impl_;
    ++impl_->sampler_allocations;
    return Sampler{std::move(result)};
}
TextureDescriptor Texture2D::descriptor() const {
    if (!impl_)
        throw std::logic_error("empty texture");
    return impl_->descriptor;
}
Device Texture2D::device() const {
    if (!impl_ || !impl_->device_owner)
        throw std::logic_error("empty texture has no device");
    return Device{impl_->device_owner};
}
SamplerDescriptor Sampler::descriptor() const {
    if (!impl_)
        throw std::logic_error("empty sampler");
    return impl_->descriptor;
}
Device Sampler::device() const {
    if (!impl_ || !impl_->device_owner)
        throw std::logic_error("empty sampler has no device");
    return Device{impl_->device_owner};
}
std::array<float, 4> Texture2D::reference_sample(const Sampler& sampler, float u, float v) const {
    const auto desc = descriptor();
    const auto sampling = sampler.descriptor();
    require(impl_->device_identity == sampler.implementation()->device_identity,
            "texture and sampler belong to different devices");
    require(std::isfinite(u) && std::isfinite(v), "texture coordinates must be finite");
    const auto* pixels = impl_->reference_pixels();
    const auto coordinate = [](float value, AddressMode mode) {
        return mode == AddressMode::Repeat ? value - std::floor(value)
                                           : std::clamp(value, 0.0f, 1.0f);
    };
    const float x = coordinate(u, sampling.address_u) * float(desc.width);
    const float y = coordinate(v, sampling.address_v) * float(desc.height);
    const auto address = [](int index, size_t dimension, AddressMode mode) {
        const auto n = static_cast<int>(dimension);
        return mode == AddressMode::Repeat ? (index % n + n) % n : std::clamp(index, 0, n - 1);
    };
    const auto texel = [&](int tx, int ty) {
        tx = address(tx, desc.width, sampling.address_u);
        ty = address(ty, desc.height, sampling.address_v);
        const auto* p = pixels + (size_t(ty) * desc.width + size_t(tx)) * pixel_size(desc.format);
        std::array<float, 4> result{0, 0, 0, 1};
        const size_t channels =
            desc.format == PixelFormat::R8Unorm || desc.format == PixelFormat::R32Float ? 1 : 4;
        for (size_t c = 0; c < channels; ++c) {
            if (desc.format == PixelFormat::R8Unorm || desc.format == PixelFormat::RGBA8Unorm)
                result[c] = float(p[c]) / 255.0f;
            else
                std::memcpy(&result[c], p + c * sizeof(float), sizeof(float));
        }
        return result;
    };
    if (sampling.filter == Filter::Nearest)
        return texel(int(std::floor(x)), int(std::floor(y)));
    const float px = x - .5f, py = y - .5f;
    const int ix = int(std::floor(px)), iy = int(std::floor(py));
    const float fx = px - float(ix), fy = py - float(iy);
    const auto a = texel(ix, iy), b = texel(ix + 1, iy);
    const auto c = texel(ix, iy + 1), d = texel(ix + 1, iy + 1);
    std::array<float, 4> result;
    for (size_t k = 0; k < 4; ++k)
        result[k] = (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy;
    return result;
}
Argument Argument::texture(uint32_t index, Texture2D image) {
    require(static_cast<bool>(image), "texture bindings must have an allocation");
    Argument result;
    result.index = index;
    result.kind = Kind::Texture;
    result.image = std::move(image);
    return result;
}
Argument Argument::sampler(uint32_t index, Sampler sampling) {
    require(static_cast<bool>(sampling), "sampler bindings must have a state");
    Argument result;
    result.index = index;
    result.kind = Kind::Sampler;
    result.sampling = std::move(sampling);
    return result;
}
const Texture2D& ComputeCommand::texture(uint32_t index) const {
    for (const auto& a : arguments)
        if (a.index == index && a.kind == Argument::Kind::Texture)
            return a.image;
    throw std::out_of_range("texture argument index is not bound");
}
const Sampler& ComputeCommand::sampler(uint32_t index) const {
    for (const auto& a : arguments)
        if (a.index == index && a.kind == Argument::Kind::Sampler)
            return a.sampling;
    throw std::out_of_range("sampler argument index is not bound");
}
} // namespace metalrt
