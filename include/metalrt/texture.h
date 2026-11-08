#pragma once
// Immutable, single-level 2D images and normalized-coordinate sampling.
#include <metalrt/types.h>
#include <array>

namespace metalrt {
enum class PixelFormat { R8Unorm, RGBA8Unorm, R32Float, RGBA32Float };
enum class Filter { Nearest, Linear };
enum class AddressMode { ClampToEdge, Repeat };
struct TextureDescriptor {
    size_t width = 0, height = 0;
    PixelFormat format = PixelFormat::RGBA8Unorm;
};
struct SamplerDescriptor {
    Filter filter = Filter::Linear;
    AddressMode address_u = AddressMode::ClampToEdge;
    AddressMode address_v = AddressMode::ClampToEdge;
};
size_t pixel_size(PixelFormat format);

class Sampler {
  public:
    Sampler() = default;
    explicit Sampler(std::shared_ptr<detail::SamplerState> impl) : impl_(std::move(impl)) {}
    SamplerDescriptor descriptor() const;
    Device device() const;
    explicit operator bool() const { return static_cast<bool>(impl_); }
    const std::shared_ptr<detail::SamplerState>& implementation() const { return impl_; }
    bool operator==(const Sampler& other) const { return impl_ == other.impl_; }

  private:
    std::shared_ptr<detail::SamplerState> impl_;
};

class Texture2D {
  public:
    Texture2D() = default;
    explicit Texture2D(std::shared_ptr<detail::TextureAllocation> impl) : impl_(std::move(impl)) {}
    TextureDescriptor descriptor() const;
    Device device() const;
    // CPU reference callbacks only. Texel centers are ((x+.5)/width, (y+.5)/height).
    // R formats return (r,0,0,1). Formats are linear values; no implicit sRGB conversion.
    std::array<float, 4> reference_sample(const Sampler&, float u, float v) const;
    explicit operator bool() const { return static_cast<bool>(impl_); }
    const std::shared_ptr<detail::TextureAllocation>& implementation() const { return impl_; }
    bool operator==(const Texture2D& other) const { return impl_ == other.impl_; }

  private:
    std::shared_ptr<detail::TextureAllocation> impl_;
};
} // namespace metalrt
