// One-time host upload into an optimized Private Metal texture.
#include "device.h"
namespace metalrt {
namespace detail {
namespace metal {
std::shared_ptr<TextureAllocation>
MetalDevice::upload_texture(TextureDescriptor desc, const void* pixels, size_t row_bytes) {
    @autoreleasepool {
        MTLPixelFormat format;
        switch (desc.format) {
        case PixelFormat::R8Unorm:
            format = MTLPixelFormatR8Unorm;
            break;
        case PixelFormat::RGBA8Unorm:
            format = MTLPixelFormatRGBA8Unorm;
            break;
        case PixelFormat::R32Float:
            format = MTLPixelFormatR32Float;
            break;
        case PixelFormat::RGBA32Float:
            format = MTLPixelFormatRGBA32Float;
            break;
        default:
            throw std::invalid_argument("unknown texture format");
        }
        auto result = std::make_shared<MetalTexture>();
        result->descriptor = desc;
        result->device_identity = this;
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                              width:desc.width
                                                                             height:desc.height
                                                                          mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
        result->texture = [device_ newTextureWithDescriptor:descriptor];
        if (!result->texture)
            metal_error("allocating Private image texture");

        // A padded staging buffer handles arbitrary user row strides, including
        // single-channel/odd-width images. 256-byte rows satisfy macOS blit alignment.
        const size_t row = checked_mul(desc.width, pixel_size(desc.format));
        const size_t pitch = checked_add(row, 255) & ~size_t{255};
        const size_t bytes = checked_mul(pitch, desc.height);
        if (bytes > device_.maxBufferLength)
            throw std::length_error("texture staging exceeds MTLDevice.maxBufferLength");
        id<MTLBuffer> staging = [device_ newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
        if (!staging)
            metal_error("allocating image upload staging buffer");
        for (size_t y = 0; y < desc.height; ++y)
            std::memcpy(static_cast<uint8_t*>(staging.contents) + y * pitch,
                        static_cast<const uint8_t*>(pixels) + y * row_bytes, row);
        id<MTLCommandQueue> queue = [device_ newCommandQueue];
        if (!queue)
            metal_error("creating image upload queue");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command)
            metal_error("creating image upload command buffer");
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!blit)
            metal_error("creating image upload blit encoder");
        [blit copyFromBuffer:staging
                   sourceOffset:0
              sourceBytesPerRow:pitch
            sourceBytesPerImage:bytes
                     sourceSize:MTLSizeMake(desc.width, desc.height, 1)
                      toTexture:result->texture
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError)
            metal_error("uploading image texture", command.error);
        // Upload completion makes this immutable texture safe on every runtime
        // queue, including borrowed command buffers. No later host waits occur.
        return result;
    }
}
std::shared_ptr<SamplerState> MetalDevice::make_sampler(SamplerDescriptor desc) {
    @autoreleasepool {
        auto result = std::make_shared<MetalSampler>();
        result->descriptor = desc;
        result->device_identity = this;
        auto* descriptor = [MTLSamplerDescriptor new];
        descriptor.normalizedCoordinates = YES;
        descriptor.minFilter = descriptor.magFilter = desc.filter == Filter::Linear
                                                          ? MTLSamplerMinMagFilterLinear
                                                          : MTLSamplerMinMagFilterNearest;
        descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
        descriptor.sAddressMode = desc.address_u == AddressMode::Repeat
                                      ? MTLSamplerAddressModeRepeat
                                      : MTLSamplerAddressModeClampToEdge;
        descriptor.tAddressMode = desc.address_v == AddressMode::Repeat
                                      ? MTLSamplerAddressModeRepeat
                                      : MTLSamplerAddressModeClampToEdge;
        result->sampler = [device_ newSamplerStateWithDescriptor:descriptor];
        if (!result->sampler)
            metal_error("creating image sampler");
        return result;
    }
}
} // namespace metal
} // namespace detail
} // namespace metalrt
