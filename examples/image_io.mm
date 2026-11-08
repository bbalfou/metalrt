// macOS example-only decoding; no image codec dependency in the runtime.
#include "image_io.h"
#include <metalrt/buffer.h>
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#include <algorithm>
#include <cmath>
#include <memory>

namespace metalrt_examples {
namespace {
template <class T> struct CFReleaseObject {
    void operator()(T* object) const {
        if (object)
            CFRelease(object);
    }
};
template <class T> using CFPtr = std::unique_ptr<T, CFReleaseObject<T>>;
} // namespace
Image load_image(const std::string& path) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        CFPtr<CGImageSource> source{CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr)};
        if (!source)
            throw std::runtime_error("cannot open image: " + path);
        CFPtr<CGImage> image{CGImageSourceCreateImageAtIndex(source.get(), 0, nullptr)};
        if (!image)
            throw std::runtime_error("cannot decode image: " + path);
        Image result;
        result.width = CGImageGetWidth(image.get());
        result.height = CGImageGetHeight(image.get());
        const auto row = metalrt::checked_mul(result.width, size_t{4});
        result.rgba.resize(metalrt::checked_mul(row, result.height));
        CFPtr<CGColorSpace> color{CGColorSpaceCreateWithName(kCGColorSpaceSRGB)};
        CFPtr<CGContext> context{CGBitmapContextCreate(
            result.rgba.data(), result.width, result.height, 8, row, color.get(),
            kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big)};
        if (!context)
            throw std::runtime_error("cannot create image decoding context");
        // CGImage's raster order already places the top row first in this
        // bitmap context. Keep it unchanged so texture v=0 is the image top.
        CGContextDrawImage(context.get(), CGRectMake(0, 0, result.width, result.height),
                           image.get());
        for (size_t i = 3; i < result.rgba.size(); i += 4)
            result.rgba[i] = 255;
        return result;
    }
}
void write_rgb_png(const std::string& path, size_t width, size_t height, const float* rgb) {
    @autoreleasepool {
        std::vector<uint8_t> rgba(
            metalrt::checked_mul(metalrt::checked_mul(width, height), size_t{4}));
        for (size_t i = 0; i < width * height; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                const float value = rgb[3 * i + c];
                if (!std::isfinite(value))
                    throw std::runtime_error("non-finite prediction");
                rgba[4 * i + c] = uint8_t(std::clamp(value, 0.f, 1.f) * 255.f + .5f);
            }
            rgba[4 * i + 3] = 255;
        }
        CFPtr<CGColorSpace> color{CGColorSpaceCreateWithName(kCGColorSpaceSRGB)};
        CFPtr<CGContext> context{
            CGBitmapContextCreate(rgba.data(), width, height, 8, width * 4, color.get(),
                                  kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big)};
        if (!context)
            throw std::runtime_error("cannot create PNG context");
        CFPtr<CGImage> image{CGBitmapContextCreateImage(context.get())};
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        CFPtr<CGImageDestination> destination{CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr)};
        if (!image || !destination)
            throw std::runtime_error("cannot create PNG: " + path);
        CGImageDestinationAddImage(destination.get(), image.get(), nullptr);
        if (!CGImageDestinationFinalize(destination.get()))
            throw std::runtime_error("cannot write PNG: " + path);
    }
}
} // namespace metalrt_examples
