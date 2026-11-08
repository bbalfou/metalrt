// Persistent Metal binary archives; failures fall back to compilation.
#include "device.h"
namespace metalrt {
namespace detail {
namespace metal {
NativeArchive MetalDevice::archive_locked(const std::string& source, const std::string& entry,
                                          const std::string& directory,
                                          const KernelCompileOptions& options) {
    NativeArchive archive;
    if (directory.empty())
        return archive;
    archive.requested = true;
    archive.directory = directory;
    try {
        // Length-prefixed fields avoid ambiguity; OS build and registry ID
        // prevent accidental reuse across driver updates or devices. The
        // schema names every compiler/pipeline option that affects code.
        std::string identity;
        const auto append = [&](const std::string& field) {
            identity += std::to_string(field.size()) + ":" + field;
        };
        append("metalrt-binary-archive-v1;msl=platform-default;direct=1");
        append(options.fast_math ? "fast-math=1" : "fast-math=0");
        append(capabilities_.indirect_compute ? "indirect=1" : "indirect=0");
        append(std::to_string(device_.registryID));
        append(capabilities_.name);
        append(ns_string(NSProcessInfo.processInfo.operatingSystemVersionString));
        append(entry);
        append(source);
        const auto root = std::filesystem::absolute(directory).lexically_normal();
        std::filesystem::create_directories(root);
        archive.path = root / ("metalrt-" + kernel_cache_digest(identity) + ".metallib");
        MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
        NSError* error = nil;
        if (std::filesystem::is_regular_file(archive.path)) {
            NSString* path = [[NSString alloc] initWithBytes:archive.path.native().data()
                                                      length:archive.path.native().size()
                                                    encoding:NSUTF8StringEncoding];
            if (!path)
                throw std::invalid_argument("Metal archive path is not valid UTF-8");
            descriptor.url = [NSURL fileURLWithPath:path];
            archive.object = [device_ newBinaryArchiveWithDescriptor:descriptor error:&error];
            archive.loaded = archive.object != nil;
            if (!archive.loaded)
                ++archive_errors;
        }
        if (!archive.object) {
            descriptor.url = nil;
            error = nil;
            archive.object = [device_ newBinaryArchiveWithDescriptor:descriptor error:&error];
            if (!archive.object)
                ++archive_errors;
        }
    } catch (const std::exception&) {
        // A cache is an optimization. An unreadable/unwritable directory
        // must never prevent execution of a shader that compiles normally.
        ++archive_errors;
        archive.object = nil;
    }
    return archive;
}

id<MTLComputePipelineState>
MetalDevice::pipeline_locked(MTLComputePipelineDescriptor* descriptor, NativeArchive& archive,
                             NSError** error, MTLComputePipelineReflection** reflection) {
    if (archive.object)
        descriptor.binaryArchives = @[ archive.object ];
    else
        descriptor.binaryArchives = nil;
    if (archive.loaded) {
        auto pipeline = [device_
            newComputePipelineStateWithDescriptor:descriptor
                                          options:MTLPipelineOptionFailOnBinaryArchiveMiss |
                                                  MTLPipelineOptionArgumentInfo
                                       reflection:reflection
                                            error:error];
        if (pipeline) {
            ++archive_hits;
            return pipeline;
        }
        // Missing/stale GPU slices are recovered by compiling the exact
        // same descriptor, never by loading a differently keyed pipeline.
        if (error)
            *error = nil;
        if (reflection)
            *reflection = nil;
    }
    if (archive.requested)
        ++archive_misses;
    if (archive.object) {
        NSError* archive_error = nil;
        if ([archive.object addComputePipelineFunctionsWithDescriptor:descriptor
                                                                error:&archive_error])
            archive.dirty = true;
        else
            ++archive_errors;
    }
    return [device_ newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionArgumentInfo
                                               reflection:reflection
                                                    error:error];
}

void MetalDevice::archive_kernel_locked(const MetalKernel& kernel, NativeArchive& archive) {
    if (!archive.object || archive.loaded)
        return;
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = kernel.function;
    for (unsigned variant = 0; variant < (kernel.indirect ? 2u : 1u); ++variant) {
        descriptor.supportIndirectCommandBuffers = variant != 0;
        NSError* error = nil;
        if ([archive.object addComputePipelineFunctionsWithDescriptor:descriptor error:&error])
            archive.dirty = true;
        else
            ++archive_errors;
    }
}

bool MetalDevice::store_archive_locked(NativeArchive& archive) {
    if (!archive.object)
        return false;
    if (!archive.dirty)
        return archive.loaded;
    std::filesystem::path temporary;
    try {
        // Each writer has a unique temporary file in the same directory.
        // Rename publishes a whole archive atomically even across processes.
        temporary = archive.path;
        temporary += "." + ns_string(NSUUID.UUID.UUIDString) + ".tmp";
        NSString* path = [[NSString alloc] initWithBytes:temporary.native().data()
                                                  length:temporary.native().size()
                                                encoding:NSUTF8StringEncoding];
        if (!path)
            throw std::invalid_argument("Metal archive path is not valid UTF-8");
        NSError* error = nil;
        if (![archive.object serializeToURL:[NSURL fileURLWithPath:path] error:&error])
            throw std::runtime_error("cannot serialize Metal binary archive: " +
                                     ns_string(error.localizedDescription));
        std::filesystem::rename(temporary, archive.path);
        ++archive_writes;
        return true;
    } catch (const std::exception&) {
        ++archive_errors;
        if (!temporary.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        }
        return false;
    }
}

} // namespace metal
} // namespace detail
} // namespace metalrt
