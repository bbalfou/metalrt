#pragma once
#include <metalrt/runtime.h>

// Framework-neutral interoperability with an existing Metal execution owner.
// Native handles are borrowed Objective-C object pointers, passed with
// (__bridge void*) from Objective-C++. No Apple or tensor-framework headers are
// required by this interface. These operations throw on a non-Metal build;
// they never select the Reference backend implicitly.
namespace metalrt {

// Waiting cannot progress until the external owner commits a dependency.
// The submitted work remains valid and the same wait may be retried afterward.
class UncommittedCommandBufferError : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};

// Retain the supplied id<MTLDevice>. Importing a device already used by this
// process returns the same runtime identity (including system_default()).
Device import_metal_device(void* native_device);
// The returned id<MTLDevice> is borrowed and remains valid while device is live.
void* native_metal_device(Device device);

// Retain a whole tracked id<MTLBuffer>, without copying or changing its storage mode.
// Use Buffer::view(byte_offset, byte_count) for tensor storage offsets. Shared
// imports are registered for resolve_pointer(); Private imports remain usable
// through BufferViews and GPU copies, but never expose a host data pointer.
// Managed storage requires an explicit CPU/GPU coherence contract and is
// rejected. This overload also rejects untracked resources; the explicit
// synchronization overload below is available to native execution owners.
//
// A native retain alone does not protect an allocation in an external caching
// allocator. Supply a lease that prevents that allocator from reusing the
// storage. Commands and graphs retain it with the imported Buffer. Repeated
// imports of one live native buffer share allocation identity, and leases with
// the same external_owner.get() are deduplicated. An aliasing shared_ptr whose
// get() is the external storage's stable identity is suitable for this purpose.
Buffer import_metal_buffer(Device device, void* native_buffer,
                           std::shared_ptr<void> external_owner = {});

enum class MetalBufferSynchronization {
    Tracked,
    // The caller orders external producers and consumers with GPU fences or
    // events. Runtime queues then fence their own compute/blit/indirect passes,
    // including successive submissions, and retain those fences until native
    // completion. A borrowed runtime stream must use one native command queue
    // and the owner must commit producer buffers before dependent consumers.
    Explicit,
};

Buffer import_metal_buffer(Device device, void* native_buffer, std::shared_ptr<void> external_owner,
                           MetalBufferSynchronization synchronization);

// Create a persistent runtime stream whose submissions are encoded into the
// caller's current id<MTLCommandBuffer>. The provider is called synchronously
// for each actual submission, never during graph-only preparation. The caller
// must serialize the entire runtime operation on its own encoding queue, end
// any active encoder before submission, and return a buffer from this device.
// The returned buffer must be uncommitted. When using an MPSCommandBuffer,
// return its rootCommandBuffer so later commitAndContinue calls cannot change
// the completion object observed by the runtime.
//
// The external owner alone commits and schedules the command buffer. All
// dispatch policies use the normal native preparation/encoding machinery;
// physical command-buffer grouping remains controlled by that external owner.
// No CPU wait occurs when a borrowed stream (or its retained alias) is released.
// Encoded work keeps its kernels, resources, external allocator leases and
// scratch allocations alive through native completion or buffer cancellation.
// Explicit waits before the owner commits report UncommittedCommandBufferError rather than
// blocking the thread that must perform that commit. After commit, Ticket and
// Stream waits retain their usual completion/error behavior. The owner remains
// responsible for reporting asynchronous errors if it never explicitly waits.
std::shared_ptr<Stream> borrow_metal_stream(Device device,
                                            std::function<void*()> current_command_buffer,
                                            DispatchPolicy policy = DispatchPolicy::Direct);

} // namespace metalrt
