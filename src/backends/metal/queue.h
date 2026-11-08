#pragma once
// Submission policy and encoder orchestration; plans retain all replay resources.
#include "metal_internal.h"
namespace metalrt {
namespace detail {
namespace metal {
class MetalQueue final : public QueueBackend {
  public:
    MetalQueue(id<MTLDevice> device, DeviceBackend* owner, bool indirect,
               std::shared_ptr<TransferPipelines> transfers,
               std::function<void*()> command_buffer = {})
        : device_(device), identity_(owner), metrics_(owner), indirect_supported_(indirect),
          transfers_(std::move(transfers)), command_buffer_(std::move(command_buffer)) {
        if (!command_buffer_) {
            queue_ = [device_ newCommandQueue];
            if (!queue_)
                metal_error("newCommandQueue");
            queue_.label = @"metalrt stream";
        }
    }
    bool externally_managed() const noexcept override { return static_cast<bool>(command_buffer_); }
    SubmitResult submit(const std::vector<Command>& commands, DispatchPolicy policy,
                        std::shared_ptr<Plan> cached, const SubmitOptions& options) override;

  private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_ = nil;
    const void* identity_;
    DeviceBackend* metrics_;
    bool indirect_supported_;
    std::shared_ptr<TransferPipelines> transfers_;
    std::function<void*()> command_buffer_;
    std::mutex mutex_;
    id<MTLFence> fence_ = nil;
    id<MTLCommandQueue> fence_queue_ = nil;
    mutable bool fence_signaled_ = false;

    template <class Encoder> void begin_encoder(Encoder encoder) const {
        if (fence_ && fence_signaled_)
            [encoder waitForFence:fence_];
    }
    template <class Encoder> void end_encoder(Encoder encoder) const {
        if (fence_) {
            [encoder updateFence:fence_];
            fence_signaled_ = true;
        }
        [encoder endEncoding];
    }

    id<MTLCommandBuffer> new_command_buffer();
    static void retain_until_complete(id<MTLCommandBuffer> cb,
                                      const std::shared_ptr<SubmissionUse>& use);
    void encode_waits(id<MTLCommandBuffer> cb, const SubmitOptions& options) const;
    void encode_signal(id<MTLCommandBuffer> cb, const SubmitOptions& options) const;
    void record_allocation(size_t bytes) const;
    bool validate(const std::vector<Command>& commands, DispatchPolicy policy) const;
    std::shared_ptr<PreparedSequence>
    prepare(const std::vector<Command>& commands,
            const std::shared_ptr<PreparedSequence>& previous = {}) const;
    static void encode_compute(id<MTLComputeCommandEncoder> encoder, const ComputeCommand& command,
                               const PreparedCommand& prepared, id<MTLBuffer> groups = nil,
                               size_t offset = 0);
    void encode_transfer(id<MTLCommandBuffer> cb, const Command& command) const;
    std::shared_ptr<KernelSample> make_sample(id<MTLCommandBuffer>, const ComputeCommand&, size_t,
                                              const GpuTimingOptions&) const;
    id<MTLComputeCommandEncoder> sampled_encoder(id<MTLCommandBuffer>,
                                                 const std::shared_ptr<KernelSample>&) const;
    void encode_profiled(id<MTLCommandBuffer>, const PreparedSequence&, size_t, size_t,
                         const std::shared_ptr<DispatchScratch>&, MetalCompletion&,
                         const GpuTimingOptions&) const;
    void encode_dynamic(id<MTLCommandBuffer> cb, const PreparedSequence& sequence, size_t index,
                        const std::shared_ptr<DispatchScratch>& scratch,
                        const std::shared_ptr<KernelSample>& sample = {}) const;
    void encode_direct(id<MTLCommandBuffer> cb, const PreparedSequence& sequence, size_t begin,
                       size_t end, const std::shared_ptr<DispatchScratch>& scratch) const;
    static std::vector<IndirectSegment> segment_layout(const PreparedSequence& sequence);
    static bool same_segment(const IndirectStorage& previous, const PreparedSequence& next,
                             const IndirectSegment& segment);
    void allocate_icb(IndirectStorage& storage) const;
    static void encode_icb(const IndirectStorage& storage);
    static std::shared_ptr<IndirectStorage> make_storage(const PreparedSequence& sequence,
                                                         const IndirectSegment& segment);
    std::shared_ptr<DispatchScratch> make_scratch(size_t bytes) const;
    void ensure_scratch_capacity(MetalPlan& plan) const;
    std::shared_ptr<DispatchScratch> acquire_scratch(MetalPlan& plan) const;
    std::shared_ptr<MetalPlan> prepare_plan(const std::vector<Command>& commands,
                                            DispatchPolicy policy, std::shared_ptr<Plan> cached,
                                            Statistics& statistics,
                                            std::unique_lock<std::mutex>& result_lock) const;
    void encode_indirect(id<MTLCommandBuffer> cb, const MetalPlan& plan,
                         const std::shared_ptr<DispatchScratch>& scratch) const;
};
} // namespace metal
} // namespace detail
} // namespace metalrt
