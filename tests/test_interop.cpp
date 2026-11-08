#include <metalrt/interop.h>
#include <metalrt/detail/backend.h>
#include <iostream>

namespace {
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}

// This probe models only external execution ownership. It remains explicitly a
// Reference test and cannot validate Metal encoding or allocator behavior.
struct ProbeCompletion final : metalrt::detail::Completion {
    size_t waits = 0;
    bool completed = false, failed = false;
    void wait() override {
        ++waits;
        if (!completed)
            throw metalrt::UncommittedCommandBufferError("execution owner has not submitted work");
        if (failed)
            throw std::runtime_error("controlled external execution failure");
    }
    bool ready() const override { return completed; }
};
struct ProbeQueue final : metalrt::detail::QueueBackend {
    std::function<void*()> provider;
    std::shared_ptr<ProbeCompletion> completion;
    bool externally_managed() const noexcept override { return true; }
    metalrt::detail::SubmitResult submit(const std::vector<metalrt::Command>&,
                                         metalrt::DispatchPolicy,
                                         std::shared_ptr<metalrt::detail::Plan>,
                                         const metalrt::detail::SubmitOptions&) override {
        check(provider() != nullptr, "provider must supply an execution identity");
        metalrt::detail::SubmitResult result;
        result.completion = completion;
        return result;
    }
};
struct ProbeAllocation final : metalrt::detail::BufferAllocation {
    std::vector<uint8_t> memory;
    std::shared_ptr<void> external_owner;
    void* contents() const override { return const_cast<uint8_t*>(memory.data()); }
};
struct ProbeDevice final : metalrt::detail::DeviceBackend {
    std::shared_ptr<ProbeCompletion> completion = std::make_shared<ProbeCompletion>();
    metalrt::Capabilities capabilities() const override {
        metalrt::Capabilities result;
        result.name = "external ownership reference probe";
        return result;
    }
    std::shared_ptr<metalrt::detail::BufferAllocation>
    allocate(size_t bytes, metalrt::StorageMode storage) override {
        auto result = std::make_shared<ProbeAllocation>();
        result->memory.resize(bytes);
        result->bytes = bytes;
        result->storage = storage;
        result->device_identity = this;
        result->device_owner = shared_from_this();
        return result;
    }
    std::shared_ptr<metalrt::detail::BufferAllocation>
    import_metal_buffer(void*, std::shared_ptr<void> owner) override {
        auto result =
            std::static_pointer_cast<ProbeAllocation>(allocate(32, metalrt::StorageMode::Shared));
        result->external_owner = std::move(owner);
        return result;
    }
    std::shared_ptr<metalrt::detail::KernelState>
    compile(const std::string&, const std::string& entry, metalrt::CpuKernel) override {
        auto result = std::make_shared<metalrt::detail::KernelState>();
        result->device_identity = this;
        result->entry = entry;
        return result;
    }
    std::unique_ptr<metalrt::detail::QueueBackend> make_queue() override {
        throw std::logic_error("borrowed construction must not allocate an ordinary queue");
    }
    std::unique_ptr<metalrt::detail::QueueBackend>
    make_external_metal_queue(std::function<void*()> provider) override {
        auto result = std::make_unique<ProbeQueue>();
        result->provider = std::move(provider);
        result->completion = completion;
        return result;
    }
    std::shared_ptr<metalrt::detail::EventState> make_event() override {
        throw std::logic_error("independent borrowed stream must not need an ordering event");
    }
};

void external_ownership() {
    auto backend = std::make_shared<ProbeDevice>();
    metalrt::Device device{backend};
    size_t provider_calls = 0;
    auto stream = metalrt::borrow_metal_stream(device, [&]() -> void* {
        ++provider_calls;
        return backend.get();
    });
    check(stream->ordering() == metalrt::StreamOrdering::Independent,
          "borrowed queues must keep external ordering");
    auto retained = stream->retain();
    stream->dispatch(device.kernel("", "probe"), {1, 1, 1}, {1, 1, 1}, {});
    auto ticket = stream->last_ticket();
    check(provider_calls == 1 && !ticket.ready(),
          "submission must use its provider without reporting completion");
    stream.reset();
    retained.reset();
    check(backend->completion->waits == 0,
          "borrowed owner and final alias destruction must not wait");
    backend->completion->completed = true;
    ticket.wait();
    check(backend->completion->waits == 1,
          "explicit ticket wait must observe the owner's completion");
    backend->completion->failed = true;
    rejects([&] { ticket.wait(); }, "external asynchronous errors must remain observable");
}

void wait_retry_and_dependency() {
    auto backend = std::make_shared<ProbeDevice>();
    metalrt::Device device{backend};
    auto stream = metalrt::borrow_metal_stream(device, [&]() -> void* { return backend.get(); });
    stream->dispatch(device.kernel("", "probe"), {1, 1, 1}, {1, 1, 1}, {});
    rejects([&] { stream->synchronize(); },
            "uncommitted external work must reject synchronization");
    check(backend->completion->waits == 1, "first synchronization must observe pending work");
    backend->completion->completed = true;
    stream->synchronize();
    check(backend->completion->waits == 2,
          "failed synchronization must retain work for retry after commit");

    struct Dependent final : metalrt::detail::Completion {
        metalrt::Ticket producer;
        size_t waits = 0;
        bool append_dependencies(std::vector<metalrt::Ticket>& out, bool) override {
            out.push_back(producer);
            return true;
        }
        void wait() override { ++waits; }
        bool ready() const override { return true; }
    };
    backend->completion->completed = false;
    auto consumer = std::make_shared<Dependent>();
    consumer->producer = metalrt::Ticket{backend->completion};
    auto dependent = metalrt::Ticket{consumer};
    rejects([&] { dependent.wait(); },
            "a committed consumer must reject waiting on an uncommitted producer");
    check(consumer->waits == 0,
          "must not block on a dependent GPU wait after rejecting its uncommitted producer");
    backend->completion->completed = true;
    dependent.wait();
    check(consumer->waits == 1, "dependency wait must remain retryable after its producer commits");
}

void unsupported_and_invalid_inputs() {
    auto reference = metalrt::Device::reference();
    int sentinel = 0;
    rejects([&] { metalrt::import_metal_device(nullptr); }, "null native device must fail");
    rejects([&] { metalrt::native_metal_device(reference); },
            "Reference must not fabricate a Metal device");
    rejects([&] { metalrt::import_metal_buffer(reference, &sentinel); },
            "Reference must not pretend to import Metal storage");
    rejects([&] { metalrt::import_metal_buffer(reference, nullptr); },
            "null native buffer must fail");
    rejects(
        [&] {
            metalrt::import_metal_buffer(reference, &sentinel, {},
                                         metalrt::MetalBufferSynchronization::Explicit);
        },
        "Reference must not pretend to implement untracked Metal synchronization");
    rejects(
        [&] {
            metalrt::import_metal_buffer(reference, &sentinel, {},
                                         static_cast<metalrt::MetalBufferSynchronization>(99));
        },
        "unknown import synchronization mode must fail");
    rejects([&] { metalrt::borrow_metal_stream(reference, [&]() -> void* { return &sentinel; }); },
            "Reference must not silently execute a borrowed Metal stream");
    rejects([&] { metalrt::borrow_metal_stream(reference, {}); }, "empty provider must fail");
    rejects([&] { metalrt::native_metal_device({}); }, "empty runtime device must fail");
}

void fresh_host_access() {
    auto backend = std::make_shared<ProbeDevice>();
    metalrt::Device device{backend};
    auto stream = metalrt::borrow_metal_stream(device, [&]() -> void* { return backend.get(); });
    auto kernel = device.kernel("", "probe");
    stream->dispatch(kernel, {1, 1, 1}, {1, 1, 1}, {});

    auto fresh = device.allocate(32);
    metalrt::synchronize_buffer(fresh.view(8, 8));
    check(backend->completion->waits == 0,
          "fresh storage must not wait for unrelated uncommitted work");
    stream->dispatch(kernel, {1, 1, 1}, {1, 1, 1},
                     {metalrt::Argument::buffer(0, fresh.view(), metalrt::Access::Read)});
    rejects([&] { metalrt::synchronize_buffer(fresh.view()); },
            "GPU-read storage must no longer take the fresh host path");

    auto recorded = device.allocate(32);
    {
        auto batch = stream->batch();
        metalrt::synchronize_buffer(recorded.view());
        stream->fill(recorded.view(), 0);
        rejects([&] { metalrt::synchronize_buffer(recorded.view()); },
                "recorded GPU storage must preserve synchronization rejection");
        batch.cancel();
    }
    rejects([&] { metalrt::synchronize_buffer(recorded.view()); },
            "cancellation must not turn previously bound storage into fresh storage");

    int sentinel = 0;
    auto imported = metalrt::import_metal_buffer(device, &sentinel);
    rejects([&] { metalrt::synchronize_buffer(imported.view()); },
            "imported storage must never take the fresh runtime allocation shortcut");
    const auto waits = backend->completion->waits;
    backend->completion->completed = true;
    metalrt::synchronize_buffer(fresh.view());
    metalrt::synchronize_buffer(imported.view());
    check(backend->completion->waits > waits,
          "used/imported host access must observe committed completion");
}

void shared_pointer_identity() {
    struct Alias final : metalrt::detail::BufferAllocation {
        std::shared_ptr<std::vector<uint8_t>> memory;
        size_t start = 0;
        void* contents() const override { return memory->data() + start; }
    };
    auto memory = std::make_shared<std::vector<uint8_t>>(128);
    auto make_alias = [&](size_t offset, size_t bytes) {
        auto alias = std::make_shared<Alias>();
        alias->memory = memory;
        alias->start = offset;
        alias->bytes = bytes;
        alias->device_owner = metalrt::Device::reference().implementation();
        alias->device_identity = alias->device_owner.get();
        return alias;
    };
    auto first = make_alias(0, 64);
    metalrt::detail::register_allocation(first);
    metalrt::detail::register_allocation(first);
    check(metalrt::resolve_pointer(memory->data() + 16, 16).buffer.implementation() == first,
          "duplicate registration must keep the exact allocation identity");
    auto duplicate = make_alias(0, 64), inside = make_alias(16, 32), enclosing = make_alias(0, 128);
    rejects([&] { metalrt::detail::register_allocation(duplicate); },
            "distinct exact aliases must not overwrite pointer identity");
    rejects([&] { metalrt::detail::register_allocation(inside); },
            "interior aliases must not overwrite pointer identity");
    rejects([&] { metalrt::detail::register_allocation(enclosing); },
            "enclosing aliases must not hide existing pointers");
    auto adjacent = make_alias(64, 64);
    metalrt::detail::register_allocation(adjacent);
    check(metalrt::resolve_pointer(memory->data() + 80, 16).buffer.implementation() == adjacent,
          "adjacent nonoverlapping Shared ranges must remain distinct");
    first.reset();
    metalrt::detail::register_allocation(duplicate);
    check(metalrt::resolve_pointer(memory->data() + 16, 16).buffer.implementation() == duplicate,
          "expired registrations must not block reuse by a new allocation");
}
} // namespace

int main() {
    try {
        external_ownership();
        wait_retry_and_dependency();
        unsupported_and_invalid_inputs();
        fresh_host_access();
        shared_pointer_identity();
        std::cout
            << "reference interop ownership/validation checks passed (no native Metal execution)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
