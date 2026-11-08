// Ring leases serialize host access with tracked GPU use and retain slot generations.
// Failed submissions poison storage until explicitly recovered, preventing unsafe reuse.
#include <metalrt/ring.h>
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace metalrt {
namespace detail {
struct RingState {
    enum class Phase { None, Waiting, Cpu, Submitting };
    Device device;
    Buffer storage;
    size_t count = 0, bytes = 0, stride = 0, cursor = 0;
    std::vector<Ticket> tickets;
    mutable std::mutex mutex;
    std::condition_variable changed;
    Phase phase = Phase::None;
    std::thread::id owner;
    uint64_t generation = 0;
    bool maintenance = false;
    std::exception_ptr failure;
    RingStatistics stats;
    BufferView view(size_t slot) const { return storage.view(slot * stride, bytes); }
    void check_lease(size_t slot, uint64_t token, Phase expected) const {
        if (owner != std::this_thread::get_id())
            throw std::logic_error(
                "ring lease access and submission must stay on its acquiring thread");
        if (slot != cursor || token != generation || phase != expected)
            throw std::logic_error("ring lease no longer owns CPU access to its slot");
    }
};
struct RingPipelineState {
    Stream stream;
    RingBuffer ring;
    RingBindingMode mode;
    Buffer selector;
    std::vector<std::unique_ptr<Graph>> graphs;
    RingPipelineState(Device device, DispatchPolicy policy, size_t slots, size_t bytes,
                      RingBindingMode binding_mode)
        : stream(device, policy), ring(std::move(device), slots, bytes), mode(binding_mode) {}
};
} // namespace detail
namespace {
void require_state(bool value, const char* message) {
    if (!value)
        throw std::logic_error(message);
}
constexpr const char* advance_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void metalrt_ring_advance(device uint* state [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i==0) {state[1]=state[0];state[0]=(state[0]+1u)%state[2];}
}
)METAL";
void reset_selector(const std::shared_ptr<detail::RingPipelineState>& state) {
    if (!state->selector)
        return;
    const uint32_t initial[4] = {0, 0, static_cast<uint32_t>(state->ring.slot_count()), 0};
    std::memcpy(state->selector.contents(), initial, sizeof(initial));
}
} // namespace

RingBuffer::RingBuffer(Device device, size_t count, size_t bytes, size_t alignment)
    : state_(std::make_shared<detail::RingState>()) {
    if (!device || !count || !bytes)
        throw std::invalid_argument(
            "ring requires a Device, a positive slot count, and nonempty slots");
    if (count > UINT32_MAX)
        throw std::invalid_argument("ring slot count exceeds uint32_t GPU selector range");
    if (!alignment || (alignment & (alignment - 1)) || alignment > 256)
        throw std::invalid_argument("ring alignment must be a power of two no greater than the "
                                    "portable 256-byte allocation alignment");
    const size_t stride = checked_add(bytes, alignment - 1) & ~(alignment - 1);
    state_->device = std::move(device);
    state_->count = count;
    state_->bytes = bytes;
    state_->stride = stride;
    state_->storage = state_->device.allocate(checked_mul(count, stride), StorageMode::Shared);
    state_->tickets.resize(count);
}
RingBuffer::Lease::Lease(std::shared_ptr<detail::RingState> state, size_t slot, uint64_t generation)
    : state_(std::move(state)), slot_(slot), generation_(generation), active_(true) {}
RingBuffer::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), generation_(other.generation_),
      active_(std::exchange(other.active_, false)) {}
RingBuffer::Lease& RingBuffer::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        abandon();
        state_ = std::move(other.state_);
        slot_ = other.slot_;
        generation_ = other.generation_;
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}
RingBuffer::Lease::~Lease() noexcept {
    abandon();
}
void RingBuffer::Lease::abandon() noexcept {
    if (!active_ || !state_)
        return;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->generation == generation_ && state_->phase == detail::RingState::Phase::Cpu) {
            state_->phase = detail::RingState::Phase::None;
            ++state_->stats.abandoned_leases;
        }
        active_ = false;
    }
    state_->changed.notify_all();
}
size_t RingBuffer::Lease::slot() const {
    require_state(active_ && state_, "inactive ring lease");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->check_lease(slot_, generation_, detail::RingState::Phase::Cpu);
    return slot_;
}
size_t RingBuffer::Lease::size_bytes() const {
    require_state(active_ && state_, "inactive ring lease");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->check_lease(slot_, generation_, detail::RingState::Phase::Cpu);
    return state_->bytes;
}
BufferView RingBuffer::Lease::view() const {
    require_state(active_ && state_, "inactive ring lease");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->check_lease(slot_, generation_, detail::RingState::Phase::Cpu);
    return state_->view(slot_);
}
void* RingBuffer::Lease::data() const {
    return view().data();
}
void RingBuffer::Lease::begin_submission(Stream& stream) {
    require_state(active_ && state_, "inactive ring lease");
    if (stream.device() != state_->device)
        throw std::invalid_argument("ring and submission stream belong to different Devices");
    if (stream.is_recording())
        throw std::logic_error(
            "ring submission requires an idle recording state; it owns the submission boundary");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->check_lease(slot_, generation_, detail::RingState::Phase::Cpu);
    if (state_->failure)
        std::rethrow_exception(state_->failure);
    state_->phase = detail::RingState::Phase::Submitting;
}
Ticket RingBuffer::Lease::complete_submission(Ticket ticket) {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->check_lease(slot_, generation_, detail::RingState::Phase::Submitting);
        state_->tickets[slot_] = ticket;
        state_->cursor = (slot_ + 1) % state_->count;
        state_->phase = detail::RingState::Phase::None;
        ++state_->stats.submissions;
        active_ = false;
    }
    state_->changed.notify_all();
    return ticket;
}
void RingBuffer::Lease::fail_submission(Ticket ticket, std::exception_ptr error) noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->tickets[slot_] = std::move(ticket);
        state_->failure = error;
        state_->phase = detail::RingState::Phase::None;
        active_ = false;
    }
    state_->changed.notify_all();
}
Ticket RingBuffer::Lease::launch(Stream& stream, Graph& graph,
                                 const std::vector<EventPoint>& waits) {
    begin_submission(stream);
    try {
        return complete_submission(graph.launch(&stream, waits));
    } catch (...) {
        fail_submission(stream.last_ticket(), std::current_exception());
        throw;
    }
}
Ticket RingBuffer::Lease::submit(Stream& stream,
                                 const std::function<void(Stream&, BufferView)>& encode) {
    if (!encode)
        throw std::invalid_argument("ring encoding callback is empty");
    BufferView binding = view();
    begin_submission(stream);
    try {
        auto batch = stream.batch();
        encode(stream, std::move(binding));
        batch.finish();
        return complete_submission(stream.last_ticket());
    } catch (...) {
        fail_submission(stream.last_ticket(), std::current_exception());
        throw;
    }
}
std::optional<RingBuffer::Lease> RingBuffer::acquire_impl(bool nonblocking) {
    require_state(static_cast<bool>(state_), "moved-from ring");
    auto state = state_;
    Ticket preceding;
    size_t index;
    uint64_t token;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        for (;;) {
            if (state->failure)
                std::rethrow_exception(state->failure);
            // Observe failures in ANY completed slot before granting new CPU
            // ownership, without waiting for unrelated in-flight slots.
            for (const auto& ticket : state->tickets)
                if (ticket && ticket.ready()) {
                    try {
                        ticket.wait();
                    } catch (...) {
                        state->failure = std::current_exception();
                        std::rethrow_exception(state->failure);
                    }
                }
            if (!state->maintenance && state->phase == detail::RingState::Phase::None)
                break;
            if (nonblocking) {
                ++state->stats.try_misses;
                return std::nullopt;
            }
            if (state->phase != detail::RingState::Phase::None &&
                state->owner == std::this_thread::get_id())
                throw std::logic_error(
                    "release or submit the current CPU lease before acquiring another");
            state->changed.wait(lock);
        }
        index = state->cursor;
        preceding = state->tickets[index];
        if (nonblocking && !preceding.ready()) {
            ++state->stats.try_misses;
            return std::nullopt;
        }
        if (!preceding.ready())
            ++state->stats.slot_waits;
        state->phase = detail::RingState::Phase::Waiting;
        state->owner = std::this_thread::get_id();
        token = ++state->generation;
    }
    try {
        preceding.wait();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->failure = std::current_exception();
            state->phase = detail::RingState::Phase::None;
        }
        state->changed.notify_all();
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->tickets[index] = {};
        state->phase = detail::RingState::Phase::Cpu;
        ++state->stats.acquisitions;
    }
    return Lease{std::move(state), index, token};
}
RingBuffer::Lease RingBuffer::acquire() {
    auto lease = acquire_impl(false);
    return std::move(*lease);
}
std::optional<RingBuffer::Lease> RingBuffer::try_acquire() {
    return acquire_impl(true);
}
void RingBuffer::drain() {
    require_state(static_cast<bool>(state_), "moved-from ring");
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->phase != detail::RingState::Phase::None || state_->maintenance)
        throw std::logic_error("drain requires no outstanding CPU lease or maintenance operation");
    state_->maintenance = true;
    const auto work = state_->tickets;
    lock.unlock();
    std::exception_ptr error;
    for (const auto& ticket : work)
        try {
            ticket.wait();
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
    lock.lock();
    if (error)
        state_->failure = error;
    state_->maintenance = false;
    lock.unlock();
    state_->changed.notify_all();
    if (error)
        std::rethrow_exception(error);
}
std::vector<uint8_t> RingBuffer::readback(size_t index) {
    require_state(static_cast<bool>(state_), "moved-from ring");
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (index >= state_->count)
        throw std::out_of_range("ring slot index");
    if (state_->failure)
        std::rethrow_exception(state_->failure);
    if (state_->phase != detail::RingState::Phase::None || state_->maintenance)
        throw std::logic_error(
            "readback requires no outstanding CPU lease or maintenance operation");
    state_->maintenance = true;
    const auto ticket = state_->tickets[index];
    if (!ticket.ready())
        ++state_->stats.slot_waits;
    lock.unlock();
    std::vector<uint8_t> result;
    try {
        ticket.wait();
        result.resize(state_->bytes);
        std::memcpy(result.data(), state_->view(index).data(), result.size());
    } catch (...) {
        lock.lock();
        state_->failure = std::current_exception();
        state_->maintenance = false;
        lock.unlock();
        state_->changed.notify_all();
        throw;
    }
    lock.lock();
    ++state_->stats.readbacks;
    state_->maintenance = false;
    lock.unlock();
    state_->changed.notify_all();
    return result;
}
void RingBuffer::recover_with(const std::function<void()>& reset) {
    require_state(static_cast<bool>(state_), "moved-from ring");
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->phase != detail::RingState::Phase::None || state_->maintenance)
        throw std::logic_error(
            "recover requires no outstanding CPU lease or maintenance operation");
    state_->maintenance = true;
    const auto work = state_->tickets;
    lock.unlock();
    // Recovery is the explicit acknowledgement of failure. Every command is
    // nevertheless waited before storage can return to CPU ownership.
    for (const auto& ticket : work)
        try {
            ticket.wait();
        } catch (...) {
        }
    try {
        if (reset)
            reset();
    } catch (...) {
        lock.lock();
        state_->failure = std::current_exception();
        state_->maintenance = false;
        lock.unlock();
        state_->changed.notify_all();
        throw;
    }
    lock.lock();
    state_->tickets.assign(state_->count, Ticket{});
    state_->cursor = 0;
    state_->failure = {};
    state_->maintenance = false;
    ++state_->stats.recoveries;
    lock.unlock();
    state_->changed.notify_all();
}
void RingBuffer::recover() {
    recover_with({});
}
bool RingBuffer::poisoned() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    std::lock_guard<std::mutex> lock(state_->mutex);
    return static_cast<bool>(state_->failure);
}
size_t RingBuffer::slot_count() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    return state_->count;
}
size_t RingBuffer::bytes_per_slot() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    return state_->bytes;
}
size_t RingBuffer::stride_bytes() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    return state_->stride;
}
Device RingBuffer::device() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    return state_->device;
}
RingStatistics RingBuffer::statistics() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->stats;
}
BufferView RingBuffer::binding(size_t index) const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    if (index >= state_->count)
        throw std::out_of_range("ring slot index");
    return state_->view(index);
}
BufferView RingBuffer::storage_binding() const {
    require_state(static_cast<bool>(state_), "moved-from ring");
    return state_->storage.view();
}

RingPipeline::RingPipeline(Device device, DispatchPolicy policy, size_t slots, size_t bytes,
                           RingBindingMode mode, Builder builder)
    : state_(std::make_shared<detail::RingPipelineState>(std::move(device), policy, slots, bytes,
                                                         mode)) {
    if (!builder)
        throw std::invalid_argument("ring pipeline builder is empty");
    if (mode != RingBindingMode::PerSlot && mode != RingBindingMode::Indexed)
        throw std::invalid_argument("invalid ring binding mode");
    const size_t variants = mode == RingBindingMode::Indexed ? 1 : slots;
    Kernel advance;
    if (mode == RingBindingMode::Indexed) {
        state_->selector = state_->stream.device().allocate(4 * sizeof(uint32_t));
        reset_selector(state_);
        advance = state_->stream.device().kernel(advance_source, "metalrt_ring_advance",
                                                 [](const ComputeCommand& c) {
                                                     auto* values = c.buffer<uint32_t>(0);
                                                     values[1] = values[0];
                                                     values[0] = (values[0] + 1u) % values[2];
                                                 });
    }
    for (size_t i = 0; i < variants; ++i) {
        auto graph = std::make_unique<Graph>();
        auto scope = graph->record_guard(&state_->stream);
        RingBindings bindings;
        bindings.slots = slots;
        bindings.bytes_per_slot = bytes;
        bindings.stride_bytes = state_->ring.stride_bytes();
        if (mode == RingBindingMode::Indexed) {
            bindings.storage = state_->ring.storage_binding();
            bindings.slot_index = state_->selector.view(sizeof(uint32_t), sizeof(uint32_t));
            Command command;
            command.compute.kernel = advance;
            command.compute.grid = {1, 1, 1};
            command.compute.group = {1, 1, 1};
            command.compute.control = true;
            command.compute.arguments = {
                Argument::buffer(0, state_->selector.view(), Access::ReadWrite)};
            state_->stream.enqueue(std::move(command));
        } else {
            bindings.storage = state_->ring.binding(i);
            bindings.slot = i;
        }
        builder(state_->stream, bindings);
        scope.finish();
        graph->prepare(&state_->stream);
        state_->graphs.push_back(std::move(graph));
    }
}
RingPipeline::Lease::Lease(std::shared_ptr<detail::RingPipelineState> state,
                           RingBuffer::Lease lease)
    : state_(std::move(state)), lease_(std::move(lease)) {}
Ticket RingPipeline::Lease::submit(const std::vector<EventPoint>& waits) {
    require_state(static_cast<bool>(state_), "inactive ring pipeline lease");
    const auto index = state_->mode == RingBindingMode::Indexed ? 0 : lease_.slot();
    return lease_.launch(state_->stream, *state_->graphs.at(index), waits);
}
RingPipeline::Lease RingPipeline::acquire() {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return Lease{state_, state_->ring.acquire()};
}
std::optional<RingPipeline::Lease> RingPipeline::try_acquire() {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    auto lease = state_->ring.try_acquire();
    if (!lease)
        return std::nullopt;
    return Lease{state_, std::move(*lease)};
}
void RingPipeline::drain() {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    state_->ring.drain();
    state_->stream.synchronize();
}
std::vector<uint8_t> RingPipeline::readback(size_t index) {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->ring.readback(index);
}
void RingPipeline::recover() {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    auto state = state_;
    state->ring.recover_with([state] {
        try {
            state->stream.synchronize();
        } catch (...) {
        }
        reset_selector(state);
    });
}
bool RingPipeline::poisoned() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->ring.poisoned();
}
size_t RingPipeline::slot_count() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->ring.slot_count();
}
size_t RingPipeline::graph_count() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->graphs.size();
}
RingBindingMode RingPipeline::mode() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->mode;
}
RingStatistics RingPipeline::ring_statistics() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->ring.statistics();
}
Statistics RingPipeline::stream_statistics() const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return state_->stream.statistics();
}
const Graph& RingPipeline::graph(size_t variant) const {
    require_state(static_cast<bool>(state_), "moved-from ring pipeline");
    return *state_->graphs.at(variant);
}
} // namespace metalrt
