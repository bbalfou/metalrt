// Structured GPU control flow records bounded compute sequences and predicate kernels.
// The CPU callbacks construct commands once; GPU predicates select payload work at replay.
#include <metalrt/control.h>
#include <cstring>

namespace metalrt {
namespace {
struct ConditionABI {
    uint32_t comparison, reference, has_parent;
};
static_assert(sizeof(ConditionABI) == 12, "control condition ABI");
// These small implementation kernels are embedded so an installed runtime has
// no filesystem dependency. User-facing examples load their own .metal files.
const char* control_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct ConditionABI {uint comparison; uint reference; uint has_parent;};
inline bool test_condition(uint value,constant ConditionABI& c) {
    switch(c.comparison) {
    case 0:return value!=0; case 1:return value==0;
    case 2:return value==c.reference; case 3:return value!=c.reference;
    case 4:return value<c.reference; case 5:return value<=c.reference;
    case 6:return value>c.reference; default:return value>=c.reference;
    }
}
kernel void metalrt_branch_gate(device const uint* value [[buffer(0)]],
                               device const uint* parent [[buffer(1)]],
                               device uint* gates [[buffer(2)]],
                               constant ConditionABI& c [[buffer(3)]]) {
    bool enabled=c.has_parent==0 || parent[0]!=0;
    bool selected=enabled && test_condition(value[0],c);
    gates[0]=uint(enabled && selected); gates[1]=uint(enabled && !selected);
}
kernel void metalrt_loop_init(device const uint* value [[buffer(0)]],
                             device const uint* parent [[buffer(1)]],
                             device uint* state [[buffer(2)]],
                             constant ConditionABI& c [[buffer(3)]]) {
    state[0]=uint(c.has_parent==0 || parent[0]!=0);state[1]=0;state[2]=0;
}
kernel void metalrt_loop_test(device const uint* value [[buffer(0)]],
                             device const uint* parent [[buffer(1)]],
                             device uint* state [[buffer(2)]],
                             constant ConditionABI& c [[buffer(3)]]) {
    state[0]=uint(state[0]!=0 && test_condition(value[0],c));
    state[1]+=state[0];
}
kernel void metalrt_loop_finish(device const uint* value [[buffer(0)]],
                               device const uint* parent [[buffer(1)]],
                               device uint* state [[buffer(2)]],
                               constant ConditionABI& c [[buffer(3)]]) {
    state[2]=uint(state[0]!=0 && test_condition(value[0],c));
}
)metal";
bool test_condition(uint32_t value, const ConditionABI& condition) {
    switch (static_cast<Compare>(condition.comparison)) {
    case Compare::NonZero:
        return value != 0;
    case Compare::Zero:
        return value == 0;
    case Compare::Equal:
        return value == condition.reference;
    case Compare::NotEqual:
        return value != condition.reference;
    case Compare::Less:
        return value < condition.reference;
    case Compare::LessEqual:
        return value <= condition.reference;
    case Compare::Greater:
        return value > condition.reference;
    case Compare::GreaterEqual:
        return value >= condition.reference;
    }
    return false;
}
void validate(Stream& stream, const Predicate& condition) {
    if (!condition.value || condition.value.length < 4 || condition.value.offset % 4)
        throw std::invalid_argument("GPU condition requires an aligned uint32 buffer view");
    if (condition.value.buffer.device() != stream.device())
        throw std::invalid_argument("GPU condition and stream belong to different devices");
    if (static_cast<uint32_t>(condition.comparison) > static_cast<uint32_t>(Compare::GreaterEqual))
        throw std::invalid_argument("unknown GPU condition comparison");
}
void control(Stream& stream, Kernel kernel, const Predicate& condition, BufferView parent,
             BufferView state) {
    ConditionABI abi{static_cast<uint32_t>(condition.comparison), condition.reference,
                     parent ? 1u : 0u};
    Command command;
    auto& compute = command.compute;
    compute.kernel = std::move(kernel);
    compute.control = true;
    compute.grid = {1, 1, 1};
    compute.group = {1, 1, 1};
    compute.arguments = {Argument::buffer(0, condition.value),
                         Argument::buffer(1, parent ? parent : condition.value),
                         Argument::buffer(2, state, Access::ReadWrite), Argument::scalar(3, abi)};
    stream.enqueue(std::move(command));
}
} // namespace

void if_else(Stream& stream, Predicate condition, const std::function<void()>& then_body,
             const std::function<void()>& else_body) {
    validate(stream, condition);
    if (!then_body)
        throw std::invalid_argument("if_else requires a then-body callback");
    auto transaction = stream.transaction();
    auto parent = stream.current_predicate();
    auto gates = stream.device().allocate(2 * sizeof(uint32_t));
    auto kernel =
        stream.device().kernel(control_source, "metalrt_branch_gate", [](const ComputeCommand& c) {
            auto config = c.scalar<ConditionABI>(3);
            auto* gates = c.buffer<uint32_t>(2);
            const bool enabled = !config.has_parent || c.buffer<uint32_t>(1)[0] != 0;
            const bool selected = enabled && test_condition(c.buffer<uint32_t>(0)[0], config);
            gates[0] = enabled && selected;
            gates[1] = enabled && !selected;
        });
    control(stream, std::move(kernel), condition, parent, gates.view());
    {
        auto gate = stream.predicate_scope(gates.view(0, 4));
        then_body();
        gate.finish();
    }
    if (else_body) {
        auto gate = stream.predicate_scope(gates.view(4, 4));
        else_body();
        gate.finish();
    }
    transaction.finish();
}

LoopResult repeat_while(Stream& stream, Predicate condition, uint32_t maximum,
                        const std::function<void(uint32_t)>& body) {
    validate(stream, condition);
    if (!maximum || !body)
        throw std::invalid_argument("repeat_while requires a positive iteration bound and a body");
    auto transaction = stream.transaction();
    auto parent = stream.current_predicate();
    auto state = stream.device().allocate(3 * sizeof(uint32_t));
    auto init =
        stream.device().kernel(control_source, "metalrt_loop_init", [](const ComputeCommand& c) {
            auto config = c.scalar<ConditionABI>(3);
            auto* state = c.buffer<uint32_t>(2);
            state[0] = !config.has_parent || c.buffer<uint32_t>(1)[0] != 0;
            state[1] = 0;
            state[2] = 0;
        });
    auto step =
        stream.device().kernel(control_source, "metalrt_loop_test", [](const ComputeCommand& c) {
            auto* state = c.buffer<uint32_t>(2);
            state[0] = state[0] != 0 &&
                       test_condition(c.buffer<uint32_t>(0)[0], c.scalar<ConditionABI>(3));
            state[1] += state[0];
        });
    auto finish =
        stream.device().kernel(control_source, "metalrt_loop_finish", [](const ComputeCommand& c) {
            auto* state = c.buffer<uint32_t>(2);
            state[2] = state[0] != 0 &&
                       test_condition(c.buffer<uint32_t>(0)[0], c.scalar<ConditionABI>(3));
        });
    control(stream, init, condition, parent, state.view());
    for (uint32_t iteration = 0; iteration < maximum; ++iteration) {
        control(stream, step, condition, parent, state.view());
        auto gate = stream.predicate_scope(state.view(0, 4));
        body(iteration);
        gate.finish();
    }
    control(stream, finish, condition, parent, state.view());
    transaction.finish();
    return LoopResult{std::move(state)};
}
BufferView LoopResult::iterations() const {
    if (!state_)
        throw std::logic_error("empty loop result");
    return state_.view(4, 4);
}
BufferView LoopResult::limit_reached() const {
    if (!state_)
        throw std::logic_error("empty loop result");
    return state_.view(8, 4);
}
LoopStatus LoopResult::read(const Ticket& last_replay) const {
    if (!state_ || !last_replay)
        throw std::invalid_argument("loop readback requires a submitted replay ticket");
    last_replay.wait();
    auto* state = state_.data<uint32_t>();
    return {state[1], state[2] != 0};
}
} // namespace metalrt
