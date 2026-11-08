// Optimistic graph edits; commits preserve in-flight command versions.
#include "runtime_internal.h"

namespace metalrt {
Graph::Update::Update(std::shared_ptr<detail::GraphState> state, uint64_t revision,
                      std::vector<Command> commands, Device device)
    : state_(std::move(state)), revision_(revision), commands_(std::move(commands)),
      device_(std::move(device)) {}
namespace {
ComputeCommand& compute_node(std::vector<Command>& commands, size_t index) {
    require(index < commands.size() && commands[index].kind == Command::Kind::Compute,
            "graph update requires a valid compute command index");
    require(!commands[index].compute.control,
            "runtime control commands cannot be edited individually; use bulk rebind or record a "
            "new graph");
    return commands[index].compute;
}
Argument& mutable_argument(ComputeCommand& command, uint32_t index, Argument::Kind kind) {
    for (auto& argument : command.arguments)
        if (argument.index == index && argument.kind == kind)
            return argument;
    throw std::invalid_argument(
        "graph update argument index or type does not match the recorded command");
}
} // namespace
Graph::Update& Graph::Update::set_buffer(size_t command, uint32_t index, BufferView buffer) {
    auto& node = compute_node(commands_, command);
    for (auto& argument : node.arguments) {
        if (argument.index == index && (argument.kind == Argument::Kind::Buffer ||
                                        argument.kind == Argument::Kind::NullBuffer)) {
            argument.resource = std::move(buffer);
            argument.kind = Argument::Kind::Buffer;
            return *this;
        }
    }
    throw std::invalid_argument(
        "graph update argument index or type does not match the recorded buffer");
}
Graph::Update& Graph::Update::set_null_buffer(size_t command, uint32_t index) {
    auto& node = compute_node(commands_, command);
    for (auto& argument : node.arguments) {
        if (argument.index == index && (argument.kind == Argument::Kind::Buffer ||
                                        argument.kind == Argument::Kind::NullBuffer)) {
            argument.resource = {};
            argument.kind = Argument::Kind::NullBuffer;
            return *this;
        }
    }
    throw std::invalid_argument(
        "graph update argument index or type does not match the recorded buffer");
}
Graph::Update& Graph::Update::set_texture(size_t command, uint32_t index, Texture2D texture) {
    mutable_argument(compute_node(commands_, command), index, Argument::Kind::Texture).image =
        std::move(texture);
    return *this;
}
Graph::Update& Graph::Update::set_sampler(size_t command, uint32_t index, Sampler sampler) {
    mutable_argument(compute_node(commands_, command), index, Argument::Kind::Sampler).sampling =
        std::move(sampler);
    return *this;
}
Graph::Update& Graph::Update::set_bytes(size_t command, uint32_t index, const void* bytes,
                                        size_t size) {
    auto& argument =
        mutable_argument(compute_node(commands_, command), index, Argument::Kind::Bytes);
    require(bytes && size == argument.value.size(),
            "graph scalar update must preserve the recorded constant ABI size");
    std::memcpy(argument.value.data(), bytes, size);
    return *this;
}
Graph::Update& Graph::Update::set_grid(size_t command, Size3 grid, Size3 group) {
    auto& node = compute_node(commands_, command);
    require(!node.indirect_groups, "use set_indirect_groups for a GPU-driven grid");
    node.grid = grid;
    node.group = group;
    return *this;
}
Graph::Update& Graph::Update::set_indirect_groups(size_t command, BufferView counts,
                                                  Size3 maximum) {
    auto& node = compute_node(commands_, command);
    require(static_cast<bool>(node.indirect_groups),
            "node was not recorded as GPU-driven dispatch");
    require(static_cast<bool>(counts), "GPU-driven dispatch update requires a count buffer");
    node.indirect_groups = std::move(counts);
    node.max_groups = maximum;
    node.grid = {checked_mul(maximum.x, node.group.x), checked_mul(maximum.y, node.group.y),
                 checked_mul(maximum.z, node.group.z)};
    return *this;
}
Graph::Update& Graph::Update::set_predicate(size_t command, BufferView predicate) {
    auto& node = compute_node(commands_, command);
    require(!node.control, "cannot predicate an unconditional control kernel");
    node.predicate = std::move(predicate);
    return *this;
}
Graph::Update& Graph::Update::rebind(const Buffer& from, const Buffer& to) {
    require(static_cast<bool>(from) && static_cast<bool>(to), "rebind requires allocated buffers");
    auto replace = [&](BufferView& view) {
        if (view.buffer == from)
            view = to.view(view.offset, view.length);
    };
    for (auto& command : commands_) {
        replace(command.source);
        replace(command.destination);
        if (command.kind == Command::Kind::Compute) {
            auto& node = command.compute;
            replace(node.indirect_groups);
            replace(node.predicate);
            replace(node.dispatch_status);
            for (auto& argument : node.arguments)
                if (argument.kind == Argument::Kind::Buffer)
                    replace(argument.resource);
        }
    }
    return *this;
}
void Graph::Update::commit() {
    require(!committed_, "graph update transaction has already committed");
    if (!commands_.empty())
        validate_sequence(commands_, device_);
    std::lock_guard<std::mutex> l(state_->mutex);
    require(!state_->capturing && state_->revision == revision_,
            "graph update conflict: the graph changed since this transaction began");
    state_->commands = std::move(commands_);
    ++state_->revision;
    committed_ = true;
}

} // namespace metalrt
