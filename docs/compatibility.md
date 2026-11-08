# Capture compatibility

metalrt preserves guard-based capture for Metal ports of code using
**tiny-cuda-nn's capture/update/execute wrapper**. It implements the subset of
CUDA stream-capture behavior needed by the port, with additional facilities for
explicit replay, transactions, parallel groups, GPU control flow, rings, and
parameters. Behavioral compatibility is the target; complete CUDA API coverage
and binary compatibility are not provided.

This comparison describes the current implementation. A missing feature does
not imply that Metal cannot support an emulation of it, and a documented
behavioral difference does not establish why it was originally introduced.

## The interface being preserved

CUDA's raw capture lifecycle separates recording, executable creation, and
launch. tiny-cuda-nn's `CudaGraph::capture_guard()` wraps ending capture,
updating or instantiating the executable, and launching it on scope exit.
See the [CUDA graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
and the [tiny-cuda-nn wrapper](https://github.com/NVlabs/tiny-cuda-nn/blob/1317f3b90efa62bf4c9ccdf39ecc4d5ef1b6ee13/include/tiny-cuda-nn/cuda_graph.h).

The corresponding metalrt usage retains that scope shape:

```cpp
// stream and the workload's kernels/buffers already exist.
metalrt::Graph graph;
{
    auto capture = graph.capture_guard(&stream);
    // Issue existing workload operations through stream here.
} // Successful scope completion prepares/submits the captured work.
```

`capture_guard()` submits once at successful completion; it is not a record-only
operation. A later call to `graph.launch(&stream)` submits another execution.
Repeated guarded captures replace the recorded sequence and let the backend
reuse, update, or rebuild its plan as appropriate.

`record_guard()`, `prepare()`, and `launch()` expose separate lifecycle steps as
additional facilities. They do not replace the guard-based interface. An
explicit `Scope::finish()` reports completion-of-recording/submission errors to
the caller; it does not normally wait for GPU execution. Scope destruction
defers such errors to the stream, and stack unwinding cancels the recording.

Implementation: [Graph](../src/runtime/graph.cpp),
[scope lifetime](../src/runtime/completion.cpp), and
[Metal plan preparation](../src/backends/metal/preparation.mm).

## How differences are classified

- **Not implemented:** functionality is absent from the current runtime. This
  is not a claim that it is impossible on Metal or a promise to implement it.
- **Different API or representation:** the runtime exposes a distinct model.
  Matching CUDA would require an adapter or a design change; Metal does not
  inherently require that public interface.
- **Unresolved mismatch:** behavior differs, but its compatibility rationale
  is not established. Preserve it during cleanup and decide its intended
  contract before changing it.
- **Native Metal constraint:** a documented platform rule requires a different
  implementation mechanism. It does not automatically prevent matching the
  user-facing behavior.

Full CUDA graph coverage is outside this repository's stated target. For
individual missing features below, whether they belong in the required subset
remains a scope decision; absence alone does not tell us whether a feature was
previously deferred or deliberately excluded.

## Current behavior and compatibility boundaries

| Area | Current behavior and comparison | Classification and explanation |
| --- | --- | --- |
| Guard completion | `capture_guard()` records and submits on successful completion, matching the wrapper's overall capture/update/execute shape. Raw CUDA end-capture alone does not launch. | **Aligned at the wrapper level.** Matching the shape does not establish equivalence for every failure or ordering case. |
| Null/default stream | A null `Stream*` selects metalrt's independent, thread-local default stream and permits capture. The tiny-cuda-nn wrapper skips capture for null/legacy streams. | **Different API or representation.** Null selects a metalrt default, not CUDA's legacy stream. This is tested behavior, not a Metal restriction. An adapter needs an explicit null-stream policy. |
| Nested guards | A guard on an already-recording stream is inactive; commands remain in the enclosing recording, and the inner graph is not replaced. The wrapper similarly skips a guard on an already-capturing stream. | **Partially aligned.** metalrt also treats an enclosing batch as recording. CUDA-wide capture interactions and the wrapper's thread-local `current_capture()` helper are not reproduced. |
| Capture across streams | `signal()` and `wait()` reject calls while recording. Fresh event points can instead be supplied to `Graph::launch()` outside recording. | **Not implemented:** CUDA-style event-driven fork/join capture into a shared graph. Launch dependencies and parallel groups do not provide that contract. Metal event synchronization exists, so the absence is not evidence of a fundamental platform barrier. |
| Capture modes | Recording state belongs to its initiating host thread; there is no Global/ThreadLocal/Relaxed capture-mode API. The tiny-cuda-nn wrapper selects Relaxed mode. | **Not implemented:** CUDA capture-mode policies. Thread ownership is the current runtime rule, not proof that Metal requires the same restriction. |
| Capture invalidation | Rejected commands throw. There is no CUDA-style invalidated-capture status. Cancellation and nested checkpoints provide rollback; a rejected operation caught inside a scope does not necessarily invalidate the whole recording. | **Different API or representation.** Local validation and rollback are implemented. A CUDA-compatible invalidation state machine is **not implemented**, and the desired failure contract needs explicit agreement. |
| Recapture and updates | One `Graph` owns commands and a cached plan. Successful recapture publishes a new snapshot; updates are transactional and use command indices. In-flight submissions retain their versions. | **Different API or representation.** CUDA exposes separate graph/executable/node handles and update results. metalrt supplies its own plan reuse/versioning rather than those exact handles or status codes. |
| Failed recapture | Cancellation leaves the previous snapshot intact; a submission failure before publication preserves the previous commands and discards a potentially touched plan cache. | **Unresolved mismatch** against the wrapper's failure/reset paths. Preservation is an implemented recovery contract, not a known Metal requirement. Compare individual failure paths before claiming equivalence. |
| Scheduled synchronization | `schedule_synchronize()` requests a stream wait after the next successful execute-on-finish capture. The wrapper performs device-wide synchronization after ending capture, before update/launch. | **Unresolved mismatch.** Both scope and timing differ. The reason is not established; this must not be described as forced by Metal. `synchronize_device()` exists in metalrt. |
| Graph topology | Commands form a sequence with checked static parallel groups. There is no general public node-and-edge graph builder. Launching a graph during recording appends command snapshots, rather than preserving a child-graph node. | **Different API or representation**, with general DAG construction **not implemented**. Supporting the port's subset does not require exposing every CUDA graph facility. |
| Host work | `Stream::host()` is a synchronous CPU boundary for batches and is rejected during capture. It is not a replayable host node. | **Not implemented:** replayable host nodes. The current boundary has a different purpose; no Metal impossibility is established. |
| Execution resources | Native submissions use fresh command buffers while reusable plans/indirect compute segments retain preparation work. Borrowed command buffers are committed by their external owner. | **Native Metal constraint:** a command buffer can be committed only once. Graph replay must therefore reuse preparation/indirect resources rather than recommit that same buffer. This implementation difference need not change the guard interface. |

The wrapper comparisons above refer to the pinned tiny-cuda-nn source, not a
claim of compatibility with every upstream revision. Its capture lifecycle,
null/nested checks, and scheduled synchronization are visible in
[`cuda_graph.h`](https://github.com/NVlabs/tiny-cuda-nn/blob/1317f3b90efa62bf4c9ccdf39ecc4d5ef1b6ee13/include/tiny-cuda-nn/cuda_graph.h).
CUDA's cross-stream capture and invalidation rules are described in the
[stream-capture sections](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture).
The mode interface is described in the
[CUDA stream reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
Apple documents the [single-commit rule](https://developer.apple.com/documentation/metal/mtlcommandbuffer/commit())
and [shared event synchronization](https://developer.apple.com/documentation/metal/mtlsharedevent).

## Source API and binary boundaries

Porting the capture wrapper does not make metalrt a drop-in CUDA runtime:

- `Stream*`, `Graph`, and `Ticket` are metalrt types, not CUDA handles.
- CUDA binaries cannot link against metalrt as an ABI-compatible replacement.
  C++ exceptions and completion tickets also differ from CUDA return codes.
- CUDA kernel launches and library calls must be ported to operations submitted
  through metalrt. Arbitrary native Metal encoding is not intercepted.
- Kernel bindings and constants follow the Metal shader ABI. Trivially copyable
  C++ parameters still need matching MSL layouts, widths, offsets, and alignment.
- Ordinary `dispatch()` takes a grid of **threads**; GPU-generated
  `dispatch_indirect()` consumes **threadgroup counts**. CUDA launch grids use
  threadblock counts; preserving the capture interface does not remove this
  launch adaptation.

Existing buffer/resource retention, optional-buffer bindings, and external-owner
contracts remain part of metalrt's API. They should not be interpreted as
matching CUDA ownership or kernel-argument ABI rules automatically.

## Compatibility test priorities

The existing suites establish metalrt behavior. They are not a differential
CUDA conformance suite and do not prove complete wrapper equivalence. Keep
current behavior tested while deciding which mismatches to preserve, adapt, or
close. The next compatibility work should retain the guard-based call pattern.

| Priority area | Existing coverage | Compatibility work to add or decide |
| --- | --- | --- |
| Default streams | `default_stream_capture` in [recording.cpp](../tests/runtime/recording.cpp) verifies null-stream capture. | Compare explicit, null, per-thread, and ordered-default cases. Decide whether a wrapper adapter skips null capture while the core keeps its current default. |
| Nested guards | `cancellation_and_nested_capture` in [recording.cpp](../tests/runtime/recording.cpp); stale and out-of-order scopes in [test_scope_safety.cpp](../tests/test_scope_safety.cpp). | Compare same-stream nesting, guards inside batches, and independent-stream nesting. Establish whether wrapper-wide capture-stack ordering is required. |
| Recapture | `policy_sequence` in [execution.cpp](../tests/runtime/execution.cpp); empty recapture in [validation.cpp](../tests/runtime/validation.cpp); version retention in [completion.cpp](../tests/runtime/completion.cpp). | Exercise unchanged work, changed bindings, changed topology, empty captures, and failures while older launches remain in flight. Define expected reuse separately from numerical correctness. |
| Synchronization | Scheduled completion and rejection during capture in [recording.cpp](../tests/runtime/recording.cpp). | Use independent queued work to distinguish a stream wait from a device wait, and work before/after capture to distinguish pre-launch from post-launch synchronization. Resolve the scheduled-synchronization mismatch. |
| Failures | Cancellation in [recording.cpp](../tests/runtime/recording.cpp), deferred errors/failed publication in [completion.cpp](../tests/runtime/completion.cpp), and rollback in [test_scope_safety.cpp](../tests/test_scope_safety.cpp). | Compare rejected operations caught inside capture, escaping exceptions, finish errors, backend failures, and failed recapture. Decide which cases invalidate capture or preserve an earlier graph. |
| Event dependencies | `cross_stream_events` and `event_validation` in [test_graph_features.cpp](../tests/test_graph_features.cpp) cover launch dependencies and capture-time rejection. | Keep rejection tests until the feature exists. If cross-stream capture enters scope, add producer/consumer fork/join, required rejoin, and event reuse tests; ordinary event tests are insufficient. |

Where CUDA hardware is available, run equivalent workloads against the pinned
wrapper and compare execution count, results, ordering, and failure outcomes.
Run the metalrt scenarios on Reference for recording semantics and on native
Metal for GPU behavior. Record deliberate deviations explicitly rather than
weakening tests to accept either outcome. These are test priorities, not newly
implemented tests or a commitment to full CUDA coverage.
