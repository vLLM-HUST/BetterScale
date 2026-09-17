# Expanded expert wire and colocated EP control

Fletcher approved the same-model control on September17. These are prototype
routes, not a public Worker default change and not unmodified vLLM results.
See `expanded-ep-gates.json` for the completed leaf gates; full-root and workload
qualification must be recorded separately.

## Communication capacity

`build.py <output> --rows 1024` builds ABI3 with1024 token rows **per source**.
Supported compile-time capacities are32/128/256/512/1024; descriptors still carry
actual rows. Capacity is read from the binary receipt, negotiated by both sides
before memory export, and used to size source/output memory and two reusable
server slots. All TOPK routes may land on one owner: output/scratch allocation
never assumes balanced routing. This is a protocol bound, not max context or
maximum active request capacity.

The1024 binary required removing stack-proportional routing arrays. Coordinator
route IDs now occupy slot-owned HBM (`slot[14]`); each worker reads at most256
map entries into scalar scratch. Numeric row scratch remains64KiB. Larger
capacities would require another coordinator UB change, not just raising a CLI
bound. The client allocates banks only for warmed shapes, not every integer
size up to capacity. A new row bucket during graph capture is rejected.

Passed on hw0, `runs/qwen38-wire-20260917T135943Z`:

- one publishing client + four owners, real target layer0 weights;
- rows1,4,32,1023,1024;30 calls drained by each owner;
- every output row is equal between changed-input FULL replay and eager;
- independent CPU arithmetic covers all rows through32 and sampled tile/boundary
  rows at1023/1024; maximum relative L2 approximately3.27e-6.

This does not yet qualify two-source co-batching at1024, BF16 MTP at1024, or
full-model long prefill throughput. Existing small MTP gates remain distinct.
Use `QWEN38_BUILD=<absolute build path>` with `run_wire.sh`/`run_model.sh` to
select an expanded capsule explicitly; the historical32 binary remains default.

## Colocated control

`colocated_ep.py` borrows the native Ascend MC2 dispatch/combine and NZ grouped
GEMM sequence used by vLLM-Ascend. Device route counts drive GEMM. Each EP rank
owns64/512 experts. Shared experts stay in each TP2 attention group. No host
route counts or manually flattened global activation all-reduce substitutes for
native EP.

`colocated_model.py` stripes a TP pair's replicated token rows once at EP ingress;
odd tails are masked. After native EP combine, TP all-gather restores the original
row order and shared output is added. All EP8 members execute the same
layer/phase, including idle masked senders; independent layer progression is
only supported by the separated topology.

The owned runtime facade equated TP with WORLD. `prepare_colocated_overlay.py`
creates a distinct TP/DP-aware overlay while reusing the pinned source/native
closure. `bootstrap_groups()` creates four TP2 and two DP4 subgroups in identical
order on all eight ranks, before model loading and graph capture. Never point
separated runs at this modified closure by accident.

Native target GMM with BF16 output requires BF16 down scales; they are converted
once during initialization. This native numeric boundary differs from the
persistent server's FP32 scaling. The eight-device layer0 gate passes uneven
active rows `[0,1,2,3,4,0,1,2]`, including idle senders, exact eager/replay active
outputs, and independent reference relative L2 <=0.005392. It is not full-model
quality evidence. The BF16 MTP branch still needs complete-model qualification.

Load catalogs under a CPU default-device context: model construction otherwise
leaves an NPU default active, and safetensors' mapped MTP slice can fail before
explicit upload. Keep ND assembly on host and only NZ tensors resident on NPU.

For full-root qualification use the ordinary admitted model launcher with
`--colocated --sources 4` and `QWEN38_OVERLAY=<colocated overlay>`. It initializes
WORLD8 instead of independent TP worlds, keeps token agreement checks inside
TP pairs, and retains the native MTP program. Do not label fixed short-window
results as SWE serving throughput.

## Completed full-model gates

- `runs/qwen38-model-20260917T141139Z`: four TP2 attention groups, colocated EP8,
  two requests/group, prompt3, MTP K1, eight decode steps. All eight roles exit
  successfully; all three same-State decode shadow tensors match exactly on
  every rank, and both members of every TP pair agree on output. Catalogs contain
  all48 target layers plus BF16 MTP. This is the native BF16-down-scale lane.
- `runs/qwen38-model-20260917T141508Z`: two TP2 sources+E4 with ABI3 capacity1024,
  two requests/source, prompt512/request (1024 actual rows/source), distinct
  prompts, MTP K1 and eight decode steps. Both sources complete long-prefill
  initialization and FULL decode. All four attention shadow checks pass. Each
  expert owner completes598 calls/source,1098 waves total (98 paired waves).
  The aggregate pairing counter does not identify which rows/phases were paired.

These gates qualify the two requested implementation changes. They do not
compare SWE makespan: their request counts/prompts differ, prefill is eager,
and they use a fixed-membership short runner. The downloaded SWE sessions still
need multi-turn continuation/admission integration with correct retained State;
the native colocated arm additionally needs coordinated layer/phase submission.
Do not manufacture performance conclusions from these unequal gate workloads.
