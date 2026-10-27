# DeepSeek-V4.1-Flash-1017 on a consumer GPU

**Current checkpoint,2026-10-27:** the complete IQ1_M/Engram model runs through
Strata's reused C++/CUDA engine with a20GiB GPU target. User-confirmed7al adds
explicit36GB memory-experimental text input;windows stays default.
User-confirmed7am records one model/eight EOS answers in896.37s,zero observed
swap,peak RSS37.77GB/minimum host available15.14GB,content6/8 and strict
format5/8. Completed7an audits context/memory offline with45 checks and
proposes an explicit2048 context candidate. It awaits user review;the current
entry remains1024 and the original7af swap failure stays unresolved.

The earlier3–4-gate V1 closeout sketch includes the bounded resident gate.
Remaining acceptance work centers on configurable context and measured budget,
broader quality/known limitations,and reproducible setup/profile delivery.
The exact review count depends on the context trial. This audit does not
validate larger-context memory,quality,latency or sustained stability.
First-token waits40–98s and fresh KV per question remain practical limits.
Full-context quality and official numerical parity remain open. Earlier
calendar/gate estimates below are historical,not a fresh delivery countdown.
Proposed7ao adds one strict opt-in context selector plus at most1 model/3
requests after no-GPU controls;it has not started.

The following introductory implementation history was recorded through7n;
the current results and review status are recorded in7an below.

This work uses Strata as the native engine base, specifically for
DeepSeek-V4.1-Flash-1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The current IQ1_M/Engram path uses the complete pinned upstream `strata-dsv41`
engine. All five base files are verified, and its first short text generation
passes on the RTX 2080 Ti with capacity restricted to 20 GiB. User-confirmed
**7g** adds an opt-in pageable embedding table to bypass an observed large
pinned-memory allocation failure. User-confirmed **7h** records five short
real prompts on that binary. User-confirmed **7i** adds opt-in observations of
Strata's existing RAM-store counters. User-confirmed **7j** compares 32 and 40
decimal GB ceilings, with a modest short-workload gain from the larger store.
User-confirmed **7k** validates three longer tasks using that explicit 40 GB
candidate, with correct completed outputs up to 187 tokens. User-confirmed
**7l** adds the local single-turn text entry. User-confirmed **7m** adds resident
independent-question input with the same native runtime. Gate **7n** compares
native LRU/LFU RAM-store policies at fixed 40 GB RAM store and 20 GiB GPU capacity.
LFU shows no benefit in the bounded five-prompt workload; the current LRU profile
is retained. Per-question read traffic and first-token delays remain practical
limits. The comparison and proposed native prefill check await review below.

The earlier original-weight control connects layers **0 through 39** on the
RTX 2080 Ti. Four-stream residuals and same-token delayed mHC coefficients pass
through all 40 backbone decoders. Layers 1 and 14 apply their own Engram.
Sources 2, 8 and 14 own ratio-2 histories; source 20 owns ratio-1 KV, index keys
and candidate blocks. Sources 24, 28, 32 and 36 independently borrow those
KV/keys and select their own indices inside source-20 candidates. Consumers
25–27, 29–31, 33–35 and 37–39 use their own group's index source.

The path reuses Strata's original-ID expert caches, native routed computation,
shared experts and the existing attention/decoder components. `CudaTextPrefix`
accepts exactly 4, 8, 14, 15, 20, 24, 28, 32, 36 or 40 layers; its default remains
four. **40 is the full backbone count, with original IDs 0–39.** All use
384 routed experts and top-6. Separate MTP draft layers 40–42 use 128/top-3 and
remain outside this interface. Steps through **7m** are confirmed, including
bounded generation, common startup admission, a global CPU cache, unified
expert file reads, an explicit model manifest and an independent native entry.
Its fixed original-weight control passes the same 20 GiB capacity simulation.
The user redirected **7e** to a bounded quantization audit before any complete
model download. Real GSQ-RCO IQ1_M expert samples and IQ4_NL Engram rows now pass
on SM75 using existing Strata kernels. Gate **7f** built the complete pinned
upstream `strata-dsv41` engine in isolation and checked quantization, intake and
memory planning. Gate **7g** validates the complete base and one known-answer
text smoke. Gate **7h** adds a small resident-engine usability baseline; broad
quality and sustained performance remain open. Each gate ends with review and
confirmation before the next begins.

## Reuse across layers

Layer numbers identify original checkpoint weights and shared-state owners.
They do not imply a separate operator implementation for each of the 40 layers.
Sources 2, 8 and 14 reuse the ratio-2 components; consumers 3–7, 9–13, 15–19
and 21–23/25–27/29–31/33–35/37–39 reuse the same shared-consumer attention and decoder
components. Sources 24, 28, 32 and 36 reuse an optional index-query owner in those
same components.
Layers 1 and 14 reuse the Engram lookup and projection components, while
retaining their different decoder roles.

Source 20 introduces a different structure: ratio-1 latents and candidate-block
publication. Its components reuse existing kernels where the arithmetic is the
same. Later repeated consumers and index-source groups should be connected
through layer weights and ownership configuration once that structure is
supported. A review gate is an implementation or validation increment, not a
requirement to write each layer separately.

## Earlier remaining-work estimate (recorded 2026-10-26)

All 40 backbone layers, embedding and the output head are connected. Gate 6ak
adds the raw-text boundary and a bounded greedy feedback loop. Its current
validation uses the original prompt IDs `[0,455]` and two generated tokens.
Completing this control closes the minimal short-generation part of step 6;
it does not establish arbitrary-prompt coverage or full-model quality.

Gate 7a checks the common CPU/GPU startup budget before loading weights. Gate
7b adds bounded CPU expert caching and direct demand reads for the prepared
original expert files. Gate 7c replaces expert payload dispatch with a common
original-ID file index/source, retaining the budget and byte checks. Complete
expert-GGUF indexing has synthetic coverage; actual generation still uses the
prepared original subset. Gate 7d unifies dense/Engram/expert loading behind a
manifest and independent native entry. These original-weight controls retain
their bounded coverage. Repeated layer arithmetic reuses connected components.
Gate 7e checked the smaller GSQ-RCO IQ1_M package using about 55 MB of real
samples and audited the newer upstream engine. Gate 7f built that complete
engine for SM75. Gate 7g has now verified the complete 231.32 GB IQ1_M/Engram
base, native pack coverage and first short generation under 20 GiB capacity.
A small opt-in host embedding allocation workaround was validated after the
first load failed; its kernel/driver trigger is still not fully established.

The remaining scope is configurable context and sustained generation; measured
SSD/CPU/GPU scheduling; and whole-engine quality, stability and a usable entry.
At 7b this was tentatively **6–10 further review gates**. The pinned upstream
engine now supplies much of the runtime, so that number is not a fresh countdown
or a requirement to duplicate its features. Gate **7h** measures five real
prompts and preserves their answers, timings and memory/I/O observations. The
completed **7i** exposes the native RAM-store hit and read-wait counters in
the resident protocol and measures substantial expert-read waiting and cache
evictions. Gate **7j** compares the existing RAM-store ceiling at 32 versus 40
decimal GB under the same 20 GiB GPU bound. The larger store improves pooled
decode throughput by 4.8% in this small workload and leaves at least 13.34 GB of
observed host memory available. The user confirmed that comparison; the default
is unchanged. Gate **7k** validates longer inputs and outputs using the explicit
40 GB candidate. All three tasks complete and pass bounded answer checks, with
peak RAM 39.05 GB and minimum host memory available 10.47 GB. The 49–127 second
first-token waits remain substantial. User-confirmed 7k is followed by **7l**'s
bounded local text entry using the validated native engine. Its real CLI checks
pass, including controlled exits and GPU cleanup. Gate **7m** now keeps the
engine resident across independent questions and validates cap/rejection recovery,
EOF and idle interruption. Gate **7n** compares native LRU/LFU RAM-store policies
under the same limits. LFU has lower pooled decode rate and no measured first-token
benefit in this small workload, so the LRU profile is retained. Proposed **7o**
audits the existing native chunked-prefill path for SM75 and its memory needs,
then attempts a bounded comparison if compatible. Cross-run wording variation
is recorded in 7i and retained in the later bounded checks.

The earlier allowance of **6–10 review gates and 3–7 effective working days**
was recorded before 6ae for the then-remaining step-6 integration. It is not a
new allowance starting at 6ak. Downloads and confirmation time were additional;
no measured delivery rate follows from that estimate.

The planning estimate recorded on 2026-10-24 for a usable single-sequence
engine on **64 GB RAM and a 20 GB GPU** was roughly **3–6 weeks of sustained
development**, including step 6. This gate does not reset that estimate or
establish a delivery date. That budget includes a common memory limit, bounded SSD/CPU/GPU caches,
measured scheduling/transfer behavior, and stability checks. It is less certain
than the step-6 estimate. Cache misses and storage traffic may require more work
or a different storage format. A specific tokens/s target or beating Xolotl is
not established by this estimate. Revisit it after the first full-model run;
optional performance experiments have no fixed completion date yet.

## Hardware target: 64 GB RAM and 20 GB GPU memory

The target discussed on 2026-10-23 is single-sequence inference with 64 GB system
RAM and a 20 GB RTX 2080 Ti. The current tests run on an RTX 2080 Ti reporting
22528 MiB. In gate 7a, reserving the excess capacity leaves at most 20 GiB for
CUDA allocations, and the fixed original-weight 40-layer short completion
succeeds. This uses the existing guarded subset of experts and Engram rows;
it does not establish arbitrary-prompt coverage or full-checkpoint capacity.
The CLI uses explicit binary MiB/GiB units. General weight management and
measured steady-state resident memory remain later gates.

The current GPU expert baseline is NVFP4 weights with Q8_1 activations. Original
Engram tables stay FP8, while selected dense projections expand to BF16. From
the fixed layouts, rather than a whole-checkpoint conversion:

| Layout estimate | GiB | Meaning |
| --- | ---: | --- |
| All backbone experts in current native GPU format | 284.766 | 40 layers x 384 experts x 19906560 bytes; storage, not simultaneous residency |
| Both original Engram tables | 188.833 | `(384006168 + 384016682) * (256 + 8)` bytes |
| Expert source reads for one token with every expert slot missing | 4.202 | 40 layers x top-6 x 18800640 original bytes; excludes dense weights and Engram |
| Expert H2D for that same all-miss case | 4.449 | 40 layers x top-6 x 19906560 native bytes |

The full weights cannot reside in the combined 64 GB RAM and 20 GB GPU memory.
SSD storage and bounded CPU/GPU caching are required for this target. The numbers
above are not measured routing locality or storage throughput. A useful decode
rate depends on cache hits, actual SSD reads and transfer/computation overlap;
no tokens-per-second estimate follows from these capacities. Engram's small row
payload also incurs page-sized I/O, as measured in step 6j.

Dense residency, expert-slot budgets, KV capacity and prefill scratch must fit
a common limit with CUDA/runtime headroom. Gate 7a accounts for all 40 layers
under one startup allowance for the fixed profile; it does not yet pool slots
across layers or provide a general context planner. This constraint does not
change the one-review-gate-at-a-time workflow.

## Step 2: model file access

The opt-in `strata_deepseek41_artifact` library and `strata-deepseek41-inspect` command:

- Reuse Strata's `GgufFile`, `GgufModel`, split-file discovery and payload bounds checks.
- Check the 1017 core text metadata, including 40 layers, width 5120, 384 experts,
  top-6 routing, expert width 2304, SWA and the KV/index source lists.
- Describe each layer's compressed KV source, index source and candidate-source use.
- Locate gate, up and down for an expert, including roles stored in different shards.
- Read uniform expert tensors or Xolotl's per-expert Q2_0 / Q3_K / PTQ1_0 type maps.
  The expert ID stays unchanged; a type-local slot selects its stored bytes.
- Return mapped byte views or copy exactly one expert into a caller-owned buffer.
  No expert weight conversion or computation happens here.

Uniform raw layouts accepted by this adapter are F32, F16, BF16, Q8_0, Q3_K,
MXFP4, Q2_0 and PTQ1_0. This is file-layout support, not a claim that Strata's
existing kernels can execute every one at DeepSeek dimensions. HF FP4 tensors and
GGUF MXFP4 blocks are different storage layouts. The separate step-3 fixture
preparer below converts one expert without changing its FP4 codes or scales.

The CLI checks all 40 backbone layers by default. `--layer N` explicitly limits the
expert-layout check to one layer. It never infers full-model support from a partial
file. `--metadata-only` checks no expert payloads. Even a full expert-layout check
does not validate dense weights, Engram, DSpark, tokenizer behavior or model quality.
Matching metadata also does not establish checkpoint identity.

The feature defaults to OFF. The existing Qwen reader, model-family dispatch,
kernels and execution loop are unchanged. No PyTorch runtime is introduced.

## Step 3: one CPU expert

`CpuExpert` runs gate, up and down for one token. It reuses Strata's existing
`native_quant_h`, `native_down_rows` and BF16 conversion helpers. Only the model
orchestration is new: BF16 boundaries, the gate's upper-only clamp at 10, the up
projection's clamp to [-10, 10], SiLU and route weighting before hidden BF16/Q8
rounding. It does not choose routes or change expert IDs.

The existing Qwen expert pool rejects this shape: its input and hidden buffers
are 4096 and 1024 bytes, while 5120/2304 with Q8_0 needs 5440 and 2448 bytes.
The adapter supplies its own bounded buffers to the existing projection functions.
It uses `native_down_rows` for each separate matrix because `native_gu_rows` fuses
a Qwen SwiGLU without the required DeepSeek rounding and clamp. Shared kernels
and pool limits are unchanged. No new matrix kernel is implemented.

The supported execution baseline is **MXFP4 weights + ggml Q8_0 activations**.
The offline Python/NumPy fixture tool preserves every original FP4 code and E8M0
scale byte; it does not re-quantize weights. Python is not in the execution path.
The official reference uses FP8 activations, so matching the Q8 reference does
not establish official-reference parity or full-model quality. The other raw
formats accepted by the file adapter are not executable by `CpuExpert` yet.

The fixture contains one expert, has architecture `deepseek41-expert-fixture`,
and must not be accepted as a complete `deepseek41` model. `CpuExpert` keeps
non-owning weight views: their source model must outlive it. An instance owns
56,016 bytes of scratch at the real dimensions and supports one caller at a time.

## Sources and reuse boundaries

| Component | Reuse and current boundary |
| --- | --- |
| Model files | Strata's existing C++ GGUF reader is used now. Only the DeepSeek metadata and expert naming adapter is new. |
| DeepSeek conventions | Xolotl's metadata keys and uniform/mixed expert tensor contracts are used. Attribution is in [NOTICE.md](../src/deepseek41/NOTICE.md). Its engine and routing bias are not imported. |
| Expert execution | CPU uses `NativeFmt`, `native_quant_h` and `native_down_rows`. CUDA uses `DeviceArena`, `native_quantize_q8_1` and `iq_mmvq(40)`. Both reuse BF16 helpers. The fixed-size CPU pool is not connected; step 4 connects the existing expert cache. |
| Routed storage / dispatch | Reuse `ExpertSource`, `ExpertCache`, BF16 GEMV and `add_inplace`. Add the fixed 1017 score/bias rules and a per-layer LRU policy. No shared kernel or Qwen execution changes. |
| Shared expert | Losslessly expand original block-32 FP8 weights to BF16 at startup; reuse Strata `bf16_gemv_pair`, `bf16_gemv`, allocation and elementwise helpers. Add FP8 activation conversion and the 1017 MoE composition. |
| mHC / normalization | Reuse Strata `DeviceArena`, BF16 helpers and `native_qsa_rms_norm_weighted`; call the existing CUDA dependency cuBLAS for the original F32 projection. Add only the fixed 1017 coefficient, Sinkhorn and residual orchestration. |
| Layer-0 attention | Reuse Strata BF16 GEMV, RMSNorm, FP8 conversion and allocation, plus the existing cuBLAS dependency for QK and probability/KV products. Add fixed 1017 tail RoPE, window ordering, sink and BF16 numerator rules. No existing attention kernel is changed. |
| Layer-0 decoder | Compose the confirmed mHC, attention and MoE components in C++. Preserve incoming/next-layer coefficient ownership, prefill candidate grouping and one sequence state. No new arithmetic kernel in step 5d. |
| Layer-2 compressor | Reuse Strata allocation, BF16 conversion and RMSNorm, with the existing cuBLAS dependency for F32 projections. Add the fixed ratio-2 per-channel gate and incomplete-pair state. Outputs remain before RoPE for the later indexer. |
| Compressed KV cache | Reuse Strata allocation/BF16 helpers and CUDA FP4/E4M3 conversions. Add fixed 1017 tail YaRN RoPE and append-only capacity/position checks. Keep the original latent for the indexer. |
| Layer-2 decoder | Compose the confirmed mHC, ratio-2 source attention and text MoE; reuse attention's two-pass orchestration and Strata expert cache. Add 100 KiB of fixed residual/output scratch, with no new arithmetic kernel in step 6h. |
| Layer-3 decoder and pair | Compose the confirmed mHC, shared-attention consumer and text MoE; pass actual layer-2 residuals/coefficients on the GPU. Reuse Strata's expert cache, add 100 KiB fixed scratch, and borrow source KV/indices without duplication. No new arithmetic kernel in step 6i. |
| Further storage work | `ExpertSource` and `ExpertCache` are connected in step 4. Pinned staging, `DirectFile`, prefetch and overlap remain later measured optimizations. |
| Model semantics | Use the fixed official 1017 configuration and the prior DeepInfer investigations for mHC, SWA/YOCO state, FP8 rounding and later DSpark validation. No DeepInfer runtime or fallback kernel has been copied. |

Source pins:

- Strata base: `011c9e600a93bbc9fd451c563501b5831c32bbca`.
- ggml dependency: Strata's existing llama.cpp pin
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`; no dependency upgrade.
- Xolotl contract: `b3452666e07a6374d06bb8c2185dffa271e8f65f`,
  `tools/v41/engine/ds41_geometry.hpp` and `tools/v41/v41_gguf_meta.py`.
- Official model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Official config SHA256:
  `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.

## Build and use

The file reader needs a C++20 compiler and CMake. It needs neither CUDA nor a
ggml download. Run from this worktree's root:

```sh
cmake -S . -B build-deepseek-cpu -G Ninja \
  -DSTRATA_ENABLE_DEEPSEEK41=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=ON
cmake --build build-deepseek-cpu --target strata-deepseek41-inspect \
  deepseek41_artifact_test gguf_reader_test gguf_split_test \
  model_family_test gguf_geometry_test -j 4
ctest --test-dir build-deepseek-cpu --output-on-failure \
  -R '^(deepseek41_artifact_test|gguf_reader_test|gguf_split_test|model_family_test|gguf_geometry_test)$'

build-deepseek-cpu/src/deepseek41/strata-deepseek41-inspect MODEL.gguf --metadata-only
build-deepseek-cpu/src/deepseek41/strata-deepseek41-inspect MODEL.gguf
build-deepseek-cpu/src/deepseek41/strata-deepseek41-inspect MODEL.gguf --layer 0 --read-expert 0
```

Use any shard path of a split model; the reader locates the other shards. The last
command copies one expert's gate/up/down bytes and prints a diagnostic FNV-1a
checksum. It does not dequantize the expert or run a forward pass. Exit codes:
0 means the requested checks passed, 1 means a model/file check failed, and 2
means invalid arguments.

The CPU expert additionally needs Strata's pinned ggml dependency. The reader,
CPU and separate CUDA options all default to OFF. The dependency used here was extracted into the ignored
`build-deps` directory; `STRATA_GGML_DIR` can instead point to another checkout of
that same pin, without modifying it.

```sh
cmake -S . -B build-deepseek-native -G Ninja \
  -DSTRATA_ENABLE_DEEPSEEK41=ON -DSTRATA_DEEPSEEK41_CPU=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=ON \
  -DSTRATA_GGML_DIR="$PWD/build-deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d"
cmake --build build-deepseek-native --target deepseek41_expert_test \
  deepseek41_artifact_test gguf_reader_test gguf_split_test \
  model_family_test gguf_geometry_test bf16_bits_test -j 4
ctest --test-dir build-deepseek-native --output-on-failure \
  -R '^(deepseek41_expert_test|deepseek41_artifact_test|gguf_reader_test|gguf_split_test|model_family_test|gguf_geometry_test|bf16_bits_test)$'
python3 -m unittest discover -s tests/deepseek41 -p test_prepare_expert.py -v

python3 tools/deepseek41/prepare_expert_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --output build-deepseek-native/real-expert0-mxfp4.gguf
build-deepseek-native/src/deepseek41/deepseek41_expert_test \
  --real-expert build-deepseek-native/real-expert0-mxfp4.gguf
```

The model-dir above is the local cache used in this run. Supply a safetensors
directory containing the six weight/scale tensors for `layers.0.ffn.experts.0`
when reproducing elsewhere. The preparer requires NumPy and refuses to overwrite
an existing output. Reuse the existing fixture for subsequent native test runs.
It records source and packed tensor SHA256 values in the GGUF metadata.

## Step 3 validation

Measured locally on 2026-10-21, Linux, GCC 13.3.0, Intel Xeon E5-2686 v4.
The GPU was not used. Inputs below are deterministic test inputs, not activations
captured from a full model.

- Release with warnings treated as errors: all seven CTests in the command above
  passed, including the five existing reader/model/BF16 tests.
- Feature ON/OFF: `strata-gguf`, `strata-plan` and `libstrata_kernels_cpu.a`
  had identical SHA256 values in the same native build tree. DeepSeek targets
  were absent when disabled. The full GPU engine was not built by this check.
- Native test with the actual layer-0 expert-0 fixture: 25 checks, zero failures.
  It covers synthetic and real weights, clamp behavior, route weighting, zero
  route, reuse of an input address, bounded scratch and invalid inputs/layouts.
- Python conversion tests: all five passed, including all 16 FP4 codes and all
  finite E8M0 scale bytes. An independent reverse of all three real GGUF payloads
  restored every source byte and matched the cached pinned-checkpoint provenance.
- AddressSanitizer and UndefinedBehaviorSanitizer: artifact and synthetic expert
  CTests passed; the actual expert's 25-check run also passed. Debug retains the
  pre-existing `gguf_split.hpp` warning described below; WERROR was OFF there.

The numerical reference decodes the same MXFP4 weights and Q8 activations, then
uses scalar FP64 accumulation. It calls none of Strata's projection functions.
The acceptance limits, set before the real run, are relative L2 <= 0.002 and
maximum absolute error <= 1e-5 + 0.02 times the reference peak.

| Real expert test input | Route weight | Relative L2 | Maximum absolute error |
| --- | --- | --- | --- |
| Deterministic sine/cosine input | 1.0 | 0 | 0 |
| Same input, route applied before down activation quantization | 0.37 | 0 | 0 |
| Input scaled by 16, clamp branches exercised | 1.5 | 0 | 0 |

These observed zero errors apply only to these test cases and this build. They
are not a claim of bitwise parity for all inputs or other activation formats.
The real expert stores 18,800,640 weight bytes. Its generated fixture SHA256 is
`1afa4ca1e0e735694b5e2ecf21a5ed7c2581978b0fb9e0765b80f2886fb290ff`.

## Step 3b: one expert on RTX 2080 Ti

`CudaExpert` uploads one expert at construction and enqueues a one-token forward
pass on an explicit CUDA stream. It uses Strata's `DeviceArena`,
`native_quantize_q8_1` and `iq_mmvq(40)` for all three projections. The new CUDA
kernels only perform BF16 rounding and DeepSeek's clamped, route-weighted SwiGLU.
The original matrix kernels, Qwen execution and default options are unchanged.
All tensor arithmetic stays on the GPU. The adapter uses preallocated buffers
and enqueues without waiting in `forward`. The caller owns valid device inputs and stream ordering, waits before destruction and keeps the
allocation device current. The CUDA object owns its weight copy.

Strata's direct MMVQ entry does not accept MXFP4. Its existing NVFP4 entry has the
same E2M1 values with a different nibble arrangement and a scale every 16 values.
`mxfp4_to_nvfp4` rearranges the codes and repeats each scale. It accepts only E8M0
scale bytes 118..135, whose powers of two fit UE4M3 exactly without a global scale.
It rejects all other scales; it never silently re-quantizes them. The tested real
expert uses bytes 119..122 for gate/up and 119..123 for down. This proves conversion
for this expert, not coverage of every expert in the checkpoint.

The converted weights retain all values and occupy 19,906,560 bytes, versus
18,800,640 in MXFP4 (+5.8824%). Scratch is 56,480 bytes. The adapter's device arena,
including alignment, is 19,963,168 bytes. Step 4 separates the weight and scratch
arenas so a cache slot can supply weights; the combined byte count stays the same. These are component allocations, not
whole-process VRAM; they exclude CUDA modules/context and caller/test buffers.
The existing kernel decodes FP4 codes and uses integer dot products on SM75; this
step does not require native FP4 Tensor Cores or introduce a matrix kernel.

### Numerical contract and measured results

Measured on 2026-10-21, Linux, GCC 13.3.0, CUDA 12.8.93, driver 570.144,
RTX 2080 Ti SM75 (22,528 MiB reported), Xeon E5-2686 v4. Same pinned expert-0
fixture and deterministic test inputs as step 3; these are not full-model
activations. The GPU uses Q8_1 activations, while the CPU baseline uses Q8_0.

An initial reference using scalar CPU division for Q8_1 failed the preset 0.3%
relative-L2 gate. A stage-by-stage probe found five input integer codes out of
5120 differed near half-integer boundaries because the existing CUDA quantizer
uses fast division. With identical quantized inputs, gate/up matrix errors were
below 6e-8 relative L2 and their BF16 outputs matched. Replacing the GPU hidden
post-op with CPU math also gave identical final outputs. Production arithmetic
was not changed to address this test mismatch.

The corrected reference uses the GPU quantizer, separately checking its FP16
scales exactly and allowing a one-code difference from scalar division only when
the scalar quotient lies within 2e-5 of a half-integer. It then decodes the original
MXFP4 weights and accumulates matrix products in scalar FP64, without Strata's
projection functions or the converted weights. Three real cases checked 1176
Q8_1 blocks and 33 such boundary-code differences. Crafted positive/negative ties
and zero blocks are also checked without the checkpoint. Sharing the quantizer
is explicit: this is an independent matrix/post-op reference plus a separate
quantization check, not an independent implementation of CUDA fast division.

The original acceptance limits remain unchanged: relative L2 <= 0.003 and maximum
absolute error <= 1e-5 + 0.02 times the reference peak. The separate CPU/GPU check
uses relative L2 <= 0.03 and maximum error <= 1e-5 + 0.08 times the CPU peak.

| Real input / route | GPU vs same-quantization FP64 reference: relative L2 / max error | GPU vs CPU Q8_0: relative L2 | GPU vs CPU: max error |
| --- | --- | --- | --- |
| Ordinary / 1.0 | 0 / 0 | 0.0042106984 | 0.015625 |
| Ordinary / 0.37 | 0 / 0 | 0.0041810129 | 0.0078125 |
| Scaled by 16 / 1.5 | 0 / 0 | 0.0031553918 | 1 |

The observed CPU/GPU differences are about 0.32%..0.42%; these paths are not
bitwise interchangeable. Zero error against the checked reference applies only
to these inputs/build. Official FP8 parity and full-model quality remain untested.

Validation:

- SM75 build passed with `STRATA_WERROR=ON` for host C++ warnings. Existing NVCC
  warnings in the reused sources remain; the build does not make those errors.
- Feature ON/OFF in the same SM75 build: `libstrata_kernels.a`, `libstrata_core.a`
  and `strata-device` had identical SHA256 values. DeepSeek CUDA targets were absent
  with the feature OFF. The main checkout stayed clean.
- Six selected CTests passed: artifact, CPU expert, FP4 layout, CUDA expert and
  existing `native_grouped_parity` / `quantize_act_parity`.
- Real-weight CUDA test: 41 checks, zero failures, including every converted
  weight value, reference comparison, zero routing and reused input storage.
- FP4 layout test: all 256 scale bytes covered (18 exact, 238 refused), all 16
  codes preserved, ggml's two decoders agree; malformed sizes rejected. This test
  also passed under AddressSanitizer and UndefinedBehaviorSanitizer.
- Compute Sanitizer memcheck on the real-weight test: zero errors and zero leaked
  bytes/allocations.

### Reproduce step 3b

Use the previously prepared expert fixture. The current tree also links Strata's
engine for step 4, so these commands use `STRATA_WERROR=OFF` for pre-existing
engine warnings. The earlier step-3b-only validation with it ON is recorded above.
New DeepSeek code passed separate strict compilation; see step 4. From this worktree:

```sh
cmake -S . -B build-deepseek-cuda -G Ninja \
  -DSTRATA_ENABLE_DEEPSEEK41=ON -DSTRATA_DEEPSEEK41_CPU=ON \
  -DSTRATA_DEEPSEEK41_CUDA=ON -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=OFF \
  -DSTRATA_GGML_DIR="$PWD/build-deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d"
cmake --build build-deepseek-cuda --target deepseek41_expert_cuda_test \
  deepseek41_expert_test deepseek41_artifact_test deepseek41_fp4_layout_test \
  native_grouped_parity quantize_act_parity -j 4
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^(deepseek41_(expert_cuda|fp4_layout|expert|artifact)_test|native_grouped_parity|quantize_act_parity)$'
build-deepseek-cuda/src/deepseek41/deepseek41_expert_cuda_test \
  --real-expert build-deepseek-native/real-expert0-mxfp4.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 \
  build-deepseek-cuda/src/deepseek41/deepseek41_expert_cuda_test \
  --real-expert build-deepseek-native/real-expert0-mxfp4.gguf
```

The CUDA feature is isolated and opt-in. Step 3b introduced one expert only;
step 4 adds caching and routing. HIP/SYCL/Metal support, decoder layers, full
generation and performance tuning remain outside these completed steps.

## Step 2 validation record

Measured locally on 2026-10-21, Linux, GCC 13.3.0:

- Release build with warnings treated as errors: passed.
- `deepseek41_artifact_test`: 50 checks passed. These cover metadata/type errors,
  KV ownership, exact expert bytes, mixed-format slot ordering, split shards,
  short destinations, truncated payloads and malformed type maps.
- Existing `gguf_reader_test`, `gguf_split_test`, `model_family_test` and
  `gguf_geometry_test`: all passed before and after this change.
- Debug build with AddressSanitizer and UndefinedBehaviorSanitizer: test passed.
  The first Debug build with `STRATA_WERROR=ON` stopped on an existing
  `gguf_split.hpp` format-truncation warning. The sanitizer run used WERROR=OFF;
  the shared reader was not edited. The Release build above used WERROR=ON.
- CLI: core metadata generated from the pinned official config passed; a
  metadata-only file failed the default expert check; help and invalid option
  combinations returned the expected codes (6 checks).
- CLI at real 1017 expert dimensions: a sparse synthetic layer (7,219,445,760
  payload bytes) read expert 383 into an 18,800,640-byte buffer. Its independently
  calculated checksum matched, including role offsets above 4 GiB. The same
  one-layer fixture was rejected by the default full-backbone check. These were
  synthetic byte patterns, not real weights.
- Feature OFF: the new target is absent. `strata-gguf` and `strata-plan` binaries
  had identical SHA256 values with this option ON and OFF in the same build tree.
  This comparison did not build or measure the full GPU engine.

Sanitizer reproduction:

```sh
cmake -S . -B build-deepseek-sanitize -G Ninja \
  -DSTRATA_ENABLE_DEEPSEEK41=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_DEEPSEEK41_CPU=ON \
  -DSTRATA_GGML_DIR="$PWD/build-deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d" \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_WERROR=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined
cmake --build build-deepseek-sanitize --target deepseek41_artifact_test deepseek41_expert_test -j 4
ctest --test-dir build-deepseek-sanitize --output-on-failure -R '^deepseek41_(artifact|expert)_test$'
build-deepseek-sanitize/src/deepseek41/deepseek41_expert_test \
  --real-expert build-deepseek-native/real-expert0-mxfp4.gguf
```

Still not run: full-model GGUF validation, official FP8 numerical parity,
HIP/SYCL/Metal builds, full generation or performance benchmarks. The
real single-expert fixture above does not establish those results. No speed claim
follows from these checks.

## Step 4: original top-6 through Strata's expert cache

Implemented on 2026-10-22 and confirmed by the user before step 5a. This is the
**routed branch of one text token in one layer**, using real layer-0 weights.
The shared expert and final MoE BF16 cast are not part of this branch.

### Reused infrastructure and exact scope

`Mxfp4ExpertSource` implements Strata's existing `ExpertSource` interface over the
step-2 mapped `ExpertIndex`. It losslessly packs only the requested expert into
one reusable NVFP4 buffer. Its `copy_blob` overrides the base method, which uses
Qwen's global expert layout. It reports transient storage, refuses a stable-pointer
promise, and honors `note_async_read` before recycling staging memory. There is no
accumulating host copy of the whole model.

`CudaRoutedLayer` links the existing `strata_engine`. Its `ExpertCache` supplies
allocation, admission, residency lookup, H2D copy and replacement publication.
A small per-layer LRU policy chooses a victim when the configured 1..384 slots
are full. Copies and expert computation share an explicit stream; replacement is
published only after its copy finishes. Missing experts raise an error. An execution
failure invalidates the layer object rather than allowing stale cache state to be
reused. Tests cover failure before the first fill and after two computed experts.

The single-expert CUDA implementation now also accepts a caller-owned cache blob,
with one reusable scratch arena. Hits and misses run exactly the same GPU kernels.
There is no CPU expert fallback and no change to the original top-6 IDs.

Routing follows the pinned official `Gate.forward`:

1. Round the input to BF16, use Strata's BF16 GEMV for the 384 router logits.
2. On the host, compute `sqrt(softplus(logit))` with temperature 1. The correction
   bias affects selection only. Select six experts; normalize their unbiased
   scores with `sum + 1e-20`, then multiply by 1.5.
3. Compute experts on the GPU. Each route weight is applied inside SwiGLU before
   hidden BF16 rounding/activation quantization, as established in step 3b.
4. Add BF16 expert outputs into F32 in ascending expert-ID order, matching the
   official routed loop before the shared expert is added.

Exact ties select the lower ID. PyTorch does not promise that tie order; parity is
claimed for the tested nondegenerate real route, not arbitrary tied inputs. The
host router uses FP32 reductions, so normalized sums can differ from 1.5 by one
rounding step. No post-hoc correction changes the weights.

This is a synchronous correctness baseline: one 1,536-byte logit readback per
token, a wait per cache fill, and a final wait. All projections and expert math
stay on the GPU; selection and cache policy run on the CPU. It does not implement
transfer/compute overlap, GPU top-k selection, prefill or graph capture. LRU has
not been measured against a real token trace and is not claimed to be optimal.

### Real-weight evidence and memory accounting

Same RTX 2080 Ti SM75, CUDA 12.8.93, driver 570.144 and GCC 13.3.0 as step 3b.
The source is the locally cached official 1017 router and six actual experts.
The input is a BF16 random vector with seed 1017, **not a hidden state from a full
model run**. The offline fixture helper checks the pinned config/source hashes
and executes the official `Gate` class with ordinary CPU PyTorch FP32 linear math.
PyTorch is only a fixture/oracle dependency; the engine is C++/CUDA.

The selected IDs are `226, 242, 215, 137, 51, 134`. All six IDs match the official
oracle. Maximum route-weight error is `2.98023224e-08` (acceptance: `2e-6`).
All 36 source weight/scale arrays reverse exactly from the prepared MXFP4 fixture;
router weight/bias bytes also match their cached sources. The sparse fixture uses
116,838,400 disk bytes for a 7,223,404,736-byte address space. Unpopulated experts
are holes with invalid conversion scales and are explicitly rejected; the fixture
architecture is `deepseek41-routed-fixture`, not a complete model.

| Component / run | Bytes | What the number covers |
| --- | ---: | --- |
| Converted NVFP4 expert / host staging | 19,906,560 | One reusable staging blob, no temporary conversion vector per miss |
| Six-slot GPU cache | 119,439,360 | Expert weights only |
| Six-slot layer GPU allocations | 123,460,384 | Cache, router, expert scratch and local buffers including alignment |
| Two-slot GPU cache | 39,813,120 | Forces replacement within top-6 |
| Two-slot layer GPU allocations | 43,834,144 | Same components as the six-slot row |
| Cold six-slot pass: expert H2D | 119,439,360 | Six misses, six source reads |
| Repeated warm pass: additional expert H2D | 0 | Six hits, no additional source reads |
| Three two-slot passes: expert H2D | 358,318,080 | 18 misses and 16 replacements |

The six-slot cold/warm results and all three two-slot eviction/reload results are
bitwise identical. They also equal a separate sum of six owned-weight GPU expert
calls with the same route weights. That sum checks storage and dispatch; it is
not an independent official FP8 expert oracle. The single-expert numerical
contract remains the declared NVFP4/Q8_1 baseline.

Device byte counts exclude CUDA context/modules and caller/test buffers. Host
staging counts exclude mapped file pages, OS page cache and small index/policy
metadata. Those mappings do not allocate the fixture's entire logical size in RAM;
these component bounds are not a whole-process RSS or system page-cache limit.
Startup router upload is excluded from the per-forward expert traffic counters.
No throughput, latency improvement or full-model quality claim follows.

### Failures found and validation

The first real routed test found a startup ordering bug in the owned-weight expert:
pageable H2D upload could still be pending when a nonblocking execution stream
started. A diagnostic wait after construction eliminated the independent-sum
mismatch. Constructors now wait for their legacy-stream upload/cache initialization;
`CudaExpert::forward` still enqueues without a wait. No projection arithmetic or
numerical acceptance threshold changed. A separate test's exact `0.25` assertion
was corrected to allow the observed single FP32 rounding step.

Passed:

- Ten selected CTests: the six DeepSeek artifact/routing/CPU/GPU/layout tests and
  existing `expert_cache_per_layer_test`, `expert_cache_slot_guard_test`,
  `native_grouped_parity`, `quantize_act_parity`.
- Combined synthetic and real routed test: 60 checks, zero failures. It includes cold/warm/reload equality,
  changed tokens at the same input address, original-ID preservation, independent
  expert accumulation, bounded buffers, byte counters and failure handling.
- Real expert-0 CUDA regression: 41 checks, zero failures. The five Python FP4
  fixture-packing tests also pass.
- Compute Sanitizer memcheck on the real routed test: zero errors and zero leaked
  bytes/allocations. Host routing and FP4 layout tests pass ASan/UBSan.
- New DeepSeek C++ sources/tests passed strict `-Werror` compilation separately.
  Shared engine/kernel sources are unchanged.
- Feature ON/OFF: original `libstrata_core.a`, `libstrata_kernels.a` and
  `libstrata_engine.a` SHA256 values match. DeepSeek targets disappear when OFF;
  options were restored ON after verification. The main checkout remains clean.

Build limitation: linking `strata_engine` exposed pre-existing warnings in
`src/core/layer.cpp` (misleading indentation and unused variables) and
`src/core/verify.cpp` (unused variable). The initial all-library `STRATA_WERROR=ON`
build failed on `layer.cpp`; the full build passes with `STRATA_WERROR=OFF`. These
unrelated sources were not edited. The separate strict checks cover our C++
additions. An additional NVCC probe with host `-Wall -Wextra -Werror` also fails on
unchanged `gguf_reader.hpp` pointer-to-member declarations as rewritten by NVCC
(`-Wparentheses`, lines 685 and 790). Normal SM75 CUDA compilation passes; no
shared-header changes or warning suppressions were added for this probe.

Not run in step 4: all-checkpoint expert scale coverage, official FP8 expert parity,
shared expert/whole decoder, real token traces, full generation, performance
benchmarks or HIP/SYCL/Metal. Step 5a below adds the shared expert. Image-token bias
is not exposed in this text-only layer API.

### Reproduce step 4

Use the SM75 configuration above with `STRATA_WERROR=OFF` and all three DeepSeek
options ON. The fixture command refuses an existing output and missing selected
experts; it never downloads or substitutes them.

```sh
cmake --build build-deepseek-cuda --target deepseek41_routed_cuda_test \
  deepseek41_routing_test deepseek41_expert_cuda_test deepseek41_fp4_layout_test \
  expert_cache_per_layer_test expert_cache_slot_guard_test -j 4
python3 tools/deepseek41/prepare_routed_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --output build-deepseek-cuda/real-routed0.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^(deepseek41_(routing|routed_cuda|expert_cuda|fp4_layout|expert|artifact)_test|expert_cache_(per_layer|slot_guard)_test|native_grouped_parity|quantize_act_parity)$'
build-deepseek-cuda/src/deepseek41/deepseek41_routed_cuda_test \
  --real-routed build-deepseek-cuda/real-routed0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 \
  build-deepseek-cuda/src/deepseek41/deepseek41_routed_cuda_test \
  --real-routed build-deepseek-cuda/real-routed0.gguf
```

Local records are in `build-deepseek-cuda/step4-{fixture.json,ctest.log,
real-routed.log,single-expert.log,memcheck.log,strict-cpp-final.log,
default-path.log}`. The failed extra NVCC probe is in `step4-strict-final-code.log`.
These logs and weight fixtures are ignored build outputs, not committed model assets.

## Step 5a: shared expert and complete single-token text MoE

Implemented on 2026-10-22 and confirmed before step 5b. The decoder work is split
into smaller review gates. This step adds the shared FFN branch and final MoE
composition; it does not add mHC, normalization, attention or a generation loop.

### Reuse and numerical contract

`CudaSharedExpert` uses Strata's existing `DeviceArena`, `bf16_gemv_pair` for
gate/up and `bf16_gemv` for down. The original Qwen shared-expert interface adds a
scalar sigmoid gate, which the fixed 1017 shared expert does not have. The new
adapter therefore calls the existing matrix primitives directly and shares the
1017 clamped SwiGLU/BF16 post-ops with `CudaExpert`. No matrix kernel was added or
changed. The routed expert's arithmetic is unchanged.

The original shared weights use E4M3FN values with one E8M0 scale per **32x32**
weight block. `fp8_block32_to_bf16` expands them losslessly at construction. It
rejects NaNs, overflow and any value that cannot be represented exactly in BF16;
it does not silently re-quantize weights. The reader requires the explicit
`e4m3fn-e8m0-block32` encoding tag and validates all six tensor types, shapes and
payload bounds. Raw bytes are stored as I8 payloads in a dedicated test fixture,
not accepted as a complete model or a general FP8 GGUF format.

Activations follow the official FP8 format: BF16 input, groups of 32 values,
absolute maximum floored at 1e-4, an E8M0 scale rounded upward to a power of two,
and E4M3 round-to-nearest-even. A small CUDA conversion kernel restores these
quantized values into BF16 for Strata's existing GEMV. This works on SM75 without
native FP8 Tensor Cores. Matrix sums use Strata's FP32 reduction order, which is
not the official TileLang FP8 GEMM order. Gate/up outputs, the clamped SwiGLU
hidden state and down output retain the official BF16 boundaries.

`CudaMoE` runs the existing routed branch, adds one **unweighted** shared expert
output in FP32, then casts once to BF16. It is synchronous and owns fixed device
buffers. It preserves original route IDs and the cache/source failure contract;
an execution failure invalidates the instance. The asynchronous standalone shared
expert requires finite valid inputs, one caller and completion before destruction.

The complete MoE remains a declared mixed baseline: routed experts use the
previous NVFP4/Q8_1 path; the shared branch uses FP8 activations and losslessly
expanded BF16 weights. This step does **not** establish official FP8 parity for
the complete MoE or full-model quality. Python/PyTorch is used only to prepare
reference fixtures; engine execution remains C++/CUDA.

### Measured results and memory

Tested on the same RTX 2080 Ti SM75, CUDA 12.8.93, driver 570.144, GCC 13.3.0 and
Xeon E5-2686 v4. The real layer-0 fixture uses the pinned 1017 checkpoint and the
hash-checked official `Expert.forward` graph. Its independent CPU linear oracle
uses NumPy block-32 accumulation, independent FP8 quantization and PyTorch 2.9.1
for BF16 and weight decoding. It does not execute the official TileLang GEMM.
The official source SHA256 is
`e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.

All 35,389,440 expanded weight values match the independent PyTorch decoder
exactly. GPU input quantization also matches the independent oracle for all
three real cases. The acceptance limits were fixed before the run: relative
L2 <= 0.003 and maximum absolute error <= 1e-5 + 0.02 times the reference peak.

| Real shared-expert input | Relative L2 | Maximum absolute error | Reference peak |
| --- | ---: | ---: | ---: |
| BF16 random vector, seed 1017 | 0 | 0 | 14.5 |
| Same vector multiplied by 16 and rounded to BF16 | 0 | 0 | 508 |
| Zero vector | 0 | 0 | 0 |

These are deterministic test inputs, not hidden states from a full model run.
Zero observed errors apply to these cases, not arbitrary inputs or GPU reduction
orders. The complete MoE equals a separately composed
`BF16(routed_output + shared_output)` bit for bit. Six-slot cold/warm runs and
three two-slot eviction/reload runs also agree exactly. This composition check
uses the separately tested GPU branches; it is not an independent official MoE
oracle. Route IDs and weights are unchanged from step 4.

| Component | Device bytes | Scope |
| --- | ---: | --- |
| Expanded shared weights | 70,778,880 | 67.5 MiB for three projections; original FP8 plus scales is 35,424,000 bytes |
| Shared expert with scratch/alignment | 70,821,376 | One fixed arena |
| Complete MoE, six expert slots | 194,302,240 | About 185.3 MiB; routed layer, shared expert and shared output |
| Complete MoE, two expert slots | 114,676,000 | About 109.4 MiB; replacement within top-6 |

These counts exclude CUDA context/modules, caller/test buffers and mapped host
file pages. Startup expansion uses one temporary 23,592,960-byte host projection
buffer. Expert-cache H2D remains 119,439,360 bytes for the cold six-slot pass,
zero additional expert bytes on warm passes, and 358,318,080 bytes for three
two-slot passes. These traffic counters exclude startup router/shared uploads.
The BF16 expansion is a correctness baseline with a measured storage cost;
no speed or whole-model memory claim follows.

### Validation and reproduction

Passed:

- Thirteen selected CTests: eight DeepSeek tests, two original cache tests,
  `bf16_gemv_parity`, `native_grouped_parity` and `quantize_act_parity`.
- Synthetic plus real MoE test: 70 checks, zero failures. This includes FP8
  grid/midpoint/sign/floor checks, real weight decoding, three shared inputs,
  cache invariance, fixed allocations and invalid-input/failure handling.
- Existing real expert-0 CUDA test: 41 checks; real routed test: 60 checks;
  all passed after sharing the elementwise wrappers. Five Python FP4 packing
  regression tests also passed.
- CUDA Compute Sanitizer on the real MoE test: zero errors and zero leaked bytes.
- ASan/UBSan artifact and shared-weight tests passed. The host weight test covers
  every E4M3 code, every finite E8M0 power, invalid values and 32x32 scale addressing.
- Seventeen DeepSeek C++ source/test strict compilation checks passed with
  `-Wall -Wextra -Werror`. Normal SM75 CUDA compilation passed. The pre-existing
  whole-library/NVCC strict-warning limits recorded in step 4 still apply.
- Feature ON/OFF: original core/kernel/engine archive hashes match each other
  and the prior step. DeepSeek targets are absent when OFF. The option was
  restored ON; the main checkout remains clean.

No functional test failed in step 5a. Not run: official TileLang execution,
full-MoE FP8 parity, other checkpoint layers, mHC/attention, prefill, generation,
performance benchmarks or HIP/SYCL/Metal. No unrelated shared engine/kernel
source was changed.

With the existing CUDA configuration, run:

```sh
cmake --build build-deepseek-cuda --target deepseek41_shared_weights_test \
  deepseek41_moe_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_shared_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --output build-deepseek-cuda/real-shared0.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(shared_weights|moe_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_moe_cuda_test \
  --real build-deepseek-cuda/real-shared0.gguf build-deepseek-cuda/real-routed0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 \
  build-deepseek-cuda/src/deepseek41/deepseek41_moe_cuda_test \
  --real build-deepseek-cuda/real-shared0.gguf build-deepseek-cuda/real-routed0.gguf
cmake --build build-deepseek-sanitize --target deepseek41_shared_weights_test \
  deepseek41_artifact_test -j 4
ctest --test-dir build-deepseek-sanitize --output-on-failure \
  -R '^deepseek41_(shared_weights|artifact)_test$'
```

Prepare the routed fixture with the step-4 command first. The shared preparer
needs the six original `layers.0.ffn.shared_experts` weight/scale tensors, checks
the config/source hashes and refuses to overwrite an existing output. It makes
no downloads. The generated 106,358,560-byte fixture SHA256 is
`30078f7fa8f9fa2fab155e69268b9fb0e20daa93beff5ab6b4285820fd26edca`.
Local records are `build-deepseek-cuda/step5a-{fixture.json,real.log,ctest.log,
memcheck.log,single-regression.log,routed-regression.log,strict-cpp.log,
default-path.log}` and `build-deepseek-sanitize/step5a-validation.log`.
Fixtures and logs remain ignored build outputs.

## Step 5b: mHC and normalization around the MoE

Implemented on 2026-10-22 and confirmed before step 5c. `CudaMhc::forward_moe`
now runs one complete **FFN sublayer**, using the step-5a `CudaMoE`. This includes
its residual input/output and delayed coefficients. Attention and KV execution
remain separate, unimplemented review gates.

### Fixed semantics and reuse

The adapter follows the pinned official `Block.hc_mixes`, `hc_pre`, `hc_post`
and `RMSNorm` methods:

1. Carry four BF16 residual streams of width 5120. Compute the 24 mixing logits
   with the original F32 `[24, 20480]` projection, then scale them using one RMS
   statistic over **all 20480 values**, with epsilon `1e-20`.
2. Split pre/post coefficients with sigmoid, pre epsilon `1e-6` and post factor
   2. Apply row softmax plus epsilon, one column normalization, then 19 row/column
   normalization pairs to the 4x4 residual matrix. This is the fixed 20-iteration
   Sinkhorn order, not an arbitrary convergence loop.
3. Collapse the residual with the **incoming** pre coefficients, cast to BF16,
   apply weighted RMSNorm with epsilon `1e-20`, and cast to BF16 again. The new
   pre coefficients from step 2 belong to the **next** sublayer. For an actual
   decoder, the FFN's incoming pre comes from the preceding attention half.
4. Execute the existing MoE, then return
   `BF16(post[dest] * moe[d] + sum_source(comb[source,dest] * residual[source,d]))`.
   Matrix rows are sources and columns are destinations. There is no extra
   RMSNorm after this residual addition.

The projection calls `cublasSgemv` with F32 weights and pedantic math through
Strata's existing CUDA toolkit dependency. It does not round the F32 projection
to BF16 or implement a matrix kernel. Weighted sublayer normalization reuses
`native_qsa_rms_norm_weighted`; buffers and BF16 boundaries reuse Strata primitives.
The new CUDA work is the flattened RMS statistic for the mixing logits, the 24
coefficients/Sinkhorn updates and four-stream collapse/expansion. The small
Sinkhorn baseline uses one GPU thread; its speed has not been benchmarked.
Strata's Qwen `gr_read`/`gr_write` have different per-stream normalization,
bottleneck and residual rules and are not substitutes for the 1017 graph.
The delayed-pre and matrix-orientation rules also agree with the earlier
DeepInfer reference work; no DeepInfer/PyTorch runtime is linked into this engine.

`prepare` owns a BF16-rounded residual snapshot and returns read-only device views
of the normalized input, next pre, post and comb. `finish` consumes that snapshot.
A second prepare before finish, a finish without prepare, overlapping output
buffers and switching execution streams are rejected. One instance uses one
explicit stream; the caller waits before destroying it or its buffers. The full
FFN helper is synchronous, returns the original route, copies next pre on device
and propagates source failures. An execution failure invalidates the mHC instance.

`read_mhc_weights` reuses the mapped GGUF reader, with typed F32/BF16 views, exact
shapes and bounds checks. It requires the explicit `1017-hc4-delayed` profile.
Both layer-0 FFN and attention mHC parameter sets were tested, but testing the
attention **mHC parameters** does not execute an attention layer. The fixture is
marked `deepseek41-mhc-fixture`, not a complete model.

### Reference and measured results

Same RTX 2080 Ti SM75, CUDA 12.8.93, driver 570.144, GCC 13.3.0 and Xeon E5-2686 v4.
All eight raw mHC/norm tensors match the local download provenance SHA256 values
for revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The offline fixture runs the hash-checked official Python methods with an
independent Torch translation of the pinned Sinkhorn kernel. It checks both
source files; the kernel SHA256 is
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
It does not run TileLang. A separate synthetic reference uses scalar FP64
projection/statistic accumulation and explicit F32/BF16 graph boundaries.

Each real parameter set uses four deterministic residual inputs: seed-1017 BF16
random values, values multiplied by 16, values multiplied by `1e-12`, and zeros.
Incoming pre alternates between `[1,0,0,0]` and `[0.1,0.7,0.2,0.4]`. A seeded
mock sublayer output isolates mHC post behavior; these are not full-model states.
The acceptance gates were set before the run: coefficient error per element
<= `2e-6 + 2e-5 * abs(reference)`, and BF16 tensor relative L2 <= `0.003` with
maximum error <= `1e-5 + 0.02 * reference_peak`.

| Real stage, maximum across four inputs | FFN mHC | Attention mHC |
| --- | ---: | ---: |
| Next-pre maximum absolute error | 7.4505806e-09 | 3.7252903e-09 |
| Post coefficient maximum absolute error | 5.96046448e-08 | 2.98023224e-08 |
| Comb maximum absolute error | 2.79396772e-09 | 5.96046448e-08 |
| Normalized BF16 input relative L2 / max error | 0 / 0 | 0 / 0 |
| Residual output maximum relative L2 | 1.5385474e-06 | 0 |
| Residual output maximum absolute error | 0.000244140625 | 0 |

The integrated real FFN test uses the first FFN residual and incoming pre
`[1,0,0,0]`. Its reference-normalized input selects the actual experts
`226, 215, 51, 242, 136, 260`. The fixture helper now accepts an explicit BF16
input and packs those selected experts, using the existing lossless writer.
Missing experts still fail; no route is replaced by an available one. All IDs
match the official gate oracle, with maximum route-weight error `2.98023224e-08`.

The complete FFN output matches a separately composed reference exactly for this
input: official normalized input, the previously tested GPU MoE baseline, and
CPU residual recombination with official coefficient values. All six-slot
cold/warm passes and all two-slot eviction/reload passes are also bitwise equal.
The reference reuses the GPU MoE, so this validates composition and storage, not
independent official FP8 parity for the entire FFN. Routed experts still use
Q8_1 activations; full-model accuracy remains unvalidated.

| Owned device allocation | Bytes | Scope |
| --- | ---: | --- |
| One mHC adapter | 2,110,464 | Projection, scales/base, expanded F32 norm, snapshot, coefficients and scratch |
| Complete FFN with six MoE cache slots | 196,412,704 | About 187.3 MiB, including the existing MoE |
| Complete FFN with two MoE cache slots | 116,786,464 | About 111.4 MiB, including the existing MoE |

These are explicit component arenas, excluding cuBLAS-owned resources, CUDA
context/modules, caller/test buffers and host mappings. Whole-process VRAM and
cuBLAS internal allocation behavior were not profiled. Expert H2D remains
119,439,360 bytes for the cold six-slot pass, zero additional bytes when warm,
and 358,318,080 bytes across three two-slot passes. Startup uploads are excluded.
No throughput or latency claim follows.

### Validation and reproduction

Passed:

- Sixteen selected CTests: ten DeepSeek tests and six existing cache/kernel tests.
- Synthetic plus real mHC/FFN test: 151 checks, zero failures. These cover stage
  references, delayed pre, matrix orientation, tiny-input epsilon, BF16 boundaries,
  snapshot ownership, call order, invalid weights, stream/buffer errors, cache
  invariance and failure propagation. The standalone synthetic run has 44 checks.
- Real step-5a MoE regression: 70 checks, zero failures.
- CUDA memcheck on the complete real test: zero errors and zero leaked bytes.
  Racecheck on the synthetic test: zero hazards, errors or warnings.
- ASan/UBSan artifact and mHC reader tests passed. The reader has 12 checks for
  byte preservation, dtype/orientation/profile errors and truncated/missing data.
- Twenty DeepSeek C++ source/test strict compilation checks passed. Normal SM75
  CUDA compilation passed. The inherited full-build strict-warning limits still
  apply; NVCC also reports existing overloaded-virtual warnings from
  `core/expert_source.hpp` through the MoE header. Shared sources were not edited.
- The routed helper's default generated file remains byte-identical to the step-4
  fixture after adding explicit-input support. Five FP4 packing tests passed.
- Feature ON/OFF original core/kernel/engine archives match the prior validated
  hashes; DeepSeek targets disappear when OFF. The option was restored ON and
  the main checkout remains clean.

No functional test failed in this step. Not run: actual attention/KV execution,
full decoder/prefill/generation, official TileLang/full-FFN FP8 parity, other
checkpoint layers, performance benchmarks or HIP/SYCL/Metal.

Use the existing CUDA and sanitizer configurations:

```sh
cmake --build build-deepseek-cuda --target deepseek41_mhc_weights_test \
  deepseek41_mhc_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_mhc_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-mhc0.gguf \
  --routed-output build-deepseek-cuda/real-mhc-routed0.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_mhc_(weights|cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_mhc_cuda_test \
  --real build-deepseek-cuda/real-mhc0.gguf build-deepseek-cuda/real-shared0.gguf \
  build-deepseek-cuda/real-mhc-routed0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_mhc_cuda_test \
  --real build-deepseek-cuda/real-mhc0.gguf build-deepseek-cuda/real-shared0.gguf \
  build-deepseek-cuda/real-mhc-routed0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  build-deepseek-cuda/src/deepseek41/deepseek41_mhc_cuda_test
cmake --build build-deepseek-sanitize --target deepseek41_mhc_weights_test \
  deepseek41_artifact_test -j 4
ctest --test-dir build-deepseek-sanitize --output-on-failure \
  -R '^deepseek41_(mhc_weights|artifact)_test$'
```

Reuse the step-5a shared fixture. The new preparer refuses existing outputs and
makes no downloads. The mHC fixture is 5,597,152 bytes with SHA256
`0da70d4a98e0b162d802e195d63b6ad1fad015c74619a9564a71dea40f34030f`.
The new sparse routed fixture uses 116,838,400 disk bytes for a 7,223,404,736-byte
address space. Logs and metrics are `build-deepseek-cuda/step5b-*` and
`build-deepseek-sanitize/step5b-validation.log`; they remain ignored build outputs.

## Step 5c: layer-0 attention and bounded KV state

This step implements complete attention for the fixed 1017 **layer 0**, which has
no compressed KV. `CudaAttention` runs all five projections, normalization,
sliding-window attention and grouped output projection. `CudaWindowAttention`
exposes the same window core separately, so its state and arithmetic can be tested
without loading dense weights. Neither class includes mHC or the residual join;
those connections belong to step 5d.

The partial GGUF reader requires the explicit profile
`1017-layer0-swa128-fp8-block32`. It checks exact projection shapes, raw FP8/scale
byte types, BF16 norms, F32 sinks and mapped payload bounds. It refuses other
layers and compressed geometry. This fixture profile does not identify a complete
checkpoint; the offline preparer separately verifies the pinned revision, config,
official source/kernel hashes and all 13 original attention tensor hashes.

### Reuse and execution contract

- Original E4M3FN/E8M0 block-32 weights expand losslessly to BF16 at startup.
  Q/KV/O projections use Strata's existing `bf16_gemv`; q/kv normalization uses
  `native_qsa_rms_norm_weighted` with epsilon `1e-20`.
- The existing DeepSeek FP8 helper now accepts up to 32,768 activation values.
  Weight expansion accepts at most 32,768 rows, 8,192 columns and 41,943,040
  elements. Each consumer still validates its exact shapes. The shared expert's
  original shapes and arithmetic are unchanged.
- The window core uses the existing cuBLAS dependency for FP32 QK and
  probability/KV matrix products, with pedantic math and atomics disabled.
  New CUDA code only supplies the fixed model operations around those products.
- Strata's current native flash-attention entry point requires Q24/KV2 with
  head dimension 256, FP16 K/V and different softmax behavior. Its NeoX RoPE
  entry points also use different pairing. Those APIs cannot directly represent
  this model's Q64/KV1, dimension 512 and BF16/sink contract. They remain unchanged.
- A sequence begins with `forward(..., tokens, start_pos=0, stream)`. Later calls
  require one token and exactly `next_position()`. `reset(stream)` clears both
  the ring and position. Calls are synchronous, use one explicit stream per
  instance and reject gaps, rewinds, overlapping buffers and stream switches.
  Execution failure poisons the instance; validation failures preserve state.
- Buffers are allocated at construction. Prefill runs one query at a time with
  fixed scratch, even when the prefill exceeds the window. This is an accuracy
  baseline, not a prefill throughput optimization. The C++/CUDA execution path
  has no Python or Torch dependency.

### Preserved 1017 semantics

1. `wq_a -> BF16 -> q_norm -> BF16 -> FP8 -> wq_b -> BF16`, with 64 query
   heads of width 512. `wkv -> BF16 -> kv_norm -> BF16` produces one shared KV.
2. RoPE rotates **adjacent pairs in the last 64 elements**, using base 10,000
   and no YaRN for this layer. Results round to BF16. The complete post-RoPE
   KV, including the tail, then receives the official block-32 FP8 round trip.
3. The ring holds 128 BF16-restored KV vectors. Prefill sees only the causal
   window. Decode writes slot `position % 128` and reads in the official ring
   order, retaining invalid prefix slots while the ring fills. A streamed
   prefill maps its absolute candidate positions into the ring before the next
   query overwrites a slot, so it never reads a future or expired token.
4. Attention uses the **same vector as K and V**, scale `1/sqrt(512)`, and
   ordered groups of 64 candidates. Unnormalized exponential weights round to
   BF16 for the numerator; the running denominator remains FP32. The learned
   per-head sink contributes only to the denominator. Prefill and decode can
   differ slightly because their BF16 candidate groups differ; both are tested
   against their own reference order, rather than requiring identical outputs.
5. Attention output rounds to BF16 and receives inverse tail RoPE. It then
   takes a block-32 FP8 round trip. `wo_a` projects eight independent groups
   from 4,096 to 1,024 values using its losslessly expanded BF16 weights. The
   resulting 8,192 values round to BF16, pass through FP8 activation conversion
   and `wo_b`, and finally round to BF16 at hidden width 5,120.

### Measured validation

Environment remains the RTX 2080 Ti SM75, CUDA 12.8.93 and GCC 13.3 described
above. These are correctness checks; no speed comparison was run.

| Check | Result |
| --- | --- |
| Related CTest selection | 18 passed, zero failures/skips |
| New attention CUDA test with real weights | 60 checks, zero failures; includes 38 synthetic checks |
| Synthetic state coverage | 129-token prefill, decode through position 256, incremental warm-up, two ring wraparounds and reset |
| Independent scalar SWA reference | Worst tested relative L2 `2.14279345e-4`, maximum absolute error `4.8828125e-4` |
| Real 5-token prefill plus two decode calls | Isolated SWA outputs and KV caches exactly equal the official-graph CPU oracle |
| Complete real attention output | Worst relative L2 `6.91222015e-5`; maximum absolute error `0.001953125` |
| Full attention reset/replay | All three calls byte-identical on replay; zero-input output exactly zero |
| Wide FP8 activation helper | All 32,768 restored BF16 values exactly match the independent oracle |
| Weight reader and expanded FP8 dimensions | 15 CPU checks passed |
| CUDA memcheck, real + synthetic | Zero errors; zero leaked bytes/allocations |
| CUDA racecheck, synthetic | Zero hazards, errors or warnings |
| ASan/UBSan | Artifact, shared FP8 and attention weight tests all passed |
| Existing real MoE / mHC regression | 70 / 151 checks passed |
| Strict C++ and Python syntax | 23 C++ files passed `-Wall -Wextra -Werror`; all fixture scripts parse |
| Feature OFF/ON isolation | Original `strata_core`, `strata_kernels`, `strata_engine` archives match step 5b byte-for-byte; development build restored ON |

Numerical gates were set before execution: SWA relative L2 at most `0.003` and
maximum absolute error at most `1e-5 + 0.02 * reference_peak`; complete attention
uses `0.005` and `1e-5 + 0.03 * reference_peak`. All outputs must be finite and
BF16-valued. No tolerance was changed to make these results pass.

The real oracle executes the hash-pinned official `Attention.forward`,
`_window_kv`, RMSNorm, RoPE and window index functions. NumPy implements the
block-32 FP8 linear reference; an independent Torch CPU implementation follows
`inference_kernel.py`'s 64-candidate sparse-attention rules. It does not execute
TileLang. GEMV/GEMM reduction order may differ from the official GPU kernels,
so these tests establish the declared component baseline, not bitwise official
GPU parity or full-model quality. Long-context numerical behavior beyond the
257-token synthetic sequence has not been validated; the API's maximum sequence
bound is 1,048,576 positions.

### Owned device memory

| Component | Bytes |
| --- | ---: |
| Five losslessly expanded BF16 projection matrices | 253,231,104 |
| Projection norms and scratch | 374,784 |
| Complete SWA core, including the ring | 855,040 |
| KV ring alone, included above | 131,072 |
| Complete `CudaAttention` owned arenas | 254,460,928 (about 242.7 MiB) |

These counters exclude CUDA/cuBLAS context and internal workspace, the caller's
input/output buffers and CPU fixture mappings. The owned arenas stay constant
across prefill, decode, wraparound and reset. This is a single-layer allocation;
it is not a full-model VRAM estimate.

### Reproduce step 5c

Use the existing CUDA build configuration from steps 5a/5b, with
`STRATA_ENABLE_DEEPSEEK41=ON`. No model download is required.

```sh
cmake --build build-deepseek-cuda --target deepseek41_attention_weights_test \
  deepseek41_attention_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_attention_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-attention0.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_attention_(weights|cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_attention_cuda_test \
  --real build-deepseek-cuda/real-attention0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_attention_cuda_test \
  --real build-deepseek-cuda/real-attention0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  build-deepseek-cuda/src/deepseek41/deepseek41_attention_cuda_test
cmake --build build-deepseek-sanitize --target deepseek41_attention_weights_test \
  deepseek41_shared_weights_test deepseek41_artifact_test -j 4
ctest --test-dir build-deepseek-sanitize --output-on-failure \
  -R '^deepseek41_(attention_weights|shared_weights|artifact)_test$'
```

The preparer refuses existing outputs. The real attention fixture is 129,886,464
bytes with SHA256
`702982e802e0f752019271b3b2344c28cc8ca917a122049037ec2952da3fff8f`.
Logs, metrics and archive hashes are in `build-deepseek-cuda/step5c-*` and
`build-deepseek-sanitize/step5c-*`; these are ignored build artifacts.

## Step 5d: one complete native decoder layer

`CudaDecoder` now connects the confirmed components for the fixed 1017 **text
layer 0**. It owns two `CudaMhc` instances, one `CudaAttention`, one `CudaMoE`
and 102,400 bytes of joining scratch. The expert source remains borrowed and
must outlive the decoder. Dense weights, normalization, expert execution and
cache storage all use the previously validated implementations. This step adds
C++ orchestration and state checks; it adds no arithmetic kernel.

### Data flow and state

For each token, the decoder performs the official `Block.forward` order:

1. Attention mHC computes its coefficients from the four-stream input residual.
   The incoming coefficients supplied by the caller collapse this token's
   residual, then attention normalization runs.
2. Complete Q/KV/O attention updates the layer's window cache. Attention mHC
   joins that result with the saved input residual.
3. FFN mHC computes coefficients from the new residual. The **attention's newly
   computed pre coefficients** collapse this token's residual for FFN
   normalization, original routing and complete MoE.
4. FFN mHC joins the MoE output and returns the four-stream result together with
   the **FFN's pre coefficients for the next layer**.

The pre coefficients belong to the same token across layers. They are not a
recurrent state passed from one token to the next. The caller supplies
`input[t,4,5120]` and `incoming_pre[t,4]`; it receives `output[t,4,5120]` and
`next_pre[t,4]`. Optional host `Route` entries expose the original six IDs and
weights for each token.

Initial prefill can contain multiple tokens. Later calls must contain one token
at exactly `next_position()`. The implementation streams prefill through both
sublayers using fixed device scratch. A private attention entry point preserves
`min(initial_prefill_length,128)` as the candidate width for every prefill query.
It therefore retains the official prefill BF16 candidate grouping, including
when a prompt crosses 64 candidates. The standalone attention API is unchanged.

Calls require one explicit stream and mutually disjoint external buffers. All
shape, range, stream and overlap checks occur before processing tokens. A
validation error leaves the live sequence untouched. Any execution failure
poisons the whole decoder, including cases where attention has already updated
KV before an expert read fails. Outputs from that call are invalid; neither
another forward nor reset can reuse the failed instance. A successful call
publishes its completed sequence position only after all tokens finish.

`reset(stream)` clears KV and sequence position. It keeps immutable expert
weights in the existing cache and leaves cache counters cumulative. Reset/replay
and a smaller expert cache do not change the tested numerical outputs. The API
bounds sequences to 1,048,576 positions; this step validates 66 synthetic and
five real-weight positions, not the maximum context length.

### Reference and real expert coverage

`prepare_decoder_fixture.py` executes the hash-pinned official `Block.forward`
with the independent attention and Sinkhorn references from steps 5b/5c. It
captures attention residuals, the actual normalized FFN inputs, FFN post/comb
coefficients, next-layer pre coefficients, KV state and official gate routes.

The C++ test runs the already-validated native `CudaMoE` on those independent
FFN inputs, then uses a separate CPU residual join to form the expected layer
output. This is a composition reference for the current **Q8_1 routed-expert
baseline**. It does not independently establish official full-FP8 MoE parity.
Shared expert and attention activation handling remain as documented in 5a/5c.

The fixture uses three prefill tokens followed by two decode tokens. Each token
has a different four-stream residual and its own incoming pre coefficients. Its
actual gate choices require 27 original experts:

```text
16 21 35 37 48 51 62 67 83 86 95 98 136 156 177 192 212 215 218 226
227 240 249 276 296 312 317
```

Twenty were absent from the local sample. Their original weight/scale tensors
were fetched from the same pinned revision with the existing bounded HTTP Range
helper. Downloaded tensor payload was 376,012,800 bytes. Inputs, routes and
numerical tolerances were not changed to fit the existing cache. The resulting
fixture verifies 185 source tensor hashes against the recorded provenance.

The existing routed fixture preparer gains an optional `extra_experts` argument
for explicit coverage across multiple inputs. It still selects the first input's
routes with the official gate, and absent experts remain errors. Regenerating
its default fixture produced a byte-identical file to step 4.

### Measured validation

Same RTX 2080 Ti SM75, CUDA 12.8.93 and GCC 13.3 environment. No throughput or
latency benchmark was run.

| Check | Result |
| --- | --- |
| Related CTest selection | 19 passed, zero failures/skips |
| Synthetic decoder | 29 checks passed |
| Real decoder | 88 checks passed |
| Synthetic 65-token prefill + decode | Output and KV exactly equal batch-attention plus independent CPU mHC joins |
| Synthetic reset and failure paths | Replay byte-identical; rejected calls preserve live state; missing expert poisons the whole layer |
| Real layer output against composition reference | Worst relative L2 `5.77837775e-7`; maximum absolute error `0.000122070312` |
| Real next-layer coefficients | Maximum absolute error `9.31322575e-9` |
| Real KV state | Exactly equal the independent CPU attention graph for all three calls |
| Real original routes | All IDs retained; reference-input native MoE weights within `2e-6` of the official gate; composed route weights within `1e-4` of that baseline |
| Reset/replay and 6-slot vs 2-slot cache | All layer outputs and returned coefficients byte-identical |
| Standalone real attention regression | 60 checks passed |
| CUDA memcheck, synthetic and real | Both passed, zero errors and leaked bytes/allocations |
| CUDA racecheck, real decoder | Zero hazards, errors or warnings |
| Strict compilation and Python syntax | 25 C++ files passed `-Wall -Wextra -Werror`; fixture scripts parse |
| Feature OFF/ON isolation | Original core/kernels/engine archives match step 5c byte-for-byte; development configuration restored ON |

Gates were fixed before numerical execution: residual/KV relative L2 at most
`0.005` with maximum absolute error at most `1e-5 + 0.03 * reference_peak`;
next-layer coefficient error at most `2e-6 + 2e-5 * abs(reference)` per element.
Residual/KV outputs must be finite and BF16-valued. No tolerance was loosened.
CPU artifact parsing and numerical kernels were not changed in this step;
their earlier ASan/UBSan and kernel-specific checks remain recorded above.

### Owned memory and expert traffic

| Expert cache slots | Complete decoder owned device bytes | MiB | Cache hits / misses / evictions in the real reset/replay test | Expert H2D bytes |
| ---: | ---: | ---: | --- | ---: |
| 6 | 453,086,496 | 432.1 | 2 / 58 / 52 | 1,154,580,480 |
| 2 | 373,460,256 | 356.2 | 0 / 60 / 58 | 1,194,393,600 |

The real test executes five tokens twice for each capacity. H2D counts describe
that test's actual routes, not a workload estimate. Owned device allocation
stays constant across prefill, decode and reset. Figures include both mHC
components, full attention, complete MoE, expert cache and joining scratch.
They exclude CUDA/cuBLAS context and internal workspace, caller buffers and host
mappings/staging. This is one layer, not a full-model memory estimate.

### Reproduce step 5d

Reuse the step-5b mHC/shared and step-5c attention fixtures. All required original
experts must already be in the local tensor cache; the fixture preparer performs
no downloads and refuses existing outputs.

```sh
cmake --build build-deepseek-cuda --target deepseek41_decoder_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_decoder_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-decoder0.gguf \
  --routed-output build-deepseek-cuda/real-decoder-routed0.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_decoder_cuda_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_decoder_cuda_test --real \
  build-deepseek-cuda/real-decoder0.gguf build-deepseek-cuda/real-mhc0.gguf \
  build-deepseek-cuda/real-attention0.gguf build-deepseek-cuda/real-shared0.gguf \
  build-deepseek-cuda/real-decoder-routed0.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_decoder_cuda_test
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_decoder_cuda_test --real \
  build-deepseek-cuda/real-decoder0.gguf build-deepseek-cuda/real-mhc0.gguf \
  build-deepseek-cuda/real-attention0.gguf build-deepseek-cuda/real-shared0.gguf \
  build-deepseek-cuda/real-decoder-routed0.gguf
```

For the real racecheck run, use the same real command with
`--tool racecheck --error-exitcode 99`, omitting `--leak-check full`.
The decoder fixture is 1,730,784 bytes with SHA256
`3697d1bedbe5ecbb44645ba37ba1b0d982eb86a8f1392f5c6b7dc3f40e9c5ec1`.
The routed fixture uses 511,909,888 disk bytes for a 7,223,418,144-byte sparse
address space. Logs, measured metrics and archive hashes are
`build-deepseek-cuda/step5d-*`, all ignored build outputs.

## Step 6a: layer-2 ratio-2 KV compressor

Step 5d was confirmed before this work began. This step implements only the first
compressed-KV component for the fixed 1017 layer 2. It does not connect that layer's
attention or change the confirmed layer-0 decoder.

### Weight access and computation

`read_compressor_weights` reads the three original BF16 tensors through Strata's
mapped GGUF reader. It requires the partial-file profile
`1017-layer2-ratio2-bf16`, layer 2, hidden width 5120 and compression ratio 2.
Missing tensors, incorrect types/shapes, wrong profiles and truncated payloads
are rejected. The partial fixture architecture is `deepseek41-compressor-fixture`;
matching the profile does not establish full-model support or checkpoint identity.

| Checkpoint tensor after `layers.2.attn.compressor.` | Partial GGUF tensor | Logical shape | Runtime representation |
| --- | --- | --- | --- |
| `wkv.weight` | `blk.2.attn_compressor_wkv.weight` | 512 x 5120 | Original BF16 promoted losslessly to F32 |
| `wgate.weight` | `blk.2.attn_compressor_wgate.weight` | 512 x 5120 | Original BF16 promoted losslessly to F32 |
| `norm.weight` | `blk.2.attn_compressor_norm.weight` | 512 | Original BF16 promoted losslessly to F32 |

The projections use cuBLAS `Sgemv` with pedantic F32 math and atomics disabled,
matching the precision required by the official ratio-2 graph. Startup copies
finish before host staging is released or a nonblocking caller stream can use them.
There is no activation FP8/Q8 conversion or BF16 rounding of projection results.
No matrix kernel is added.

The new CUDA pooling kernel computes a separate stable two-token softmax for each
of the 512 channels. It multiplies and sums in F32, then rounds the pooled vector
to BF16 **before** normalization. Strata's existing
`native_qsa_rms_norm_weighted` supplies RMSNorm with epsilon `1e-20`; the result
is rounded to BF16. This preserves the official projection/pooling/norm boundaries.

### Sequence and ownership contract

`CudaCompressor::forward` accepts F32 device input `[tokens,5120]`, rounded to
BF16 before projection, and writes BF16-valued F32 `[groups,512]`. It supports
batch size 1, an initial prefill at position 0, then contiguous single-token decode.
The position bound is 1,048,576. Prefill is serialized through fixed scratch;
this step establishes correctness and makes no throughput claim.

For a call at position `p` with `n` tokens, the number of emitted groups is
`floor((p+n)/2) - floor(p/2)`. `CompressedBatch` reports that count and the first
represented token position, `2*floor(p/2)`, or `-1` when no group is emitted.
Following groups represent positions spaced by 2. For example, a five-token
prefill emits groups at positions 0 and 2 and keeps token 4 pending. Decode at
position 5 completes the group representing position 4.

An incomplete pair produces no output; the output pointer may then be null and
no output memory is accessed. The active pending KV and scores remain F32.
Read-only device views expose that one pending token while the next position is
odd; no view is exposed after a pair completes. Inactive slots are unspecified,
so tests compare only the active tail against the official buffers.

An instance owns its weight copies and scratch, requires one explicit device
stream and one caller, and synchronizes each public operation. Input and emitted
output buffers must be disjoint and outside its arena. Invalid arguments are
rejected before state changes. Execution failures poison the instance and its
output; reset cannot recover a failed instance. Ordinary reset discards the tail
and restarts at position 0 while retaining weights. Inputs and intermediate
arithmetic must be finite and within F32 range. CUDA device-fault recovery has
not been fault-injected in this step.

The returned latent is **before RoPE**. The indexer needs that representation.
Compressed YaRN RoPE, the group-16 FP4/E4M3 cache conversion, compressed KV storage,
index selection and shared source ownership are later review gates. The compressor
is not connected to the full model or a generation loop.

### Validation and measured memory

Measured on the RTX 2080 Ti (22,528 MiB, SM75), CUDA 12.8, driver 570.144.
The fixture uses the same pinned checkpoint revision and official source hash as
previous steps. Only three missing layer-2 tensors were fetched, with bounded
HTTP ranges; their payload totals 10,486,784 bytes. The fixture preparer verifies
all three payload hashes, the revision, config hash and official source hash.
It uses local files only and refuses to overwrite an existing fixture.

The real oracle executes the pinned official `Compressor` and `RMSNorm` classes
on CPU. Their non-quantized projections use ordinary F32 `torch.nn.Linear`, with
original BF16 checkpoint weights promoted to F32. It imports no C++ runtime code.
Eight plans make 47 calls: initial prefill lengths 13, 5, 6, 1 and 2 over the same
13 generated BF16 tokens, plus zero, very small and scaled-input cases. These are
controlled component inputs, not hidden states from a complete model run.

The synthetic scalar CPU oracle accumulates full projection rows in double and
independently calculates pooling and normalization over complete pairs. It detects
premature BF16 projection rounding, exercises different gates per channel and
scores above ordinary exponential overflow, and checks output sentinels, reset,
position/count reporting, malformed calls and fixed memory capacity.

Acceptance bounds were set before the first run. For BF16 output, relative L2
must be at most `0.003`, with maximum absolute error at most
`1e-5 + 0.02 * reference_peak`. For active F32 projections, the bounds are `1e-5`
and `2e-6 + 2e-5 * reference_peak`. They were not relaxed.

| Measurement against the official CPU graph | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Emitted normalized latent, 25 comparisons | 2.74205071e-8 | 2.38418579e-7 |
| Active pending KV, 24 comparisons | 2.18336859e-7 | 1.90734863e-6 |
| Active pending scores, 24 comparisons | 2.3897418e-7 | 1.90734863e-6 |

Native outputs are byte-identical across the tested prefill/decode partitions and
reset replays. This is component validation against an official CPU graph, not a
claim of official GPU reduction-order parity, full-model quality or faster inference.

| Check | Result |
| --- | --- |
| New weight reader | 19 checks passed; also passed under ASan/UBSan with leak detection |
| Synthetic compressor CUDA test | 160 checks passed |
| Real compressor CUDA test | 177 checks passed |
| Selected CTest suite | 21 passed, including existing cache, GEMV, activation, MoE, mHC, attention and decoder checks |
| CUDA memcheck, synthetic and real | Zero errors and zero leaked bytes/allocations |
| CUDA racecheck, real | Zero hazards, errors and warnings |
| Strict C++ and Python syntax | All 3 new C++ translation units passed `-Wall -Wextra -Werror`; fixture Python parsed |
| Feature isolation | Existing core/kernels/engine archive hashes match step 5d with the feature OFF and ON |

| Owned device allocation | Bytes |
| --- | ---: |
| Expanded projection and norm weights | 20,973,568 |
| Input, two-slot KV/scores and pooled scratch | 30,720 |
| Total | 21,004,288 (20.03125 MiB) |

Allocation remains constant across the tested calls. It excludes CUDA/cuBLAS
context and internal workspace, caller input/output buffers and host mappings or
startup staging. It does not include storage for the growing compressed KV cache,
which this component deliberately does not own.

### Reproduce step 6a

Use the existing opt-in CUDA build configuration from step 5d. The three original
layer-2 tensors and their provenance must already be in the local tensor cache.

```sh
cmake --build build-deepseek-cuda --target \
  deepseek41_compressor_weights_test deepseek41_compressor_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_compressor_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --output build-deepseek-cuda/real-compressor2.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_compressor_(weights|cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_compressor_cuda_test --real \
  build-deepseek-cuda/real-compressor2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_compressor_cuda_test
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_compressor_cuda_test --real \
  build-deepseek-cuda/real-compressor2.gguf
```

For racecheck, use the last command with `--tool racecheck --error-exitcode 99`,
omitting `--leak-check full`. The fixture is 12,112,704 bytes with SHA256
`411f00a14396108b13c2c915f6348dc05f8a69817b2f269f7b750b5463e79583`.
Logs, metrics and archive hashes are `build-deepseek-cuda/step6a-*`; the CPU
sanitizer logs are `build-deepseek-sanitize/step6a-*`. These remain ignored build
outputs. Existing engine sources and CUDA/HIP/SYCL paths are unchanged.

## Step 6b: compressed KV post-processing and bounded cache

Step 6a was confirmed before this work began. The layer-2 compressor can now feed
`postprocess_compressed_kv` and `CudaCompressedKvCache`. This step covers the
compressed KV representation and its storage. The indexer, shared ownership and
compressed attention computation are not connected yet.

### Numerical path and reuse

The post-processor accepts F32 `[groups,512]`, rounds input to BF16 and rotates
adjacent pairs in the final 64 channels. Its fixed 1017 YaRN parameters are base
160000, original context 65536, factor 16 and beta 32/1. The official correction
band is pairs 15 through 25. Each latent takes the first position of its ratio-2
group: `first_position + 2*i`. No ggml magnitude correction is applied.

Strata's existing RoPE entry points target different channel pairing, head widths
and scaling semantics. A small model-specific adapter therefore performs this
rotation. It preserves the official F32 products/additions and BF16 cast before
quantization, while leaving the caller's original pre-RoPE latent unchanged.
The future indexer can read that original latent directly.

The rotated vector is quantized independently in each 16-value group:

1. Clamp absolute maximum to at least `6 * 2^-9`.
2. Round `maximum / 6` to an E4M3 scale.
3. Divide by that rounded scale, clamp to `[-6,6]` and round to E2M1 FP4.
4. Multiply by the scale and store BF16, matching official in-place quantization.

The implementation uses Strata BF16 helpers and CUDA's E4M3/FP4 conversions.
CUDA 12.8 supplies the FP4 software path used on SM75; no FP4 Tensor Cores or new
matrix kernel are required. The experimental CUDA feature now explicitly requires
CUDA Toolkit 12.8 or later. Existing CUDA/HIP/SYCL engine sources are unchanged.

This cache contains **BF16 values after the FP4 round trip**. It does not store
packed FP4 codes. Input must be finite with absolute value at most 1024, which
keeps rotated group scales within the finite E4M3 range. The low-level transform
is asynchronous, performs no device allocation, and requires aligned, disjoint
buffers and an explicit stream. Its last allowed group position is 1,048,574.

### Cache and sequence contract

Capacity is an explicit number of groups, from 1 to 524,288, allocated at startup.
There is no default full-context allocation, ring wraparound or history eviction.
Each group occupies 512 BF16 values, or **1,024 bytes**. The data pointer exposes a
read-only prefix of `size()` groups; previous entries remain stable across appends.

`append` accepts the confirmed compressor's `CompressedBatch`. Nonempty batches
must start at position `2*size()` and fit entirely in the remaining capacity.
Empty batches require the descriptor `{0,-1}` and do not access their input.
Invalid descriptors, gaps, duplicate writes, overlaps or capacity overflow are
rejected before any data is written. A batch that fits only partly is rejected
as a whole, preserving both existing history and free slots.

The cache tracks completed groups; incomplete-token state still belongs to the
compressor. Future layer orchestration must check the expected emitted group count
against `capacity()-size()` **before advancing the compressor**. These two separate
components do not provide a joint rollback operation. The integration tests
allocate enough capacity for the entire tested sequence.

Append and reset synchronize on one explicit stream per instance. Reset clears
the full allocation and group count while retaining capacity. Execution failures
invalidate the cache and require reconstruction; reset cannot recover a failed
cache. Device-fault recovery was not fault-injected. The implementation adds no
per-group allocations or copies of the original latent.

### Validation and measured memory

Validation ran on the RTX 2080 Ti (22,528 MiB, SM75), CUDA 12.8, driver 570.144.
The local fixture verifies the same pinned model/config/kernel hashes and the
three original layer-2 compressor tensor hashes. No additional model download
was needed. It contains two complementary comparisons:

- Ten post-processing cases use the pinned official CPU frequency/RoPE functions
  and an independent NumPy nearest-even FP4/E4M3 oracle. They exercise signed zero,
  FP4 midpoint ties and their adjacent BF16 values, E4M3 scale ties, subnormal
  scales, and group positions near 64K, 128K and the 1M context limit.
- Five sequence plans execute the official CPU `Compressor` and
  `Attention._compress_kv` methods with real compressor weights, using prefill
  lengths 13, 5, 6, 1 and 2 over the same 13 generated tokens. Index selection is
  replaced with a hook that captures the unrotated latent. The C++ test composes
  `CudaCompressor` with the new cache and compares every resulting cache prefix.

All **97,280 BF16 values across 52 comparisons matched byte for byte**, including
signed zero. The tested native cache is also byte-identical across sequence
partitions and reset replays. The largest tested first position is 1,048,574;
a three-group case also covers 1,048,570, 1,048,572 and 1,048,574 together.
This validates the selected component cases against a pinned CPU graph and
independent quantization oracle. It does not establish official TileLang GPU
parity, index selection, full-model quality or inference speed.

The initial scalar synthetic oracle omitted rotation of its zero-valued tails.
Diagnostics showed only BF16 `0x8000` versus `0x0000` differences at rotated tail
positions, with zero numerical error. The oracle was corrected to preserve those
signs through rotation. GPU code and the byte-equality acceptance criterion were
unchanged; the official-graph tests already passed before that correction.

| Check | Result |
| --- | --- |
| Synthetic CUDA test | 40 checks passed, including whole-batch capacity rejection, preserved history/input, reset and malformed calls |
| Pinned reference and real compressor integration | 138 checks passed |
| Selected CTest suite | 22 passed, including prior cache, matrix, MoE, mHC, attention, decoder and compressor checks |
| CUDA memcheck, synthetic and real | Zero errors and zero leaked bytes/allocations |
| CUDA racecheck, real | Zero hazards, errors and warnings |
| Build and syntax | SM75 CUDA build passed; new C++ test passed `-Wall -Wextra -Werror`; Python fixture parsed |
| Feature isolation | Original core/kernels/engine archive hashes match step 6a with the feature OFF and ON |

The integration test allocates **8 groups / 8,192 device bytes** for the cache.
Together with the confirmed compressor, the two objects own 21,012,480 bytes.
These figures exclude caller buffers, host mappings, CUDA/cuBLAS context and
internal workspace. Cache memory is exactly `capacity_groups * 1024` bytes and
does not grow during append/reset. Packed storage and performance tuning remain
later measured changes.

### Reproduce step 6b

Use the existing opt-in CUDA build and the step-6a `real-compressor2.gguf` weight
fixture. The new fixture preparer reads local tensors only and refuses to replace
an existing output.

```sh
cmake --build build-deepseek-cuda --target deepseek41_compressed_kv_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_compressed_kv_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-compressed-kv2.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_compressed_kv_cuda_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_compressed_kv_cuda_test --real \
  build-deepseek-cuda/real-compressed-kv2.gguf build-deepseek-cuda/real-compressor2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_compressed_kv_cuda_test
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_compressed_kv_cuda_test --real \
  build-deepseek-cuda/real-compressed-kv2.gguf build-deepseek-cuda/real-compressor2.gguf
```

For racecheck, replace `--tool memcheck --leak-check full` in the real command with
`--tool racecheck`. The fixture is 636,448 bytes, SHA256
`a96af462bd76672eea964da362475de5efae9d46f16f44ef3e7b440da6c7397d`.
Logs, diagnostics, metrics and archive hashes are `build-deepseek-cuda/step6b-*`,
all ignored build outputs.

## Step 6c: layer-2 index key generation and bounded cache

The confirmed scope adds the key side of the fixed 1017 layer-2 indexer. It
consumes the original pre-RoPE latent from the compressor and generates a BF16
128-dimensional key for each completed ratio-2 group. It composes with the
confirmed compressor and compressed KV cache. Query projection, index scores,
causal top-k and shared-layer ownership remain separate review gates.

### Numerical path and Strata reuse

`read_index_key_weights` maps the original BF16 `wk` `[128,512]` and `k_norm`
`[128]` tensors. It requires the partial profile `1017-layer2-index-key128-bf16`
and the fixed layer-2 geometry; malformed shape, type, profile or bounds fail.
This partial profile does not independently establish checkpoint identity.

`CudaIndexKeyCache` uses the following path for each completed group:

1. Round the original F32 `[512]` compressor output to BF16 with Strata's
   `f32_to_bf16_bulk`.
2. Project with the existing `bf16_gemv` and round its F32 accumulation to BF16.
3. Apply the existing `native_qsa_rms_norm_weighted` with epsilon `1e-20` and
   the checkpoint's gamma, then round to BF16.
4. Rotate adjacent pairs in the final 64 channels. This uses the **same kernel
   and frequency table as step 6b**, specialized for a width of 128. Positions
   are the first token in each group: `0,2,4,...`.
5. Apply FP4/E2M1 with **group-32 E8M0 scales**, restoring the result to BF16.
   The scale is the power-of-two ceiling of the F32 multiplication
   `max(abs(x), 6*2^-126) * F32(1/6)`. It is distinct from the compressed KV's
   group-16 E4M3 scale. Both use CUDA 12.8 software conversions on SM75.

No matrix or normalization kernel is added. The model-specific post-processing
has two fixed specializations; there is no new general kernel framework. Python
and PyTorch are used only for offline reference preparation, not native execution.
The post-processing API requires finite input with magnitude at most `2^120`;
the projection/normalization path requires finite intermediates. These numerical
preconditions are not checked by copying activations back to the CPU.

### Cache and ownership contract

The key cache stores **BF16 values after an FP4 round trip**, not packed FP4.
Capacity is an explicit number of completed groups, from 1 through 524,288.
Each append must begin at position `2*size()` and fit entirely. Empty emissions
are `{0,-1}`, do not read the input and are accepted even at full capacity.
Malformed descriptors, duplicate or skipped groups, overlapping input and
capacity overflow fail before writes. The original latent is never modified.

One group of scratch is reused across the append. Allocation occurs at startup;
append/reset do not grow storage. Calls synchronize on one explicit stream and
must have a single caller. Reset clears the cache and size while retaining the
weights and allocation. Execution failures require reconstruction; device-fault
recovery was not fault-injected.

The three components do not provide a joint rollback. Future layer orchestration
must check the expected group count against **both** remaining cache capacities
before advancing the compressor. The integration test performs that preflight.
It also alternates key-first and KV-first consumption to verify that neither
consumer overwrites the shared input. This is component composition, not the
cross-layer `shared_attn` ownership wiring.

### Validation and measured memory

Validation ran on the RTX 2080 Ti (22,528 MiB, SM75), CUDA 12.8, driver 570.144.
Only the two new tensors were fetched by bounded byte ranges from pinned shard
`model-00005-of-00048.safetensors`: **131,328 payload bytes**. The fixture verifies
the pinned config/model/kernel hashes and all five source tensor hashes,
including the three existing compressor tensors.

The offline reference executes the official `Compressor` and the exact
`Indexer.forward` prefix up to key-cache publication. The query and selection
code is excluded. RoPE uses the official CPU functions; E8M0/FP4 rounding uses
an independent NumPy implementation with nearest-even FP4 and power-of-two scales.
Projection and normalized intermediates are retained in the fixture for diagnosis.

| Comparison | Result |
| --- | --- |
| 12 post-processing cases, including signed zero, FP4 ties, E8M0 transitions, subnormals and large values | 4,736 BF16 values matched byte for byte |
| 5 prefill/decode plans, 43 calls with prefill lengths 13, 5, 6, 1 and 2 | All 42 nonempty key-cache prefixes matched: 20,992 BF16 values |
| Compressed KV prefixes in the same integration | All 83,968 BF16 values matched the unchanged step-6b fixture |
| Reset and sequence partitions | Identical final key-cache bytes across all five plans |
| Largest tested group position | 1,048,574, including a four-group edge case ending there |

The initial fixture preparation hit a Python integer-conversion error on the
`2**117` edge-case literal. Changing it to `2.0**117` made the intended finite F32
case representable to PyTorch. No GPU math or comparison threshold changed.

| Check | Result |
| --- | --- |
| New weight loader | 17 checks passed; ASan/UBSan with leak detection passed |
| New synthetic CUDA test | 46 checks passed, including format boundaries and rejected-call state preservation |
| Pinned reference and dual-cache integration | 185 checks passed |
| Step-6b regression after sharing the kernel | 40 synthetic and 138 real checks passed |
| Selected CTest suite | 24 passed |
| CUDA memcheck, synthetic and real | Zero errors and leaked bytes/allocations |
| CUDA racecheck, real | Zero hazards, errors or warnings |
| Build and syntax | SM75 build passed; four new C++ units passed `-Wall -Wextra -Werror`; Python AST parsed |
| Feature isolation | Original core/kernels/engine archives match step 6b with the feature OFF and ON |

The key object owns **133,120 fixed bytes**: 131,072 for BF16 projection weights,
512 for promoted F32 gamma, 1,024 for BF16 input scratch and 512 for F32 key
scratch. Cache storage adds **256 bytes per group**. At the tested capacity of
8 groups, the object owns **135,168 bytes (132 KiB)**. Combined with the compressor
and 8-group compressed KV cache, the three objects own **21,147,648 bytes**.
These figures exclude caller buffers, host mappings and CUDA/cuBLAS context or
internal workspace. They are allocation measurements, not speed measurements.

These component checks do not establish official TileLang GPU parity, attention
selection correctness, model quality or generation speed. The runtime currently
processes projected keys one group at a time; batched projection and packed cache
storage are later optimizations requiring their own measurements.

### Reproduce step 6c

Use the opt-in SM75 build and the step-6a/6b fixtures described above. The preparer
reads local checkpoint data only and refuses to overwrite an existing fixture.

```sh
cmake --build build-deepseek-cuda --target \
  deepseek41_index_key_weights_test deepseek41_index_key_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_index_key_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-index-key2.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_index_key_(weights|cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_index_key_cuda_test --real \
  build-deepseek-cuda/real-index-key2.gguf \
  build-deepseek-cuda/real-compressor2.gguf build-deepseek-cuda/real-compressed-kv2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_index_key_cuda_test --real \
  build-deepseek-cuda/real-index-key2.gguf \
  build-deepseek-cuda/real-compressor2.gguf build-deepseek-cuda/real-compressed-kv2.gguf
```

For racecheck, replace `--tool memcheck --leak-check full` with `--tool racecheck`.
Omit `--real` and its three paths to run synthetic cases under memcheck. The new
fixture is 590,784 bytes, SHA256
`08674eeee485211cc61de6e27f8e8a8e6ecaeddf88596d56a68dd8ecd523143e`.
Logs and metrics are `build-deepseek-cuda/step6c-*`; CPU sanitizer outputs are
`build-deepseek-sanitize/step6c-*`. All are ignored build outputs.

## Step 6d: layer-2 index query, scores and causal selection

The confirmed scope completes the query and selection side of the fixed 1017
layer-2 indexer. `CudaIndexQuery` consumes the key cache from step 6c and returns
compressed positions for one query at a time. Its inputs are the attention
input `[5120]` and the normalized low-rank query `[1280]`. It does not produce
that low-rank query itself or execute attention over the selected values.
Candidate-block filtering belongs to later layers starting at layer 20 and is
not part of this layer-2 implementation.

### Reuse and numerical contract

`read_index_query_weights` maps three original tensors under the explicit
`1017-layer2-index32x128-fp8-block32` profile: `wq_b` E4M3 `[4096,1280]`, its E8M0
`[128,40]` scale grid, and BF16 `weights_proj` `[32,5120]`. It checks fixed layer-2
geometry, tensor shapes, types and bounds. The FP8 projection uses the existing
lossless BF16 expansion; the runtime rejects nonfinite BF16 head weights.
The partial profile alone does not prove checkpoint identity.

The runtime reuses Strata's BF16 GEMV for query projection, head weights and
query/key dot products. It reuses the confirmed block-32 FP8 activation helper
and the compressed YaRN RoPE/FP4 implementation. All 32 query heads rotate at
**the query token's position**, including odd positions. Key groups still rotate
at the first token in their completed pair; the shared kernel now accepts these
two fixed position strides without changing their frequency or rounding math.

The score follows the official dtype boundaries:

1. FP8-quantize the BF16 low-rank input, project to 32 heads of width 128, round
   to BF16, apply RoPE, then group-32 FP4/E8M0 and restore to BF16.
2. Project the BF16 attention input to 32 BF16 head weights; multiply by `1/64`
   and round to BF16. This equals `128^-0.5 * 32^-0.5` in the source.
3. Compute each head's dot products, round to BF16, apply ReLU, multiply by its
   head weight and round to BF16. Sum those products in F32 and round to BF16.
4. Mask groups `j >= floor((position+1)/2)` to negative infinity.

One F32 dot-product buffer is reused across heads, avoiding a `[32,capacity]`
scratch allocation. This is a correctness baseline with separate GEMV launches;
no throughput improvement is claimed.

Strata's existing Qwen QSA selection includes pooled-block cell counts and a
forced tail-block preference. Those rules do not implement this model's flat
compressed-group selection. The new path therefore uses CUDA's **CUB radix
sorting**, with a small model-specific score accumulation/mask/offset adapter.
No new matrix kernel or general sorting implementation is added.

### Selection, ties and bounded storage

For a prefill, every query receives the whole prefill's completed key count;
causal masking prevents early queries from seeing later groups. Decode normally
passes exactly the number of completed groups. The result has
`min(512,key_count)` entries. Valid original group IDs are sorted by position and
shifted by the caller's nonnegative offset; unreachable entries become `-1`.
A zero-key first token returns an empty selection.

CUB's stable descending score sort uses ascending original IDs as its input.
At a cutoff tie, **the lower group ID wins**. The selected IDs are then sorted
into position order. Torch's `topk(sorted=False)` does not specify which tied
indices it returns. The test checks both this deterministic policy and that the
selected scores match Torch's top-k, rather than requiring arbitrary tied IDs
to be identical. This may choose different downstream attention values at an
exact score tie; full-model parity and quality remain separate validation gates.

Capacity is fixed at construction, from 1 through 524,288 groups. Weight storage,
one-query scratch, score/ID arrays and CUB workspace are allocated at startup.
Each call checks actual workspace requirements before modifying outputs and
performs no device allocation. It rejects missing completed groups, capacity or
position overflow, bad pointers, overlapping owned buffers and overflowing
output offsets before execution. Inputs and the external key history are read-only.
Calls use one explicit stream and synchronize before returning. The object has
no sequence-position state; it can evaluate individual prefill or decode queries
against the caller's current cache. Rejected arguments preserve the prior result;
execution failures require reconstruction. Fault recovery was not injected.

### Validation and measured memory

Tests ran on the RTX 2080 Ti (22,528 MiB, SM75), CUDA 12.8, driver 570.144.
Three new tensors were fetched by bounded ranges from the fixed revision,
**5,575,680 payload bytes**. The fixture verifies config/model/kernel hashes and
eight tensor hashes: three query, two key and three compressor tensors.

The offline fixture executes the pinned official `Indexer.forward` method with
CPU BF16 head projection, the existing independent FP8 linear oracle and the
independent FP4 oracle. Generated hidden/low-rank activations exercise a 13-row
prefill and eight one-query decode cases. The C++ test generates the real key
cache from the supplied latents, then evaluates each query against that cache.
This complements, rather than replaces, step 6c's compressor/cache integration.

| Comparison | Result |
| --- | --- |
| 21 quantized queries | 86,016 BF16 values matched byte for byte |
| Scaled head weights and masked scores | All 672 weights and 3,022 scores matched numerically exactly |
| Returned index lists | 20/21 matched Torch exactly; the all-zero-score case used different tied IDs with identical selected scores |
| Deterministic low-ID tie policy and causal selection | All 21 queries matched the independent canonical selection |
| Six query RoPE cases, positions 0, 1, 65,535, 65,536, 131,071 and 1,048,575 | 24,576 BF16 values matched byte for byte |
| Real key caches, 6 and 640 groups | 82,688 BF16 values matched byte for byte; query evaluation preserved history |

Synthetic tests also exercise negative head weights, ReLU, future-group masks,
all-zero ties, invalid calls, safe/overflowing offsets, and the CUB multi-block
sort with **8,192 groups**. A larger-capacity object also processes a 513-group
prefix, verifying that the actual query size can shrink without reallocating.
The runtime's full 1M-context score scan was not tested; the largest query RoPE
position was tested separately as shown above.

| Check | Result |
| --- | --- |
| New weight loader | 19 checks passed; ASan/UBSan and leak detection passed |
| Synthetic query/selection CUDA test | 58 checks passed |
| Pinned reference and real key-cache integration | 126 checks passed |
| Selected CTest suite | 26 passed |
| Shared RoPE regression | Step-6b real 138 checks and step-6c real 185 checks passed; synthetic cases also passed in CTest |
| CUDA memcheck, synthetic and real | Zero errors and leaked bytes/allocations |
| CUDA racecheck, synthetic and real | Zero hazards, errors or warnings, including multi-block CUB sorting |
| Build and syntax | SM75 CUDA build passed; three new C++ units passed `-Wall -Wextra -Werror`; Python AST parsed |
| Feature isolation | Original core/kernels/engine archives match step 6c with the feature OFF and ON |

The initial fixture used GGML I32 index tensors, which Strata's mapped reader
does not support. Test indices now use exact F32 integer values, leaving the
original reader unchanged. This was a fixture format correction; inference
arithmetic and comparison criteria were unchanged.

The query object owns **10,850,560 fixed bytes** for expanded BF16 weights,
one-query buffers and selected IDs. It adds four aligned arrays of
`capacity_groups * 4` bytes and the aligned CUB workspace. On the tested card:

| Capacity | Query object device bytes | CUB requested workspace |
| --- | --- | --- |
| 640 groups | 10,861,056 | 1 byte; arena rounds allocation to 256 bytes |
| 8,192 groups | 11,052,800 | 71,167 bytes; arena rounds to 71,168 bytes |

These figures exclude the external key cache, caller buffers, host mappings and
CUDA context. The tested 640-group key cache separately owns 296,960 bytes.
Workspace requirements depend on CUB/device policy. Python/Torch is offline only.
No end-to-end generation, speed or official TileLang GPU parity claim is made.

### Reproduce step 6d

The fixture preparer reads local cached tensors only and refuses to overwrite an
existing output. Use the opt-in SM75 build described in the earlier steps.

```sh
cmake --build build-deepseek-cuda --target \
  deepseek41_index_query_weights_test deepseek41_index_query_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_index_query_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-index-query2.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_index_query_(weights|cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_index_query_cuda_test --real \
  build-deepseek-cuda/real-index-query2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_index_query_cuda_test --real \
  build-deepseek-cuda/real-index-query2.gguf
```

For the synthetic cases, omit `--real` and its fixture path. For racecheck,
replace `--tool memcheck --leak-check full` with `--tool racecheck`. The fixture is
8,003,264 bytes, SHA256
`0b9358e02536c70baefee147c8de8badb575b006e79c6a58b2664bbc51abf6d3`.
Logs/metrics/archive hashes are `build-deepseek-cuda/step6d-*`; CPU sanitizer
outputs are `build-deepseek-sanitize/step6d-*`. All are ignored build outputs.

## Step 6e: layer-2 attention over window and selected compressed KV

`CudaCompressedAttention` implements the fixed 1017 layer-2 attention core.
Inputs are projected query `[tokens,64,512]`, normalized pre-RoPE window KV
`[tokens,512]`, and the existing compressed BF16 cache plus indexer's selected
IDs. Output is `[tokens,64,512]` after inverse RoPE. This step includes the
6a→6b/6c→6d→6e component integration test. Main Q/KV projections, output
projections and their complete layer-2 orchestration are the next review gate.

### Numerical path and reuse

The implementation shares the existing attention translation unit's cuBLAS
handle setup, FP32 matrix products, ring-slot mapping and BF16 helpers. It uses
Strata's `DeviceArena` and the existing FP8 KV quantizer. The compressed-layer
frequency table is shared with the confirmed KV/index-key/index-query path.
Only joining/gathering the two candidate sources, index validation and the
model's chunked softmax accumulation are added. No matrix kernel, general
allocator, Python inference loop or replacement engine is introduced.

The computation follows the pinned source and sparse-attention kernel:

1. Round Q and normalized window KV to BF16. Rotate their adjacent final 64
   channels with compressed-layer YaRN: base 160000, original context 65536,
   factor 16, beta 32/1. Window KV uses this rotation too; pure-SWA base 10000
   would be incorrect here. Quantize window KV with group-32 FP8/E8M0 and write
   its BF16 reconstruction to the existing 128-slot ring layout.
2. Gather window candidates followed immediately by selected compressed KV.
   Initial prefill uses `min(tokens,128)` window candidates and compressed IDs
   shifted by `tokens`. Decode uses 128 window candidates and offset 128. Invalid
   entries are `-1`. Pad only the end of the joined list to a multiple of 64.
3. Compute FP32 QK products and apply `512**-0.5`. Each 64-candidate chunk updates
   the running maximum and denominator. Probabilities stay FP32 for the
   denominator and round to BF16 for the PV numerator. Rescale prior numerator
   chunks in FP32 when the running maximum changes.
4. Add the head's sink to the common denominator, round the normalized result
   to BF16, and apply inverse compressed-layer RoPE. Compressed KV arrives
   already rotated and quantized by step 6b and is left unchanged.

Two separately normalized attention results cannot reproduce this operation.
Even padding the window to 128 before appending compressed candidates changes
BF16 probability rounding for short prefill. A constructed two-token test makes
one output channel cancel exactly only when both sources occupy the same
64-candidate chunk. It supplements the numerical error gate with a precise
check of this layout rule.

### Sequence, selection and ownership

The core accepts initial prefill at zero followed by contiguous single-token
decode, up to 1,048,576 positions. It owns the window ring and fixed scratch,
uses one explicit stream and caller, and synchronizes before returning.
`reset` clears the ring and position without reallocating. As in earlier steps,
execution failures require reconstruction; fault recovery was not injected.

The external compressed cache must contain `(start_pos + tokens) / 2` completed
groups. Each query supplies at most 512 shifted, strictly increasing group IDs,
followed by `-1` padding. The core permits a supplied subset; step 6d normally
supplies `min(512,groups)` slots. For prefill, future groups may exist in storage
but may not appear in an early query's selection. The core validates every
query's IDs before changing any output, window history or sequence position.
Only a four-byte validation result is copied to the host. Invalid IDs, offsets,
counts, aliases and sequence/stream changes are rejected. Caller inputs and
external caches remain read-only, including after rejected calls.

This is one component's failure contract. Coordinating compressor/cache capacity
checks and failure handling across the complete layer remains orchestration
work; there is no joint rollback of independently advanced components yet.
Prefill runs one query at a time internally. Caller Q/output arrays still scale
with the requested prefill size, and are excluded from owned-memory figures.

### Validation and measured memory

Validation ran on the RTX 2080 Ti (22,528 MiB, SM75), CUDA 12.8 and driver
570.144. Only the layer-2 sink needed downloading in this step: 64 original F32
values, **256 payload bytes**, from the fixed revision. The offline fixture
verifies the config/model/kernel hashes and nine source tensors: sink, three
compressor tensors, two index-key tensors and three index-query tensors.

The fixture executes official `Attention.forward` through inverse RoPE, with
supplied projected Q and normalized window KV. It retains official `_window_kv`.
For integration it also retains `_compress_kv`, `Compressor` and `Indexer` with
original weights. Independent CPU FP8/FP4 implementations and the existing
64-candidate sparse-attention oracle stand in for the GPU kernels. This is
numerical comparison with the pinned CPU graph, not TileLang GPU bit parity.

Six plans use prefill sizes 1, 13, 43, 44, 65 and 129 and continue through token
130. They cover empty compressed history, future-group masking, sparse subsets,
64-candidate boundaries, short-prefill offsets, early decode's invalid ring
slots and window wraparound. The 41 sampled attention comparisons passed the
unchanged gate: relative L2 <=0.003 and max absolute error <=`1e-5+0.02*peak`.
The largest observed relative L2 was **6.32665704e-5**; the largest absolute error
was **0.00390625**. All six final window rings matched byte for byte.

The real-weight integration sequence uses 13-token prefill and two decode calls,
covering one new compressed group and one incomplete group. Both caches and all
15 index lists match the CPU reference exactly. The prefill attention relative
L2 was **5.18329965e-6**, max absolute **0.0009765625**. Both decode attention
outputs matched numerically exactly; all three window snapshots matched bytes.

Six isolated RoPE positions test forward and inverse rotation at 0, 1, 65,535,
65,536, 131,071 and 1,048,575. Of 393,216 output values, 393,215 matched
numerically exactly. One inverse value at position 65,536 differed by one BF16
ULP (`5.960464477539063e-8`). A scalar diagnostic found identical frequencies
and angles, but a one-FP32-ULP CPU/CUDA sine difference exposed by cancellation.
The isolated rotation gate therefore requires exact unrotated channels and at
most one BF16 ULP in rotated channels. Attention's established error gate is
unchanged. This is not a full-context attention execution test.

| Check | Result |
| --- | --- |
| Synthetic core and state tests | 71 checks passed, including 512 selections from 513 groups and exact short-prefill chunk cancellation |
| Pinned reference and 6a–6e integration | 95 checks passed |
| Selected CTest suite | 27 passed |
| Existing layer-0 attention and 6b/6c/6d real fixtures | 60 / 138 / 185 / 126 checks passed |
| CUDA memcheck, synthetic and real | Zero errors and leaked bytes/allocations |
| CUDA racecheck, synthetic and real | Zero hazards, errors or warnings with bounded detector resource settings below |
| Build and syntax | SM75 CUDA build; new C++ test with `-Wall -Wextra -Werror`; Python AST parsed |
| Feature isolation | Core/kernels/engine archives match step 6d with the feature OFF and ON; feature restored ON |

The initial racecheck with unrestricted launch tracking ended with exit 9 before
producing a valid completion result. The session cgroup recorded an OOM kill
and a roughly 62 GB peak; kernel logs were unavailable, so the killed PID was
not independently confirmed. The rerun limits the detector to two CPU workers
and forces stream synchronization after at most 16 launches. Both complete test
cases passed without filtering or skipping any kernels. These settings constrain
the detector's resource use and do not change the engine implementation. The
unrestricted run is retained as an unsuccessful check, not counted as a pass.

The core owns **2,021,120 device bytes** (about 1.93 MiB), including a
**131,072-byte** window ring. This stays constant across the tested history and
selection sizes. It excludes the caller's Q/KV/output/ID buffers, external
compressed/key caches, projections, cuBLAS and CUDA context. Compressed cache
storage remains the step-6b BF16 reconstruction; this step does not add packed
FP4 storage or an eviction policy. No throughput or full-model quality claim is
made. Python/Torch is used only for offline fixtures.

### Reproduce step 6e

Use the existing opt-in SM75 build and local pinned source/weight cache. The
fixture preparer refuses to overwrite an existing output.

```sh
cmake --build build-deepseek-cuda --target deepseek41_compressed_attention_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_compressed_attention_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-compressed-attention2.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_compressed_attention_cuda_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_compressed_attention_cuda_test --real \
  build-deepseek-cuda/real-compressed-attention2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_compressed_attention_cuda_test --real \
  build-deepseek-cuda/real-compressed-attention2.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  --force-synchronization-limit 16 --racecheck-num-workers 2 \
  build-deepseek-cuda/src/deepseek41/deepseek41_compressed_attention_cuda_test --real \
  build-deepseek-cuda/real-compressed-attention2.gguf
```

For synthetic cases omit `--real` and its fixture path. The fixture is
50,023,360 bytes, SHA256
`3a3841188bf4421c43ec884bd08bafb0de4938e7b037ac50ac8badb5ba24d018`.
Logs, numerical diagnostics, metrics and archive hashes are
`build-deepseek-cuda/step6e-*`; all are ignored build outputs.

## Step 6f: complete layer-2 attention

`CudaLayer2Attention` implements the pinned layer-2 `Attention.forward` for batch
1. Input and output are F32 device arrays `[tokens,5120]` with the graph's BF16
rounding boundaries. It accepts initial prefill followed by contiguous one-token
decode, with a caller-selected capacity of 1–524288 compressed groups and a
1048576-position limit. This completes attention only. mHC, MoE, residual joins,
shared ownership across layers and full-model execution are outside this class.

### Reuse and execution

The layer-0 Q/KV and grouped output projections are extracted into one private
`AttentionProjections` component. The projection method bodies are unchanged.
Both layers use Strata's existing BF16 GEMV, weighted RMS normalization,
`DeviceArena`, and the confirmed FP8 conversion helpers. The loader now accepts
explicit layer-0 or layer-2 profiles and records the layer in the weight view;
the two attention wrappers reject a weight view for the wrong layer. There are
no new CUDA kernels in step 6f.

The new wrapper owns the confirmed ratio-2 compressor, compressed KV cache,
index-key cache, query/index selector and compressed-attention core. It performs
prefill in two passes:

1. Read each input token to form compressor groups. Send each original unrotated
   latent to both the key and compressed-KV paths before reusing its scratch.
2. Read each input token again to project Q/KV, select causal compressed groups,
   run the joint attention core and project its output. Pass the whole prefill's
   key count, the original `min(tokens,128)` window width and `tokens` offset;
   decode uses the 128-slot ring and offset 128.

Only the owned indexer can use the core's private one-query entry point. The
public core still validates externally supplied indices. This retains the
original 64-candidate grouping for short prefill without allocating a full
`[tokens,64,512]` query array. Scratch is fixed at construction; caller input and
output arrays still scale with token count. Execution remains synchronous on
one explicit stream, and the existing component synchronizations remain. No
throughput improvement or prefill batching optimization is claimed.

Sequence, stream, pointer alias and all three cache capacities are checked before
any component advances. A full cache can accept one token of an incomplete
ratio-2 group, but rejects the next token that would emit another group. Rejected
arguments preserve the output, all three histories and the pending compressor
group. Reset clears window/compressed/key caches and the partial group. Execution
failures poison the entire wrapper and require reconstruction; this is not a
transactional rollback for failed GPU execution. Component finite-value and
intermediate-range preconditions still apply.

### Fixed source and reference

The fixture adds the original layer-2 main projections and norms to the nine
source tensors from step 6e: 21 tensors verified against the same pinned revision
and per-tensor hashes. The five main matrices are checkpoint FP8/E4M3 with
block-32 E8M0 scales. All expand losslessly to BF16. This includes `wo_a`, which
the official conversion also expands for the grouped output projection.

`prepare_layer2_attention_fixture.py` executes the official full
`Attention.forward`, window/compression methods, `Indexer.forward`, compressor,
RMS normalization and RoPE. All projections use original weights. As in earlier
steps, independent CPU FP8/FP4 arithmetic and the explicit 64-candidate sparse
attention oracle replace GPU-only kernels. This tests the declared numerical
path; it does not establish official TileLang bit parity or model quality.

Five prefill/decode plans use the same seeded BF16 inputs: prefill 1, 5, 6 or 13
followed by single-token decode through position 14, and prefill 129 followed by
positions 129 and 130. There are 42 reference calls and 191 evaluated token rows.
Every call checks output, ring, completed compressed KV and index keys. The four
short plans produce byte-identical GPU histories after the same 15 tokens;
reset/replay of the first plan is byte-identical in both output and history.
Output equality across different partitions is not required because short
prefill and decode have different 64-candidate layouts.

| Comparison | Calls | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Full attention output | 42 | 0.00304962996 | 0.0107421875 |
| Window ring | 42 | 0 | 0 |
| Compressed KV | 41 | 0 | 0 |
| Index keys | 41 | 0 | 0 |

The full-output gate remains the step-5c gate: relative L2 <=0.005 and maximum
absolute error <=`1e-5 + 0.03 * reference_peak`, with finite BF16-valued output.
The two compressed-cache gates remain <=0.003 and `1e-5 + 0.02 * reference_peak`.
No gate was relaxed. All cache values in this fixture matched numerically; the
partition/replay checks additionally compare raw bytes. The largest relative
output error occurs at decode position 130; its absolute error is 0.001953125
and reference peak is 0.458984375. Maxima in the table need not occur in one call.
This integrated fixture reaches 65 completed groups; the existing step-6d/6e
component tests cover the 512-selection boundary separately.

### Measured allocation and checks

On the RTX 2080 Ti / SM75 with CUDA 12.8 and Torch 2.9.1 CPU reference:

| Allocation | Bytes | MiB |
| --- | ---: | ---: |
| Five main matrices expanded to BF16 | 253231104 | 241.5 |
| Whole wrapper at capacity 3 | 287622144 | 274.298 |
| Whole wrapper at capacity 66 | 287703808 | 274.376 |

The wrapper totals include all owned weights, caches, scratch and the indexer's
preallocated CUB workspace. They exclude the CUDA context, cuBLAS allocations,
caller buffers and test instrumentation. The explicit extra wrapper scratch is
one 512-element F32 latent (2048 bytes). Allocation stays constant through
prefill, decode, rejection and reset. These are correctness-stage allocations;
FP8 weight residency and faster prefill remain later optimization work.

| Check | Result |
| --- | --- |
| Sparse nonzero synthetic weights and state contract | 37 checks passed |
| Synthetic plus pinned full-attention reference | 251 checks passed |
| CPU weight profiles, shapes and rejection paths | 26 checks passed |
| Relevant DeepSeek and existing Strata regressions | 28/28 CTests passed |
| Layer-0 real attention after shared projection extraction | 60 checks passed |
| Step-6e real core/integration regression | 95 checks passed |
| Changed C++ units with `-Wall -Wextra -Werror` | Passed |
| CPU weight reader ASan/UBSan | 26 checks passed; no diagnostics |
| Full synthetic/reference memcheck | 0 errors; 0 leaked bytes |
| Full synthetic/reference racecheck | 0 errors; 0 warnings/hazards |
| Default library feature isolation | Byte-identical before/OFF/ON, matching step 6e |

State tests cover capacity rejection before initial prefill and at the next
emitting decode token, full-capacity pending groups, bad sequence/stream/buffer
arguments, read-only cache aliases, reset after an odd prefill, zero input after
reset, and exact replay after rejected calls. The default Strata core, kernel
and engine archive hashes remain those recorded in step 6e. The feature stays
opt-in; the main checkout and DeepInfer code are unchanged.

### Reproduce step 6f

Use the existing opt-in SM75 build and local pinned source/weight cache. The
fixture preparer refuses to overwrite an existing file.

```sh
cmake --build build-deepseek-cuda --target deepseek41_layer2_attention_cuda_test \
  deepseek41_attention_weights_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_attention_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-layer2-attention.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(layer2_attention_cuda|attention_weights)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_layer2_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_layer2_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  --force-synchronization-limit 16 --racecheck-num-workers 2 \
  build-deepseek-cuda/src/deepseek41/deepseek41_layer2_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf
```

The `--real` invocation also runs synthetic checks. Racecheck uses the bounded
resource settings established in step 6e, without kernel filters. The fixture
is 152231008 bytes, SHA256
`2f0f73fff0b88f50c0b923855a7df011b8300bd0663b8884e50b06c8cc6f6ee2`.
Generated fixtures, logs, error metrics and archive hashes are ignored
`build-deepseek-cuda/step6f-*` artifacts.

## Step 6g: layer-2 source and layer-3 consumer

`CudaLayer3Attention` executes the fixed 1017 layer-3 attention graph. It owns
its Q/KV/output projections, norms, sink and 128-slot window ring. It reads the
compressed KV and selected indices produced by `CudaLayer2Attention`; it has no
compressor, indexer or second compressed/index-key cache. The fixed config names
layer 2 as both KV and index source for layer 3, with ratio 2 and no candidate
blocks. The new layer-3 weight profile validates this topology explicitly.

This step reuses the existing `AttentionProjections`, compressed-attention core,
Strata BF16 GEMV/RMS normalization, cuBLAS and `DeviceArena`. The projection method
bodies and compressed core implementation are unchanged. No CUDA arithmetic
kernel or Python runtime inference path is added. The implementation is specific
to this source/consumer pair; later sources, candidate-block selection, Engram,
mHC/MoE/residual decoder assembly and text generation remain separate work.

### Bounded publication and lifetime

The source constructor gains an optional `shared_prefill_capacity` after the
device argument. Zero keeps the step-6f allocation and disables publication.
A positive capacity `P` reserves an aligned GPU index buffer at construction:

`align256(4 * P * min(512, compressed_group_capacity))` bytes.

The source copies every query's selected indices into that buffer during its
existing prefill/decode loop. It retains the whole latest call, including causal
`-1` padding, the actual row width and the original concatenation offset. Merely
retaining the indexer's final query result would be incorrect for prefill.
Compressed KV is borrowed in place; it is not copied. Initial prefill exceeding
`P` is rejected before output or any component state changes. There is no
per-call device allocation. Large prefill index storage must be budgeted by the
caller; this step does not introduce chunked prefill or a memory planner.

`shared_batch()` returns an opaque `SharedAttentionBatch` only after a completed
source call. The handle carries a weak source identity and publication version.
It does not extend GPU storage lifetime. The consumer validates the current
publication, exact start/count, device, stream, source identity and sequence
before touching its output or ring. One caller serializes source/consumer calls
and their destruction, on one explicit stream.

| Event | Publication and consumer behavior |
| --- | --- |
| Source successfully advances | Replaces the latest batch; older copies are invalid |
| Source rejects an argument or capacity | Keeps the previous batch and all source history intact |
| Source resets | Invalidates the batch and starts a new sequence; consumer must also reset |
| Source execution fails | Invalidates the batch and requires source reconstruction |
| Source is destroyed | Weak handles expire; consumer rejects without accessing freed GPU storage |
| Consumer rejects a stale/wrong batch | Leaves output, own ring and source state unchanged |
| Consumer resets | Clears only its ring and source binding; the current source publication remains valid |
| Consumer execution fails | Poisons the consumer; the source remains independent |

The source does not track whether each dependent layer has run. The caller must
consume a batch before advancing the source; attempting to catch up with stale
indices is rejected. A consumer also rejects a new source sequence at the same
absolute position, so matching token numbers alone cannot reuse stale history.

```cpp
// Fixed layer-2 weights/components and layer-3 projection weights, batch 1.
CudaLayer2Attention source(w2, compressor, key, query, 66, 0, 129);
CudaLayer3Attention consumer(w3, 0);
source.forward(x2, y2, tokens, start_pos, stream);
consumer.forward(x3, y3, tokens, start_pos, source.shared_batch(), stream);
// Before starting another sequence, reset both instances on the same stream.
source.reset(stream);
consumer.reset(stream);
```

`x2` and `x3` are the respective attention inputs. The decoder transformations
between those inputs are not assembled by this API or the reference fixture.

### Fixed weights and numerical reference

The local bounded fetch adds 13 original layer-3 attention tensors, 126743040
payload bytes, from revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Projection formats and shapes match the confirmed attention layout; all FP8
weights expand losslessly to BF16. The fixture verifies 34 source tensor hashes:
21 for the layer-2 source graph and 13 for the layer-3 consumer. It stores the
layer-3 weights; the test loads source weights from the existing step-6f fixture.

`prepare_layer2_attention_fixture.py --shared-layer3` runs the pinned official
full `Attention.forward` for layer 2 and then layer 3, sharing the official
`shared_attn.compress_kv` and `shared_attn.topk_idxs`. The consumer has its own
original projection/norm/sink weights and independent seeded BF16 hidden inputs.
The official non-source branches perform the sharing. The oracle checks that
the consumer leaves both shared arrays intact. CPU FP8/FP4 and sparse-attention
arithmetic retain the same limitations as earlier steps: no official TileLang
bit-parity or model-quality claim.

Three plans use prefill 1 or 6 followed by decode through position 8, and prefill
129 followed by decode at 129 and 130. This evaluates 149 token rows per layer in
16 paired calls. Every call checks both outputs, both rings, compressed KV and
index keys. GPU source histories/positions are compared byte-for-byte before
and after consumption. Short plans converge to identical histories; paired
reset/replay reproduces both outputs and histories byte-for-byte.

| Comparison | Calls | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Layer-2 source output | 16 | 0.00304962996 | 0.0107421875 |
| Layer-3 consumer output | 16 | 0.00187028752 | 0.0078125 |
| Source window ring | 16 | 0 | 0 |
| Consumer window ring | 16 | 0 | 0 |
| Shared compressed KV | 15 | 0 | 0 |
| Source index keys | 15 | 0 | 0 |

Both full-output gates remain relative L2 <=0.005 and maximum absolute error
<=`1e-5 + 0.03 * reference_peak`, with finite BF16-valued output. Compressed KV
and index-key gates remain <=0.003 and `1e-5 + 0.02 * reference_peak`. No gate was
changed. Table maxima can come from different calls. These integrated tests
reach 65 completed groups; top-512 boundary math remains covered separately by
the existing indexer/core tests. They do not validate a full multi-layer model.

### Measured allocation and validation

On the RTX 2080 Ti / SM75, CUDA 12.8, using the same Torch 2.9.1 CPU oracle:

| Allocation | Bytes | MiB |
| --- | ---: | ---: |
| Source with 66 compressed groups and 129-token publication | 287738112 | 274.408 |
| Consumer's own projections/core/window | 255627008 | 243.785 |
| Sum of owned GPU arenas | 543365120 | 518.193 |
| Extra source index reservation over step 6f | 34304 | 0.032715 |

These figures exclude caller arrays, CUDA context, cuBLAS allocations and
sanitizer overhead. The source's allocation with publication disabled remains
unchanged; at four groups and a three-token publication the added arena space is
exactly 256 bytes after alignment. All owned allocations remain fixed through
prefill, decode, rejection and reset. No speed measurement or improvement is
claimed in this step.

| Check | Result |
| --- | --- |
| Synthetic lifecycle, capacity, alias and state isolation | 51 checks passed |
| Synthetic plus original-weight shared attention reference | 182 checks passed |
| CPU weight/profile/source topology checks | 37 checks passed |
| Relevant DeepSeek and existing Strata regressions | 29/29 CTests passed |
| Layer-0 / standalone layer-2 / core real regressions | 60 / 251 / 95 checks passed |
| Changed C++ units with `-Wall -Wextra -Werror` | Passed |
| Default layer-2 fixture regenerated after oracle extension | Byte-identical to step 6f |
| CPU weight reader ASan/UBSan | 37 checks passed; no diagnostics |
| Complete shared synthetic/reference memcheck | 0 errors; 0 leaked bytes |
| Complete shared synthetic/reference racecheck | 0 errors; 0 warnings/hazards |
| Default Strata library feature isolation | Byte-identical before/OFF/ON, matching step 6f |

The lifecycle checks include missing/disabled publication, rejected source
prefill capacity before state changes, copied handles, wrong shape/stream,
source/consumer cache aliases, duplicate consumption, source advancing ahead,
source reset with a matching old token position, a different source at the same
position, source destruction, consumer-only reset/replay, and consumption after a
source capacity rejection. A full source cache still permits its pending odd
token.

### Reproduce step 6g

Use the opt-in SM75 build, existing step-6f source fixture and local pinned
weight/source cache. The preparer refuses to overwrite existing files.

```sh
cmake --build build-deepseek-cuda --target deepseek41_shared_attention_cuda_test \
  deepseek41_attention_weights_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_attention_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --shared-layer3 --output build-deepseek-cuda/real-shared-attention23.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(shared_attention_cuda|attention_weights)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_shared_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf \
  build-deepseek-cuda/real-shared-attention23.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 build-deepseek-cuda/src/deepseek41/deepseek41_shared_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf build-deepseek-cuda/real-shared-attention23.gguf
/usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  --force-synchronization-limit 16 --racecheck-num-workers 2 \
  build-deepseek-cuda/src/deepseek41/deepseek41_shared_attention_cuda_test --real \
  build-deepseek-cuda/real-layer2-attention.gguf build-deepseek-cuda/real-shared-attention23.gguf
```

The `--real` test includes synthetic checks. Racecheck uses the resource settings
established in step 6e without kernel filters. The new shared fixture is
137010272 bytes, SHA256
`7849b8401f9800d3c08f3cf8b155650d0541603fb9201cc52fea9c7750b18b9e`.
The default preparer still reproduces the step-6f fixture hash
`2f0f73fff0b88f50c0b923855a7df011b8300bd0663b8884e50b06c8cc6f6ee2`.
Fixtures, numerical metrics, validation logs and archive hashes are ignored
`build-deepseek-cuda/step6g-*` artifacts.

## Step 6h: complete layer-2 decoder

`CudaLayer2Decoder` assembles the confirmed layer-2 attention with the existing
mHC and text MoE components. It is fixed to the 1017 layer-2 geometry and source
topology. The original layer-0 decoder is unchanged. This step adds orchestration;
it adds no CUDA arithmetic kernel and does not change shared Strata kernels.

### Two passes with bounded scratch

The source attention needs every compressed group from the initial prefill before
attending its queries. Calling a one-token decoder repeatedly would change the
prefill candidate layout. The implementation therefore reads the caller's input
twice, on the same explicit stream:

1. Round each four-stream residual to BF16, collapse it with the incoming mHC
   coefficients, normalize it, and pass it through the confirmed compressor and
   compressed/index-key caches. This reuses the existing mHC collapse and RMSNorm
   code without calculating mixes or opening a `prepare`/`finish` pair.
2. For each token, run mHC preparation, the confirmed attention query/output path,
   the first residual join, FFN mHC, the original top-6 routed experts and shared
   expert, and the second residual join. Return the FFN `next_pre` for the same
   token in the following layer.

Attention's private two-pass operations are shared by its existing public wrapper
and the new decoder. Its projections, index selection, quantization and attention
math are unchanged. The decoder adds only **102400 bytes (100 KiB)** of intermediate
storage: one 5120-value attention output and one four-stream residual. It does not
allocate a prefill-sized hidden tensor. Component weights, caches and optional
shared indices remain allocated at construction. Caller input/output buffers,
cuBLAS allocations and the CUDA context are outside `device_bytes()`.

Argument, stream, route-span, cache-capacity and alias checks run before mutation.
All four caller buffers must also be disjoint from exposed attention caches. The
shared batch and completed position become available only after every MoE and
residual join succeeds. An execution failure poisons the whole decoder and expires
borrowed publications; reset cannot recover it. Reset clears sequence caches and
partial groups while retaining immutable expert weights and cumulative cache
statistics. This remains synchronous batch-1 execution.

### Reference and validation

The new offline preparer reuses the pinned layer-2 attention oracle and runs the
original `Block.forward`, mHC/Sinkhorn and Gate logic. The fixture uses prefill 3,
then decode at positions 3 and 4, with distinct per-token incoming coefficients.
The resulting 27 original expert IDs are fetched from the pinned revision, without
changing routing bias or substituting experts. Every used source tensor is checked
against the local pinned-download provenance.

As in step 5d, the C++ test injects the already validated native **Q8_1-activation
MoE baseline** into an independent CPU residual join. It validates composition
around the reference FFN input; it does not establish official full-FP8 MoE parity
or full-model quality. A guarded source rejects unpopulated expert holes in the
partial routed fixture. Python/Torch is used only by this offline oracle.

Output and window state retain relative L2 <= 0.005 and max absolute error <=
`1e-5 + 0.03 * reference_peak`; compressed/key state retains 0.003 and 0.02.
Original route IDs must match exactly; complete layer route weights retain the
absolute tolerance `1e-4`. The mHC **same-input** coefficient gate remains
`2e-6 + 2e-5 * abs(reference)`.

The initial real run failed that coefficient gate for decode position 3 in all
four cache/replay combinations. The end-to-end maximum difference was
`1.19805336e-5`. Diagnosis reproduced it outside the decoder and isolated the
stages: attention inputs were identical to the reference; attention output had
relative L2 `0.00136103548`, within its existing gate; the CPU residual join on
that native output matched exactly. On the same resulting residual, native mHC
matched the independent Torch coefficient calculation within `3.73e-8` for this
token. The failure was applying a same-input operator threshold across different
upstream attention results. Runtime arithmetic was not changed to address it.

The final test makes this distinction explicit. It validates the native attention
residual against the original CPU graph, checks a separate double-precision CPU
coefficient calculation against the pinned oracle on identical inputs, and checks
the full decoder's coefficients against that CPU calculation on the native
attention boundary. The end-to-end comparison to the original CPU graph then uses
a calculated propagation bound, from `abs(sigmoid'(z)) <= 1/4`:

```text
z = native_residual / rms(native_residual)
r = reference_residual / rms(reference_residual)
bound_i = abs(scale * dot(W_i, z - r)) / 4
          + 2e-6 + 2e-5 * abs(reference_pre_i)
```

This is a new **composition** gate; it is not a claim that the original
end-to-end pointwise check passed. The strict operator gate is preserved. Both
checks must pass, alongside the unchanged full-output/cache/route gates. In the
final run the largest coefficient error used 26.3% of its calculated bound.
The initial failure and the independent boundary diagnostics are retained in
`step6h-real-initial.log`, `step6h-mhc-diagnostic.json` and
`step6h-upstream-diagnostic.json`.

Measured on the same RTX 2080 Ti (SM75), CUDA 12.8.93 and pinned sources used
in step 6g:

| Check | Result |
| --- | --- |
| Synthetic decoder | 55 checks pass; 129-token prefill + decode matches the separate attention/shared-expert/CPU-join composition byte-for-byte |
| Real decoder | 142 checks pass; prefill 3 + two decode calls, slots 6 and 2, each replayed after reset |
| Real output | Maximum relative L2 `0.000882023253`; maximum absolute error `0.015625` |
| Same-input mHC coefficients | Maximum absolute error `1.1920929e-7` against the independent CPU calculation |
| End-to-end mHC coefficients | Maximum absolute difference `1.19805336e-5`; all calculated propagation bounds pass |
| Window, compressed KV and index keys | All sampled real values match the CPU fixture exactly |
| Routing and cache invariance | Original top-6 IDs retained; outputs and coefficients byte-identical across reset and slot eviction |
| Regression | 30 CTests pass; layer-0 decoder 88, layer-2 attention 251 and shared attention 182 real-fixture suite checks pass |
| CUDA memcheck | Synthetic 55 and real 142 checks pass; 0 errors, 0 leaked bytes |
| CUDA racecheck | Synthetic 55 and real 142 checks pass; 0 hazards, errors or warnings; no kernel filters |
| Strict host C++ checks | Both decoder sources and the new test pass `-Wall -Wextra -Werror` syntax checks |
| Feature isolation | Default core/kernels/engine archives match step 6g before, with the feature OFF and with it ON |
| Oracle reuse | Default layer-2 attention fixture reproduces its original SHA256; default layer-0 routed fixture is byte-identical |

| Startup budget | Owned device bytes | MiB |
| --- | ---: | ---: |
| Real dimensions, 6 expert slots, 3 groups, shared prefill limit 3 | 486247968 | 463.722 |
| Real dimensions, 2 expert slots, 3 groups, shared prefill limit 3 | 406621728 | 387.785 |
| Synthetic long prefill, 6 slots, 65 groups, shared prefill limit 129 | 486361888 | 463.831 |

The two-slot real run has 60 misses and 58 evictions over the two five-token
passes; the six-slot run has 58 misses, 2 hits and 52 evictions. Existing cache
counters match original expert reads and H2D bytes. These are correctness/cache
accounting results, not throughput measurements.

The combined decoder fixture is 184061888 bytes, SHA256
`e1d08b80d97c31a8f034ed75d29099f4137ba4147e693481d5333a24ff7dae7c`,
with 199 verified source tensors. The sparse routed fixture has 7223418208 logical
bytes and 511897600 allocated disk bytes; only the 27 required original experts
are populated.


The synthetic fixture has nonzero attention, compression, key/query projections,
mHC coefficients and shared-expert weights. It covers 129-token prefill, window
wrap, decode, cache saturation with one pending odd token, aliases into all three
exposed caches, rejected calls, reset/replay and an expert-loading failure after
the compressor has advanced. It also consumes a completed decoder publication
with layer-3 attention and rejects that publication after a later decoder failure.
No layer-3 decoder is introduced here.

### Reproduce step 6h

Use the opt-in SM75 build and the existing pinned local source/tensor cache.
`--discover` prints the required original routes without writing a fixture. Final
preparation requires their locally cached weights and refuses existing outputs.

```sh
cmake --build build-deepseek-cuda --target deepseek41_layer2_decoder_cuda_test -j 4
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_decoder_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --output build-deepseek-cuda/real-layer2-decoder.gguf \
  --routed-output build-deepseek-cuda/real-layer2-decoder-routed.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_layer2_decoder_cuda_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_layer2_decoder_cuda_test --real \
  build-deepseek-cuda/real-layer2-decoder.gguf build-deepseek-cuda/real-layer2-decoder-routed.gguf
```

The default test runs synthetic cases; `--real` runs the independent real-weight
composition cases. Run both under each sanitizer. For memcheck, prepend
`compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99`.
For racecheck, prepend `compute-sanitizer --tool racecheck --error-exitcode 99
--force-synchronization-limit 16 --racecheck-num-workers 2`. These are the bounded
resource settings from step 6e, with no kernel filters.

Fixture hashes, source records, route IDs, numerical metrics, validation results
and logs are ignored `build-deepseek-cuda/step6h-*` artifacts. No full-checkpoint
conversion, embedding, Engram, remaining decoder layers, output head or text
generation is included. Throughput comparison and full-model quality remain
separate review gates.

## Step 6i: layer-3 decoder and the layer-2/3 pair

`CudaLayer3Decoder` composes the existing mHC, shared-attention consumer and text
MoE. The geometry is fixed to 1017 layer 3, whose compressed KV and index source
is layer 2. It owns its window ring and Strata expert cache. The source's compressed
KV and selected indices remain borrowed; no duplicate cache or indexer is added.
This step adds no arithmetic kernel and does not change the routed Q8_1 baseline.

The decoder consumes the preceding layer's four-stream residual and delayed mHC
coefficients **for the same tokens**, directly from their GPU buffers. It returns
its residuals and next-layer coefficients. Attention's private per-token phases
retain the original whole-prefill publication layout. The public attention-only
path uses the same phases. The completed consumer position advances only after
all mHC, attention, MoE and residual operations finish successfully.

The new assembly adds 102400 bytes (100 KiB) of fixed scratch. It does not allocate
prompt-sized intermediate hidden arrays. All four caller buffers are checked
against each other, the consumer's storage and the source's exposed caches before
mutation. A stale or mismatched publication preserves both layers. Execution
failure poisons only this consumer, leaves its completed position unchanged and
does not revoke the source publication. The caller must discard partial outputs
and reconstruct the failed instance. Consumer reset clears only its own sequence
state, retaining expert weights/statistics. The caller resets both layers before
starting a new source sequence and serializes their lifetimes on one explicit
stream. This remains synchronous, batch-1 execution.

### Independent reference and actual layer handoff

The real fixture uses the existing five-token layer-2 case: prefill 3, followed
by decode at positions 3 and 4. The test's `--capture` mode exports layer 2's actual
residuals and coefficients after its complete native MoE and residual joins. The
offline preparer reconstructs source attention from the original layer-2 input
and runs the hash-pinned original layer-3 `Block.forward` on that captured handoff.
It uses the original Gate, all original selected expert IDs and provenance-checked
weights. It does not generate an unrelated layer-3 hidden input or replace routed
expert contributions with zeros in the layer-2 handoff.

The pair test reruns layer 2 and requires its outputs and coefficients to match
the oracle's captured inputs **byte-for-byte**. It then passes those same GPU
addresses to layer 3. The main layer-3 reference combines the CPU graph through
the FFN input with the declared native Q8_1 MoE and an independent CPU residual
join, as in steps 5d/6h. It separately compares source/consumer caches, original
routes and the native attention boundary. Thus this is a real two-decoder
integration check conditioned on native layer-2 output, not an independent
reference for two complete official FP8 decoder blocks.

The numerical gates are unchanged from confirmed step 6h: output/window relative
L2 <= 0.005 and max absolute error <= `1e-5 + 0.03 * peak`; compressed/key state
uses 0.003 and 0.02. Original expert IDs must match exactly; route weights use
`1e-4` for the complete decoder and `2e-6` for MoE on identical reference inputs.
Same-input coefficients retain `2e-6 + 2e-5 * abs(reference)`. The separate
end-to-end coefficient check uses the documented attention-error propagation
bound. Both coefficient checks and the upstream residual gate must pass.

The first synthetic run exposed an assertion error in the new test: uniform
routing was compared to exactly 0.25, whereas the existing F32 router produces
0.249999985. All six original IDs and the full residuals were already correct.
The assertion now uses the existing routing test's `3e-8` tolerance. Runtime
arithmetic was not changed. The original failure and observed values are retained
in `step6i-synthetic.log` and `step6i-route-diagnostic.log`.

Measured on the RTX 2080 Ti (SM75), CUDA 12.8.93, with the same pinned sources
as step 6h:

| Check | Result |
| --- | --- |
| Synthetic pair | 94 checks pass; prefill 5 + two decode calls and reset/replay |
| Real pair | 192 checks pass; prefill 3 + two decode calls; slots 6 and 2, each replayed after reset |
| Actual layer-2 handoff | Residuals and coefficients match the oracle inputs byte-for-byte in all runs |
| Layer-3 output | Maximum relative L2 `3.37992712e-5`; maximum absolute error `0.0078125` |
| Same-input coefficients | Maximum absolute error `5.96046448e-8` |
| End-to-end coefficients | Maximum absolute difference `1.49011612e-8`; all propagation bounds pass |
| Window/compressed/index state | All sampled real source and consumer cache values match the CPU fixture exactly |
| Routing and replay | Original expert IDs retained; outputs and coefficients byte-identical across reset and slot eviction |
| Regression | 31 CTests pass; real layer-0 decoder 88, layer-2 decoder 142, layer-2 attention 251 and shared attention 182 checks pass |
| CUDA memcheck | Synthetic 94 and real 192 checks pass; 0 errors and 0 leaked bytes |
| CUDA racecheck | Synthetic 94 and real 192 checks pass; 0 hazards, errors or warnings; no kernel filters |
| Strict host C++ | New decoder and both decoder tests pass `-Wall -Wextra -Werror` syntax checks |
| Feature isolation | Default core/kernels/engine archive hashes match step 6h before, OFF and ON; main checkout clean |
| Offline compatibility | The default layer-0 routed fixture remains byte-identical after allowing explicit layer 3 |

| Per-layer expert slots | Layer 2 bytes | Layer 3 bytes | Pair bytes | Pair MiB |
| --- | ---: | ---: | ---: | ---: |
| 6 each | 486247968 | 454252576 | 940500544 | 896.931 |
| 2 each | 406621728 | 374626336 | 781248064 | 745.056 |

These fixed allocations use capacity 3 compressed groups and shared prefill limit
3. The consumer has no compressed/index cache of its own. Caller buffers, cuBLAS
allocations and the CUDA context are excluded. These are two-layer budgets, not
full-model memory requirements. No throughput measurement is claimed.

Across two five-token passes, layer 3 has 60 misses and 54 evictions with six
slots, and 60 misses and 58 evictions with two slots. Layer 2 retains the previous
58 misses/2 hits/52 evictions with six slots and 60/0/58 with two. All H2D/read
counters match Strata's cache accounting without changing outputs.

The layer-3 fixture has 28 original experts and 219 provenance-checked source
tensors. Its combined weights/reference file is 169066176 bytes, SHA256
`3af753b338c34420bd7ef93a230bbaf632f80afcc9801b421436a15aeae538e4`.
The routed fixture has 7223418816 logical bytes and 530698240 allocated disk
bytes. Unpopulated expert IDs are guarded holes. The native layer-2 handoff SHA256
is `351785f17caa73e771f39855d6f120458bb164b8f067a6edaf72d7c6c2344d5e`.


The synthetic test has nonzero attention, compression, shared-expert weights and
input-dependent outgoing mHC coefficients. Prefill 5 plus two decode calls cover
both partial and completed compressed groups. Shifting the per-token coefficients
changes the result. It checks every caller-buffer pair and all four buffers against
both layers' exposed caches, rejected arguments, stale publications, reset/replay,
two-slot eviction, and expert-load failure after consumer attention has advanced.
A healthy second consumer can still consume the source batch after that failure.
The attention-only regression additionally retains the long prefill/window-wrap,
source identity/sequence and source-destruction coverage from step 6g.

### Reproduce step 6i

Use the existing step-6h layer-2 fixtures and pinned local source/tensor cache.
The capture/preparer refuse existing output paths. `--discover` uses the handoff
to report the original layer-3 routes before fetching any missing expert weights.
The final preparation requires those weights locally; Python/Torch is offline only.

```sh
cmake --build build-deepseek-cuda --target deepseek41_layer3_decoder_cuda_test -j 4
build-deepseek-cuda/src/deepseek41/deepseek41_layer3_decoder_cuda_test --capture \
  build-deepseek-cuda/real-layer2-decoder.gguf \
  build-deepseek-cuda/real-layer2-decoder-routed.gguf \
  build-deepseek-cuda/step6i-handoff.bin
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer3_decoder_fixture.py \
  --model-dir /tmp/deepinfer-consumer-reference/expert0 \
  --reference-model /tmp/deepinfer-consumer-reference/inference_model.py \
  --reference-kernel /tmp/deepinfer-consumer-reference/inference_kernel.py \
  --handoff build-deepseek-cuda/step6i-handoff.bin \
  --output build-deepseek-cuda/real-layer3-decoder.gguf \
  --routed-output build-deepseek-cuda/real-layer3-decoder-routed.gguf
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_layer3_decoder_cuda_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_layer3_decoder_cuda_test --real \
  build-deepseek-cuda/real-layer2-decoder.gguf \
  build-deepseek-cuda/real-layer2-decoder-routed.gguf \
  build-deepseek-cuda/real-layer3-decoder.gguf \
  build-deepseek-cuda/real-layer3-decoder-routed.gguf
```

Run both the default synthetic mode and `--real` under memcheck/racecheck using
the bounded settings documented in step 6h, without kernel filters. The ignored
`build-deepseek-cuda/step6i-*` artifacts contain source hashes, original route IDs,
fixture and handoff hashes, validation commands, metrics and logs. No full-model
weight conversion, text generation, throughput claim or quality claim is included.

## Step 6j: layer-1 Engram lookup and sequence state

`Layer1EngramHash` and `Layer1EngramTable` implement the fixed 1017 layer-1
lookup boundary on the host. They reuse Strata's `GgufFile`/`GgufModel`,
`PleReader`, `DirectFile`, bounded row caches and BF16 conversion helpers.
This step adds no GPU kernel, projection, residual gate or decoder assembly.
Python/Torch is used only to prepare the independent test oracle offline.
The feature remains opt-in; shared Strata sources are unchanged.

### Pinned layout and causal token state

The original tokenizer has 129280 IDs. The pinned `build_compressed_token_map`
normalizes decoded tokens and maps them into 99092 compressed IDs. The native
runtime consumes that offline map; it does not reimplement tokenizer normalization.
The pad ID is 2. Original token 0 is valid; EOS 1 is ordinary hash input and does
not cut history. Qwen's existing PLE hash cannot supply this model's semantics.

Each token produces 24 original table row IDs: eight heads for each of the
2-, 3- and 4-grams. Layer-1 multipliers and prime moduli are fixed from the pinned
`NgramHashState`/`EngramLayout`. The native state retains the last three compressed
IDs. A dead-token mask blocks that token and all history before it; blocked slots
use the compressed pad ID. Contiguous chunks and one-token decode yield identical
rows. Reset clears history and position, retaining the map.

This is a synchronous, single-sequence interface, with a 1048576-position limit.
It owns 517120 bytes (505 KiB) for the map and 12 bytes for the token history,
without a sequence-length-sized cache. Input, shape, range, mask and alias errors
are rejected before changing either history or output.

| Original layer-1 tensor | Source shape | Bytes per row | Native artifact representation |
| --- | --- | ---: | --- |
| `layers.1.engram.embed.weight` | `[384006168,256]` FP8 E4M3FN | 256 | `blk.1.engram_embed.weight`, I8 `[256,384006168]` |
| `layers.1.engram.embed.scale` | `[384006168,8]` E8M0 | 8 | `blk.1.engram_embed.scale`, I8 `[8,384006168]` |

There are eight scales **per row**, one for every 32 values. The scale layout is
different from the 32x32 blocks used by attention/shared-expert matrices. The
adapter restores these row values to BF16 on the CPU, matching the pinned
`ParallelEngramEmbedding.forward` boundary. No full-table conversion is performed.

Two existing `PleReader` instances read values and scales, including when they
are in separate shards. All GGUF mappings are released before direct I/O opens.
A gather reads one token's 24 rows and returns 6144 BF16 values. It owns 18624
bytes of raw/restored staging, excluding the readers' bounded caches and I/O
buffers. The default is 4096 cached rows and 24 inflight reads per reader.
Original row IDs and offsets are retained, including offsets above 4 GiB.

The table profile is `1017-layer1-fp8-row32`. A full table must declare
`strata.deepseek41.engram_complete=1`. A sparse validation artifact declares 0,
requires explicit `allow_partial=true`, and carries sorted populated original IDs
in `blk.1.engram_populated_rows`, an I8 `[4,N]` tensor of little-endian uint32s.
Unpopulated rows are rejected before I/O. Shape/profile validation does not
authenticate arbitrary checkpoint payloads; the offline preparer checks the pinned
source hashes and cached row provenance. A full-checkpoint converter is later work.

Gather commits caller output only after both reads and all conversions succeed.
Invalid arguments preserve output and issue no reads. Short reads, nonfinite rows
and other execution failures close/drain both readers and poison the instance.
The caller must abort its enclosing sequence and reconstruct after such a failure.
Hash and lookup are separate components at this gate; no decoder-wide transaction
is claimed. Each table instance has one caller, with immutable input files.

### Reference, real rows and observed cache behavior

The source revision remains `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Additional pinned inputs for this gate are:

| Input | SHA256 |
| --- | --- |
| `inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| Generated little-endian uint32 token map | `c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481` |
| Original `model-00047-of-00048.safetensors` header | `e5c5c7fc900caec400644bef07aa5d7e28fc6ca941c8c47592b393963b218242` |

The nine-token case includes BOS, EOS, three spellings that normalize to the same
compressed token, the last original token ID, and a dead-token barrier. It checks
prefill lengths 1/3/9, decode, reset and masked/unmasked hashing against the
original `NgramHashState`. A seeded 257-token case adds multiple barriers and
chunk lengths 1/7/31/257. Hash row IDs must match exactly.

Only 220 original rows were fetched: the 216 masked-sequence rows plus rows
0, 15, 16 and the final table row. The row payload is 58080 bytes, with revision,
source offsets and per-row SHA256 recorded. The independent lookup oracle runs
the original `ParallelEngramEmbedding.forward`; only its embedding storage lookup
maps global row IDs to the compact downloaded row cache. Its scale arithmetic and
BF16 conversion remain original. All sampled output bits match the native path.

The reference file is 841088 bytes, SHA256
`73f18de9a9a12c6680fe9c38c5ff26f94c6b16c3f4b50d54f93c09a4b15e8d07`.
The table fixture preserves 101377629888 logical bytes while occupying 1785856
allocated disk bytes on this filesystem. It is an explicitly guarded partial
fixture, not a complete checkpoint.

Measured on Linux x86-64, Xeon E5-2686 v4, GCC 13.3, Torch 2.9.1+cu128 for the
offline oracle, with `STRATA_IO_THREADS=2`. The existing CUDA regressions use the
RTX 2080 Ti/SM75 and CUDA 12.8.93. New Engram execution itself is host-only.

| Check | Result |
| --- | --- |
| Synthetic Engram | 87 checks pass, including the full position limit, split shards, >4 GiB offsets, malformed inputs, guarded holes, corrupt rows, short reads and poisoned-instance rejection |
| Real Engram | 42 checks pass; original hash IDs and all 220 restored rows match exactly, with reset/replay and cache capacities 4096/512/8/0 |
| CPU ASan/UBSan | Synthetic 87 and real 42 checks pass with leak detection and halt-on-error enabled; no sanitizer diagnostics |
| Regression | 33 CTests pass: the previous 31, new Engram test, and Strata `ple_reader_selftest` covering existing 90/110-byte row formats |
| Strict host C++ | New Engram implementation/test pass `-Wall -Wextra -Werror`; preparer syntax checked |
| Feature isolation | Default core/kernels/engine archive hashes match step 6i before, OFF and ON; main checkout clean |

For the nine-token sequence, every cold configuration reads 216 value pages and
216 scale pages, totaling 1769472 bytes. The actual source rows total 57024 bytes;
page-granularity direct I/O is a material amplification at this boundary.
The second pass has the following observed counters **per reader**:

| Cache capacity (rows) | Cache hits | Read operations | Combined value+scale read bytes |
| ---: | ---: | ---: | ---: |
| 4096 | 216 | 0 | 0 |
| 512 | 202 | 14 | 114688 |
| 8 | 0 | 216 | 1769472 |
| 0 | 0 | 216 | 1769472 |

These are repeated-fixture counters, not a prompt-locality study or a throughput
benchmark. Strata's cache is eight-way set associative. The 512-row case places
12 and 9 distinct rows into two sets, so it cannot retain this sequence despite
having sufficient total slots. At 4096 rows the maximum is three per set.

Two test-development corrections are recorded in the ignored validation logs.
First, Strata's GGUF reader retains only the first 64 metadata-array elements.
The full token map and coverage list therefore use validated tensor payloads;
the shared reader is unchanged. Second, the initial 512-row warm-cache assertion
incorrectly assumed full associativity. The final tests retain this conflict case
and use the default 4096-row cache for the zero-reread case. Runtime hash and
lookup arithmetic did not change to accommodate either test correction.

### Reproduce step 6j

Use the pinned local sources, tokenizer and provenance-checked `engram1-rows`
cache under `/tmp/deepinfer-consumer-reference`. The preparer makes no network
requests and refuses existing output paths. `--discover` reports required original
row IDs before fetching any rows; it does not require the row cache.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_engram_fixture.py \
  --reference-root /tmp/deepinfer-consumer-reference --discover
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_engram_fixture.py \
  --reference-root /tmp/deepinfer-consumer-reference \
  --output build-deepseek-cuda/real-engram1-reference.gguf \
  --table-output build-deepseek-cuda/real-engram1-table.gguf
cmake --build build-deepseek-cuda --target deepseek41_engram_test ple_reader_test -j 4
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^(deepseek41_engram_test|ple_reader_selftest)$'
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_test --real \
  build-deepseek-cuda/real-engram1-reference.gguf \
  build-deepseek-cuda/real-engram1-table.gguf
```

The same Engram test builds with `STRATA_ENABLE_CUDA=OFF` for ASan/UBSan.
Ignored `build-deepseek-cuda/step6j-*` artifacts contain the source/fixture hashes,
cache diagnosis, test commands, metrics and results. This step does not validate
Engram projection/gating, layer 14, a complete layer-1 decoder or text generation.

## Step 6k: layer-1 Engram projection and residual gate

`CudaLayer1Engram` consumes the confirmed lookup boundary: 6144 BF16-valued
inputs and a `[4,5120]` residual, returning the updated residual. The real test
uses `Layer1EngramHash` and `Layer1EngramTable` directly, restores their host BF16
rows to F32 and copies those values into the GPU projection. The runtime remains
C++/CUDA. Python/Torch is only an offline source of reference outputs.

The new weight adapter accepts the fixed profile
`1017-layer1-wkv-fp8-block32`, with original `wkv.weight` `[25600,6144]`,
`wkv.scale` `[800,192]`, and BF16 `q_weight`/`k_weight` `[4,5120]`.
The projection scales use 32x32 blocks, distinct from the row-local scales in
the Engram embedding table. Malformed dimensions, dtypes and payload ranges are
rejected; the runtime profile is not checkpoint authentication.

### Reused computation and model-specific gate

The projection reuses `quantize_fp8_bf16`, Strata's `bf16_gemv`, BF16 rounding,
`DeviceArena`, and the existing lossless FP8 weight expansion. The expansion runs
in at most 4096-row chunks without increasing the converter's existing limits.
No matrix kernel or shared Strata source is changed.

The new CUDA gate kernel follows the pinned `Engram.forward` operations:

1. Split BF16-rounded projection output into four keys and one shared value.
2. Form `q_weight * k_weight` in F32 once at startup.
3. Reduce residual squares, key squares and the weighted dot separately for each
   5120-wide residual stream. Apply both reciprocal RMS factors after the dot,
   with epsilon `1e-20`, then multiply by `5120**-0.5`.
4. Apply `sigmoid(copysign(sqrt(max(abs(dot),1e-6)),dot))`. A positive zero score
   therefore produces approximately `0.50025`, not `0.5`.
5. A false token mask sets all four gates to zero. Broadcast the shared value,
   add the gated update and round the residual back to BF16.

Strata's Qwen PLE post-operations cannot directly replace this gate: their
normalization/rounding order, zero-sign handling and convolution differ. This
step adds only the model-specific reduction and residual kernel around existing
projection machinery. It does not introduce the PLE convolution.

Calls are synchronous, single-token and stateless with respect to sequence
position. Hash history remains in the step-6j component. Each instance uses one
caller and one explicit stream on its device. Lookup, residual and output buffers
must be mutually disjoint and outside the owned arena. Invalid arguments preserve
output; an execution failure poisons the instance and requires discarding output
and reconstructing it. Caller inputs/intermediates must remain finite, including
the existing activation quantizer's input-range contract. The returned projection
and gate views are read-only and valid until the next call or destruction.

| Owned allocation | Bytes | MiB |
| --- | ---: | ---: |
| BF16 projection matrix | 314572800 | 300.000 |
| F32 q/k product, quantized input, projection and gate scratch | 196624 | 0.187515 |
| Total device arena | 314769424 | 300.187515 |
| Largest temporary host expansion chunk | 50331648 | 48.000 |

The matrix remains resident in this correctness baseline. Startup host mapping,
row-reader caches, caller buffers and CUDA context are excluded from device arena
bytes. There is no per-forward allocation inside the component. This is one
Engram projection budget, not a full-model 20 GB residency result.

### Independent reference and validation

The source revision and model SHA remain pinned as in step 6j. Four original
projection/gate tensors were fetched with bounded range requests from
`model-00047-of-00048.safetensors`, totaling 157521920 bytes. Original tensor
shapes, offsets, header hash and per-tensor payload hashes are recorded. No new
embedding-table rows or complete checkpoint were downloaded.

| Original tensor | SHA256 |
| --- | --- |
| `layers.1.engram.q_weight` | `af0db71b7f9be7d4578321da60e2522dd7cb627f69f880054c2c757c79606386` |
| `layers.1.engram.k_weight` | `7609238ee984cf9d738ceb87b24850918357c89624edd1fbc0028271e74b7aa4` |
| `layers.1.engram.wkv.weight` | `85b765f5766dd4a5d3575146349026d246717a8a92320b05a937df6df034bf32` |
| `layers.1.engram.wkv.scale` | `20e50c9226692fffa94ab89270c466c52a01beea8f5dc6731b7bc783a078cdb8` |

The offline preparer executes the hash-pinned original `Engram.forward` and
`ParallelEngramEmbedding.forward`. Sparse lookup storage retains original row
IDs. The large linear uses the already-established independent NumPy block-32
FP8 reference, with original FP8 weights and activation quantization; it does
not reproduce TileLang's reduction order. A separate scalar reference checks
gates on actual native projection output, isolating gate error from projection
error. Residual inputs are seeded BF16 test vectors, not a layer-0 decoder handoff.

The nine-token case reuses the original masked lookup sequence, including EOS,
a dead-token barrier and normalization-colliding tokens. Stream magnitudes differ
by design, and one residual is entirely zero. It runs prefill 3 plus decode and
prefill 9, reset/replay, and row-cache capacities 4096 and 0. This exercises 36
complete lookup/projection/gate calls. All original hash IDs and restored FP8
activations match their references exactly.

Projection/residual numerical gates are relative L2 <= 0.003 and maximum absolute
error <= `1e-5 + 0.02 * peak(reference)`, following the existing dense-component
baseline. End-to-end gate absolute error must be <= `2e-4`. On identical projected
keys, the gate must satisfy `2e-6 + 2e-5 * abs(reference)` pointwise. These limits
were set before running the real fixture; sampled bitwise output equality is
reported separately rather than required for all inputs and GPU architectures.

Measured on the same RTX 2080 Ti/SM75, CUDA 12.8.93, GCC 13.3 and Torch 2.9.1
reference environment as step 6j, with `STRATA_IO_THREADS=2`:

| Check | Result |
| --- | --- |
| Weight layout | 16 checks pass, including wrong layer/profile, each missing/wrong-shaped/wrong-typed tensor and truncation |
| Synthetic CUDA | 42 checks pass; full-size sparse analytic projection, all chunk boundaries, per-stream gates, zero-score rule, token masking, alias/stream rejection and replay |
| Real lookup through gate | 263 checks pass; prefill/decode/reset and enabled/disabled row caching |
| Projected keys/value | Maximum relative L2 `1.94247321e-7`; maximum absolute error `3.05175781e-5` |
| End-to-end gates | Maximum absolute error `1.1920929e-7` |
| Gate on native projected keys | Maximum absolute error `9.68575478e-8` |
| Final residual | Maximum absolute error 0; separate byte comparison confirms sampled outputs match reference bits in all 36 calls |
| Replay/cache invariance | Full Engram output bytes unchanged across chunking, reset and disabled row cache |
| CUDA memcheck | Synthetic 42 and real 263 checks pass; 0 errors and 0 leaked bytes |
| CUDA racecheck | Synthetic 42 and real 263 checks pass; 0 errors/warnings/hazards, no kernel filters |
| CPU ASan/UBSan | New weight-layout 16 and existing lookup 87 checks pass, with no diagnostics |
| Regression | 35 CTests pass: the previous 33 plus new weight-layout and CUDA Engram tests |
| Strict host C++ | New loader and both tests pass `-Wall -Wextra -Werror`; preparer syntax checked |
| Feature isolation | Default core/kernels/engine archive hashes match step 6j before, OFF and ON; main checkout clean |

The reference/weight fixture is 160253792 bytes, SHA256
`b4c633b6bfcf17d54a77d002744032891a2e2fe583b433497118594e1b604786`.
The final byte-comparison diagnostic only adds host-side reporting; CUDA kernels
are unchanged from the successful memcheck/racecheck runs. No full-model quality,
throughput or official GPU reduction-order parity claim follows from this test.

### Reproduce step 6k

Use the pinned local source/tokenizer, step-6j lookup fixtures/row cache and
`/tmp/deepinfer-consumer-reference/engram1-projection` with its provenance record.
The preparer makes no network requests and refuses existing output paths.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_engram_projection_fixture.py \
  --reference-root /tmp/deepinfer-consumer-reference \
  --output build-deepseek-cuda/real-engram1-projection.gguf
cmake --build build-deepseek-cuda \
  --target deepseek41_engram_projection_test deepseek41_engram_cuda_test -j 4
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_engram_(projection|cuda)_test$'
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_cuda_test --real \
  build-deepseek-cuda/real-engram1-projection.gguf \
  build-deepseek-cuda/real-engram1-reference.gguf \
  build-deepseek-cuda/real-engram1-table.gguf
```

Run synthetic and real CUDA cases under memcheck/racecheck with the resource
settings from step 6i. The ignored `build-deepseek-cuda/step6k-*` records contain
source/fixture hashes, commands, metrics, sanitizer logs and file isolation checks.
This gate does not assemble a layer-1 decoder, modify delayed mHC coefficients,
implement layer-14 Engram or generate text.

## Step 6l: layer-1 text decoder with Engram

This gate assembles the confirmed components into `CudaLayer1Decoder` for the
fixed 1017 text layer 1. It reuses Engram hashing, bounded row reads, projection
and gating; the existing pure sliding-window attention; delayed mHC; and Strata's
original-ID expert cache, native routed expert compute and shared expert. There
are no new CUDA kernels in this gate. `CudaAttention` accepts an explicit layer-1
profile; existing calls still require layer-0 weights by default.

### Data and sequence contract

The caller supplies original token IDs on the host, device residuals
`[tokens,4,5120]`, and the previous layer's coefficients `[tokens,4]`. For each
same-position token the layer performs:

1. Hash the original token with the preceding token history, fetch its 24 original
   Engram rows through the two existing `PleReader` instances, and apply the
   confirmed Engram projection/gated residual update.
2. Compute attention mHC from that updated residual. Collapse the attention input
   using the **incoming** coefficients, then run attention and its residual join.
3. Compute FFN mHC from the attention residual. Collapse the FFN input using the
   **attention** coefficients, run original top-6 MoE plus the shared expert, and
   join the residual. Return the **FFN** coefficients for the next layer.

Engram preserves the incoming coefficients. Token zero and EOS remain ordinary
text; this wrapper does not implement image routing or dead-token masks. The
first call is prefill, followed by contiguous one-token decode, with at most
1048576 positions. Prefill retains its original attention candidate grouping
while using fixed one-token scratch. Reset clears the token history and KV ring,
and retains row/expert cache contents and counters.

All caller-buffer, token-ID, position and route-size checks precede row reads or
state changes. Buffers must be aligned, disjoint and outside the exposed KV
cache. The API is synchronous and binds to one explicit device stream. A row or
expert read failure can occur after history or attention has advanced, so the
whole layer is then unusable: discard the failed call's output and reconstruct
it. Reset deliberately cannot hide that failure.

### Validation and reference limits

On the RTX 2080 Ti, the synthetic test passes **65 checks**. Its 129-token prefill
and one-token decode cross the 128-slot ring boundary. Nonzero Engram, attention
and shared FFN contributions match the separately composed native components and
CPU residual joins bit-for-bit. Tests also cover same-token incoming coefficients,
reset/replay, cache retention, original expert IDs, rejected continuations and
buffer aliases, a late invalid token, expert-source failure and a truncated row
file after construction.

The real fixture uses the first five ordinary text tokens from the pinned 6k
Engram fixture: `[0,455,1805,1,20852]`, split into a three-token prefill and two
one-token decodes. Its Engram reference residuals are reused only after verifying
the complete 6k fixture SHA-256. The pinned original `Block`, attention and gate
then execute with the existing CPU linear/attention/Sinkhorn oracles. This is a
seeded layer-1 input, not output from an assembled layer-0/1 chain.

All **179** required dense and selected-expert source tensors are checked against
the pinned-revision download manifest. The reference selects **25 original
expert IDs**; only these are populated in the guarded sparse routed fixture.
There is no route substitution. The C++ composition oracle supplies the already
declared native NVFP4/Q8_1 routed MoE at the original CPU graph's FFN input, then
performs the final residual join on the CPU. This is not a full official FP8-MoE
parity claim.

The real decoder passes **111 checks**. The Engram boundary is bit-exact against
the pinned original output. Maximum attention-residual relative L2 is
`4.70257205e-6`; maximum final-output relative L2 is `4.82304394e-6`, with maximum
absolute error `0.00390625`. The KV ring is bit-exact for these cases. This is
within the existing residual/output gate (relative L2 at most `0.005`, maximum
absolute error at most `1e-5 + 0.03 * peak`), with BF16-rounded finite outputs.

Delayed coefficients retain the earlier strict same-input mHC check
(`2e-6 + 2e-5 * abs(reference)`). A separately measured native attention residual
is used for that comparison. The independent original-graph attention residual
also has to pass its own output gate; differences propagated to coefficients
must satisfy the existing sigmoid derivative bound. No acceptance tolerance was
widened for this gate.

The six-slot run uses a 4096-row cache per reader; the two-slot run disables row
caching. Both run the sequence twice with reset. Outputs and coefficients are
bit-exact across those configurations. The two-slot expert cache records 60
misses and 58 evictions across the ten tokens; each Engram reader records 240 row
requests. These are small-fixture correctness measurements, not a throughput
benchmark or a full-model cache-hit forecast.

All **36 selected DeepSeek, native-kernel and cache/I/O regression tests** pass.
The layer-0 real decoder also passes its **88 checks** after the attention wrapper
gains explicit layer-1 support.
The host attention loader passes **44 checks** and the reused Engram hash/row
fixtures pass **87 checks** under ASan/UBSan. Both new synthetic and real CUDA
runs pass full memcheck with zero errors/leaks and racecheck with zero hazards,
using the step-6i resource settings (`--force-synchronization-limit 16`,
`--racecheck-num-workers 2`). No kernel filters are used. Strict C++
`-Wall -Wextra -Werror` checks and Python preparer syntax checks pass.

The DeepSeek feature remains opt-in. With it disabled and enabled, the
`strata_core`, `strata_kernels` and `strata_engine` archive hashes match step 6k.
The main Strata checkout remains clean. Only the isolated feature's CUDA backend
and its host loaders/tests are touched; no HIP/SYCL kernel implementation changes.

### Bounded memory

The layer reports these owned device allocations, including resident weights,
its expert slots, attention state and fixed scratch:

| Expert slots | Owned device bytes | MiB |
| --- | ---: | ---: |
| 6 | 767962416 | 732.386 |
| 2 | 688336176 | 656.449 |

They remain unchanged across prefill, decode, cache eviction and reset.

This accounting excludes caller input/output buffers, CUDA/cuBLAS context,
source mappings/staging and host row-cache storage. The assembly adds 208896
bytes of device scratch and 36960 bytes of fixed host lookup/row scratch, plus
Engram's owned token map/history and configured reader caches. The existing
Engram projection still retains its 300 MiB BF16 matrix after bounded startup
expansion. The original-size Engram table remains on disk. None of these
single-layer results proves the 64 GB RAM/20 GB GPU whole-model target.

### Reproduce step 6l

Pinned source files and the bounded layer-1 checkpoint cache now live under the
ignored `build-deepseek-cuda/reference-cache/`, because the old `/tmp` reference
cache was absent. The 6j/6k reference and sparse table artifacts are reused.
The source-restore log/manifest and expert-download provenance record the exact
revision, hashes, shard offsets and HTTP range validation.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer1_decoder_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/layer1 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --engram-fixture build-deepseek-cuda/real-engram1-projection.gguf \
  --output build-deepseek-cuda/real-layer1-decoder.gguf \
  --routed-output build-deepseek-cuda/real-layer1-decoder-routed.gguf
cmake --build build-deepseek-cuda --target deepseek41_layer1_decoder_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_layer1_decoder_cuda_test
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_layer1_decoder_cuda_test \
  --real build-deepseek-cuda/real-layer1-decoder.gguf \
  build-deepseek-cuda/real-layer1-decoder-routed.gguf \
  build-deepseek-cuda/real-engram1-projection.gguf \
  build-deepseek-cuda/real-engram1-reference.gguf \
  build-deepseek-cuda/real-engram1-table.gguf
```

The preparer is offline, refuses existing output files, and supports `--discover`
in place of the output arguments to report which original experts are needed.
The ignored `build-deepseek-cuda/step6l-*` files record commands, source and fixture
hashes, numerical metrics, sanitizer results and isolation checks. This gate
stops at one complete text layer; connecting layers 0 through 3 is the next
proposed review gate.

## Step 6m: connect the native layer-0/1/2/3 text prefix

`CudaTextPrefix` owns and connects the four confirmed decoders. Layer 1 applies
Engram to the actual layer-0 residual while retaining that token's layer-0
coefficients. Layer 2 receives the actual layer-1 result and publishes its
completed compressed KV/index batch. Layer 3 consumes that exact publication
before the source can advance. Its output and delayed coefficients are the
prefix result for the next layer. No new numerical kernels are introduced.

### Bounded orchestration and ownership

The prefix alternates two residual buffers and two coefficient buffers. A
complete prefill batch passes through one layer before the next layer starts;
this preserves the existing prefill candidate grouping and the layer-2
compressor prepass. It does not switch the prefill to per-token traversal across
layers. The workspace is allocated once from the caller's prefill capacity:

```text
2 * (prefill_capacity * 4 * 5120 * sizeof(float)
     + align256(prefill_capacity * 4 * sizeof(float)))
```

The caller provides original host token IDs, device `[tokens,4,5120]` residuals
and `[tokens,4]` incoming coefficients. The first call is bounded prefill; later
calls are contiguous one-token decode. Optional host routes have layer-major
shape `[4,tokens]`, using the current call's token count, not the capacity.
Each layer keeps its own original-ID expert cache; layer 2 owns shared compressed
state. The prefix exposes read-only cache views for inspection but no shared
publication escapes its owner.

The outer call checks every token ID, all buffer ranges/aliases, the stream,
prefill capacity and the future compressed-group count before layer 0 executes.
That includes capacity failures which would otherwise occur only after layers 0
and 1 had advanced. Only a successful call through all four layers advances the
public prefix position. Any execution or I/O failure poisons the prefix, even
if earlier layers completed. Discard outputs from that call and reconstruct it.

One reset clears all four histories and the layer-2/3 publication bindings,
starting with the consumer, while retaining expert/row caches and their counters.
All calls are synchronous on one explicit device stream. This remains a
correctness baseline; overlap, graph capture and prefill throughput are later
work.

### Validation and its limits

The synthetic test passes **115 checks**. It compares against separate calls to
the confirmed decoders, using four independent destination buffers instead of
the prefix's two reused buffers. Inputs have different coefficients per token
and per layer, and nonzero attention, Engram and shared-expert paths. Tests cover
prefill plus two decode calls, all four KV rings, shared KV/index state,
layer-major routes, reset/replay, cache retention, initial and decode capacity
rejection, a late invalid token, aliases, and source failures injected into each
of layers 0, 1, 2 and 3. Failed and rejected calls must not publish prefix progress.

The real case starts with the already validated layer-0 fixture residuals and
incoming coefficients. Original tokens `[0,455,1805,1,20852]` drive the confirmed
Engram table, split into a three-token prefill and two single-token decode calls.
All later inputs are actual preceding-layer outputs, including native Q8_1 MoE.
No layer receives the independently seeded input used by its earlier standalone
test.

The real prefix passes **107 checks**. Final outputs, delayed coefficients,
original expert IDs/weights, all four window rings and the shared compressed
KV/index buffers are bit-exact against the separate-buffer control. Both six-slot
and two-slot expert caches run the sequence twice, with reset. The six-slot run
uses prefill capacity 3 and 4096 cached rows per Engram reader; the two-slot run
uses capacity 7 and disables row caching. All four two-slot caches record 60
misses and 58 evictions over the ten tokens, while output stays unchanged.
The six-slot cache miss counts by layer are `[58,60,60,58]`. Each Engram reader
records 240 requests; warm-row reuse also preserves the exact outputs.

This is an integration comparison against the already validated native
components. It does not add a four-block official CPU/FP8 oracle or establish
full-model quantization quality. Earlier per-component numerical references and
their documented error limits remain applicable to those earlier test inputs.
The prefix test additionally requires finite BF16-valued residuals and finite,
in-range delayed coefficients.

The five-token trace uses **112 original layer/expert combinations**, with
per-layer counts `[27,28,29,28]`. Existing guarded fixtures supply 51 of them;
**61 supplemental experts** are downloaded in bounded ranges from the pinned
revision, preserving all original IDs. Their MXFP4 payloads total 1146839040
bytes. An independent inverse of the GGUF nibble layout reconstructs the original
packed FP4 code bytes and E8M0 scales for every one of the 112 selected experts;
all hashes match their recorded original-source hashes.

The test's `--probe` mode stops before reading any missing selected expert. Exit
code 78 reports the original layer and IDs; it is a request for fixture coverage,
not an accepted inference result. A fresh process reconstructs the sequence
after those weights are supplied. The offline `prepare_prefix_experts.py` checks
the pinned revision/config, tensor shapes and source hashes, then reuses the
existing lossless FP4 repacker. Supplemental files are test-only concatenated
MXFP4 gate/up/down projections with provenance sidecars. Normal inference tests
require complete coverage and never substitute a route or a zero expert.

All **37 selected DeepSeek, native-kernel and cache/I/O regression tests** pass.
Both final synthetic and real runs pass full CUDA memcheck with zero errors and
zero leaked bytes, and racecheck with zero hazards/warnings. Racecheck uses the
step-6i resource settings (`--force-synchronization-limit 16`,
`--racecheck-num-workers 2`), without kernel filters. Strict C++
`-Wall -Wextra -Werror` and offline Python syntax checks pass. No cgroup OOM events
were recorded during this gate. No existing runtime weight loader or numerical kernel was changed.

The feature remains opt-in. The default `strata_core`, `strata_kernels` and
`strata_engine` archive hashes match step 6l with the feature both disabled and
enabled. The main Strata checkout remains clean. The only existing test-helper
change marks reusable header functions inline so the prefix test can use a
subset without unused-function warnings; their arithmetic is unchanged.

### Measured allocation and remaining model work

For compressed-cache capacity 3 groups on the real five-token test:

| Slots per layer | Prefill capacity | Owned device bytes | MiB | Prefix workspace bytes |
| --- | ---: | ---: | ---: | ---: |
| 6 | 3 | 2162041488 | 2061.883 | 492032 |
| 2 | 7 | 1844191888 | 1758.758 | 1147392 |

These are owned device allocations, including all four decoders, expert slots,
Engram projection, KV/index state and prefix workspace. They exclude caller
input/output buffers, CUDA/cuBLAS context and host mappings/staging/caches.
Allocation is fixed across the measured prefill/decode/reset calls. The seeded
five-token run does not demonstrate complete-model execution under the target
64 GB RAM/20 GB GPU budget, and it is not a throughput measurement.

Layers 0 through 3 are now connected. Remaining layers, layer-14 Engram,
embedding and the output head still have to be connected and validated before
text generation. The fixed config makes layers 4 through 7 ratio-2 consumers of
layer 2's KV/index state. Reusing that consumer path for those layers is the next
proposed review gate; it has not started.

### Reproduce step 6m

Use the retained fixtures from steps 5d, 6h, 6i and 6l, including the pinned
layer-1 lookup/projection fixtures. Missing experts identified by `--probe` are
fetched from the fixed revision into the ignored reference cache, then prepared
locally; the test itself performs no network access.

```sh
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --probe build-deepseek-cuda build-deepseek-cuda/prefix-experts
# For the original layer/IDs reported by --probe, after caching pinned tensors:
python3 tools/deepseek41/prepare_prefix_experts.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer1 \
  --layer 1 --experts 127 --output-dir build-deepseek-cuda/prefix-experts
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The preparer refuses existing output files. The ignored `step6m-*` records include
coverage requests, source and fixture hashes, independent expert-payload checks,
test commands, allocation/cache counters, sanitizer logs and feature-isolation
results. Stop at this gate for user confirmation before extending the prefix.

## Step 6n: extend the shared consumers through layer 7

`CudaTextPrefix` now accepts exactly four or eight original layers. The default
weight list still contains four entries; explicitly resize it to eight and supply
layers 0 through 7 for this gate. `layer_count()` and layer-major route output
reflect the chosen prefix. The earlier four-layer entry point remains covered.

### Reuse and state ownership

`CudaSharedAttention` and `CudaSharedDecoder` are the existing layer-3
implementations with an explicit consumer layer, restricted to 3–7. The old
`CudaLayer3Attention` / `CudaLayer3Decoder` names remain aliases with layer 3 as
the default. Each consumer loads its own projection, mHC, router and shared
expert weights, and passes its original layer ID into Strata's expert cache.
No attention, mHC or MoE numerical kernel is added or changed.

All five consumers borrow the same completed layer-2 publication. Layer 2 alone
owns compressed KV, index keys and per-query selected indices. Every consumer
has its own SWA ring; it neither resets the source nor duplicates its compressed
cache. All consumers finish before the next layer-2 call. The prefix resets
consumers first, then the source and earlier layers, keeping expert caches and
counters. Failure in any layer invalidates the complete prefix instance.

Two residual buffers and two coefficient buffers still suffice. Layers 0/2/4/6
write one pair, layers 1/3/5 write the other, and layer 7 writes caller output.
Every token receives the previous layer's residual and delayed coefficients for
that same token. The four-layer path writes caller output at layer 3. Routes have
`layer_count() * T` entries. Layer count, all layer identities/sharing topology,
positions, cache capacity, original tokens and buffer aliases are validated
before layer 0 starts. Layer 8 is explicitly outside this gate.

### Validation on the RTX 2080 Ti

The eight-layer synthetic suite passed **176 checks**. It uses
nonzero attention/shared-expert paths, distinct per-layer expert IDs and
token-dependent mHC coefficients. It checks separate-buffer direct calls against
the two-buffer prefix, prefill then decode, reset, source immutability after each
consumer, late invalid inputs and failure injection at every layer 0–7.

The real-weight suite passed **146 checks** using the pinned revision,
tokens `[0,455,1805,1,20852]`, three-token prefill and two single-token decode
calls. All later layers receive actual previous-layer output. It repeats after
reset with six and two expert slots per layer, and Engram row caching enabled
and disabled. Residuals, delayed coefficients, original top-six IDs/weights,
window histories and layer-2 compressed/index state match direct native calls
bit-for-bit. The eight layers select respectively `[27, 28, 29, 28, 26, 26, 24, 28]` distinct experts
for these five tokens. Every selected expert's packed FP4 codes and E8M0 scales
were independently reconstructed from its GGUF MXFP4 layout and hash-checked
against the original source; no missing expert was replaced by zeros.

The real control is a composition of the confirmed native decoder components
with eight independent destination buffers. It verifies orchestration and state
ownership. It is **not an eight-block official CPU/FP8 numerical oracle**. Earlier
component-oracle and native Q8_1 activation limitations continue to apply.

Further checks completed:

- 38 selected CTests, including both four- and eight-layer prefixes.
- The original four-layer real-weight suite after the shared-class refactor.
- Strict C++ warnings and offline Python syntax checks.
- 70 attention-weight loader checks under ASan/UBSan, including
  explicit layer 4–7 profiles and malformed source/ratio/identity inputs.
- Eight-layer synthetic and real CUDA memcheck: zero errors and zero leaks.
- Eight-layer synthetic and real CUDA racecheck: zero errors and zero warnings;
  no kernel filters, synchronization limit 16 and two racecheck workers.
- Feature OFF/ON rebuilds left `strata_core`, `strata_kernels` and `strata_engine`
  byte-identical to step 6m. Main checkout stayed clean; cgroup OOM counters stayed zero.

### Bounded allocation and fixtures

The following are **prefix-owned device allocations**, measured on the RTX 2080
Ti reporting 22528 MiB. They exclude CUDA/cuBLAS, source host buffers, mapped
files and caller/control buffers. Cache capacity was three compressed groups.
They do not establish full-model capacity, a 20 GB cap, or decode throughput.

| Expert slots per layer | Prefill capacity | Owned device bytes | GiB | Prefix scratch bytes |
| --- | --- | --- | --- | --- |
| 6 | 3 | 3979051792 | 3.705781 | 492032 |
| 2 | 7 | 3342697232 | 3.113129 | 1147392 |

Prefix scratch is independent of layer count for a fixed prefill capacity. The
new consumers add their own weights, SWA state and bounded expert caches.

`prepare_shared_prefix_weights.py` prepares one layer 4–7 dense fixture offline,
checking the pinned config, revision, dtype, shape and SHA256 of all 29 source
tensors. `prepare_prefix_experts.py` now accepts original layers 0–7 and continues
to prepare at most 32 specified experts per call. It preserves original IDs and
hashes and refuses overwrites. The ignored fixed-revision downloader reads only
requested byte ranges; the final coverage contains 165 supplemental experts,
of which 104 were added for this gate. Existing layer-0..3 fixtures are reused.

The dense preparer initially lacked GGUF writer type registrations for I8/BF16.
It was aligned with the existing fixture preparers; all four dense files were
then regenerated successfully. This was a fixture-writing issue, with no change
to runtime math.

```sh
python3 tools/deepseek41/prepare_shared_prefix_weights.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer4 \
  --layer 4 --output build-deepseek-cuda/real-layer4-decoder.gguf
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 8
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 8 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

Use fresh output paths for fixture preparation; existing files are refused.
Ignored `build-deepseek-cuda/step6n-*` files contain fixture/source hashes,
original route coverage, independent expert-byte verification, exact commands,
sanitizer logs, allocation measurements and default-build isolation results.
`step6n-validation.json` is the combined record. This gate stops at layer 7.

## Step 6o: independent layer-8 KV/index owner

This gate reuses the existing source-2 implementation for original source layers
2 and 8. The classes are now `CudaSourceAttention` / `CudaSourceDecoder`; the
`CudaLayer2Attention` / `CudaLayer2Decoder` names remain aliases. Existing calls
default to layer 2. Layer 8 requires an explicit final `layer=8` constructor
argument and matching original-layer weight descriptors. No numerical attention,
compression, mHC or MoE kernel is added or changed.

### Original identity and independent state

Attention, compressor, index-key and index-query loaders accept the explicit
layer-8 profiles and original tensor names. The compressor/key/query descriptors
now carry their original layer ID. The owner rejects mixed source-2/source-8
weights and requires the fixed 40-layer, ratio-2, owned-KV/index topology without
candidate blocks. Only owners 2 and 8 are supported in this gate. The original
layer ID also reaches Strata's existing expert cache and expert source.

Each owner has separate window, compressed-KV and index-key storage, as well as
its own publication lifetime. `SharedAttentionBatch::source_layer()` returns the
original source ID only for a current, successful publication. Advance, reset,
execution failure and destruction expire the relevant publication; rejected
calls preserve it. Existing consumers 3–7 explicitly require source 2, including
when an unrelated source-8 batch has the same position and token count. Source-8
execution and reset do not change source-2 state, and vice versa.

`CudaTextPrefix` still accepts exactly four or eight layers. A new fixture/test
executable captures the actual layer-7 output, and separately drives the existing
eight-layer prefix followed by `CudaSourceDecoder` for layer 8 on the same CUDA
stream. This validates the caller's handoff without adding a public nine-layer
prefix or implementing consumers 9–13.

### Real-weight numerical reference and handoff

The input remains tokens `[0,455,1805,1,20852]`, three-token prefill followed by two
single-token decode calls. `source8_handoff_cuda_test --capture` records the actual
native layer-7 residual and same-token delayed coefficients. The version-1
`DS41PFX7` file contains five I32 tokens, five 20480-element F32 residuals and five
four-element F32 coefficient vectors after its 16-byte header. It is 409716 bytes;
the capture SHA256 is
`f9339c8fc6e01d4a100facc2f757d3703bc5fcc127634d644fa68307ef81d1a0`.

The existing offline source-2 attention/Block fixture preparers now also accept
`--layer 8 --handoff PATH`. They run the pinned layer-8 CPU reference graph on
those actual layer-7 outputs and require the original layer-8 weights. The
reference combines the independent CPU attention/Block path with the existing
native Q8_1 MoE baseline. It is conditioned on native upstream results; it is not
an official nine-block CPU/FP8 or TileLang parity result. Torch remains confined
to offline preparation.

Only 37 dense source tensors and the 26 experts selected by these inputs were
fetched from the fixed revision: 193 verified source tensors in total. The
selected original expert IDs are
`[6,20,35,57,84,85,93,111,126,165,173,179,187,237,240,247,253,265,279,280,299,328,350,360,361,381]`.
Every selected expert's packed FP4 codes and E8M0 scales were independently
reconstructed from its GGUF MXFP4 payload and hash-checked against the original
source. Unpopulated expert IDs remain guarded; sparse holes are not weights.

The real source-8 decoder suite passed **154 checks**, preserving the earlier
numerical tolerances. The live handoff suite passed **78 checks** with six and two
expert slots and reset/replay. Actual layer-7 outputs match the captured inputs
bit-for-bit; layer-8 residuals, delayed coefficients, routes and all three cache
histories match separate source-8 calls bit-for-bit. The upstream source-2 state
remains unchanged by each source-8 call.

Further checks completed:

- 95 source-8 synthetic checks, including 129-token prefill/window wrap,
  capacity preflight, incomplete groups, source identity, cross-group rejection,
  independent reset, failure and publication expiry.
- 39 selected CTests and the original eight-layer real suite (146 checks).
- 80 attention, 30 compressor, 28 index-key and 30 index-query loader checks
  under ASan/UBSan, including malformed layer-8 identities and topology.
- Source-8 synthetic and live real handoff CUDA memcheck: zero errors and leaks.
- Both suites under CUDA racecheck: zero errors and warnings, no kernel filters,
  synchronization limit 16 and two racecheck workers.
- Strict C++ warnings for 11 touched translation units and offline Python syntax.
- Feature OFF/ON rebuilds kept all three default Strata archives byte-identical
  to step 6n. Main checkout stayed clean; cgroup OOM counters stayed zero.

The initial class rename exposed one stale `CudaLayer2Decoder` friend declaration
in `mhc_cuda.hpp`. It was updated to `CudaSourceDecoder`; the subsequent build and
all checks passed. Runtime numerical behavior was unchanged.

### Bounded device allocation and reproduction

Both live configurations use prefill capacity three and three compressed groups.
These are owned device allocations only; CUDA/cuBLAS, mapped sources, source host
staging and caller/control buffers are excluded. The measurements use the RTX
2080 Ti reporting 22528 MiB and do not establish whole-model capacity or speed.

| Expert slots per layer | Existing eight-layer prefix bytes | Source-8 bytes | Combined bytes | Combined GiB |
| --- | --- | --- | --- | --- |
| 6 | 3979051792 | 486247968 | 4465299760 | 4.158634 |
| 2 | 3342041872 | 406621728 | 3748663600 | 3.491215 |

The two-slot prefix allocation here uses prefill capacity three; the earlier 6n
measurement used capacity seven. Repeating that earlier configuration produces
exactly the original 6n allocation and workspace counts.

```sh
cmake --build build-deepseek-cuda --target \
  deepseek41_layer2_decoder_cuda_test deepseek41_source8_handoff_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_source8_handoff_cuda_test \
  --capture build-deepseek-cuda build-deepseek-cuda/prefix-experts \
  build-deepseek-cuda/real-prefix7-handoff.bin
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_decoder_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer8 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 8 --handoff build-deepseek-cuda/real-prefix7-handoff.bin \
  --output build-deepseek-cuda/real-layer8-decoder.gguf \
  --routed-output build-deepseek-cuda/real-layer8-decoder-routed.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_layer2_decoder_cuda_test \
  --layer 8 --real build-deepseek-cuda/real-layer8-decoder.gguf \
  build-deepseek-cuda/real-layer8-decoder-routed.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_source8_handoff_cuda_test \
  --chain build-deepseek-cuda build-deepseek-cuda/prefix-experts \
  build-deepseek-cuda/real-layer8-decoder.gguf \
  build-deepseek-cuda/real-layer8-decoder-routed.gguf
```

Capture and preparation require fresh output paths. The local source tensors and
prior prefix fixtures must already exist. Ignored `build-deepseek-cuda/step6o-*`
files record exact commands, fixed-source and fixture hashes, original routing,
independent expert verification, sanitizer logs and default-build isolation.
`step6o-validation.json` is the combined record. The gate stops after validating
the separate source-8 owner and its actual upstream handoff.

## Step 6p: connect the source-8 consumers through layer 13

`CudaSharedAttention` and `CudaSharedDecoder` now accept original consumers 9–13,
using the same computation as consumers 3–7. Their expected source identity is
fixed by the original layer number. No numerical attention, compression, mHC or
MoE kernel is added or changed. Each consumer retains its own projections, mHC,
router, shared expert, SWA ring and original-ID Strata expert cache.

| Original source | Shared consumers | State owned only by the source |
| --- | --- | --- |
| 2 | 3–7 | Compressed KV, index keys and completed-call indices |
| 8 | 9–13 | Separate compressed KV, index keys and completed-call indices |

A matching token count and position cannot substitute for the correct source.
Consumers reject the other group's publication before changing their output or
ring. Existing publication lifetime, source-instance binding and sequence checks
remain in place. Resetting a consumer affects only its own state.

### Public prefix and bounded orchestration

`CudaTextPrefix` accepts exactly 4, 8 or 14 entries in `TextPrefixWeights::layers`.
The existing four-layer default and eight-layer path remain supported. For the
fourteen-layer path, supply original layers 0–13 plus the explicit
`compressor8`, `key8` and `query8` weight descriptors. The original
`compressor`, `key` and `query` fields still belong to source 2. Layer order,
both sharing groups and source weight identities are checked before table I/O.

The prefix retains two residual buffers and two coefficient buffers, independent
of layer count. At layer 8 it runs the existing source owner, then passes that
completed publication to layers 9–13. Every layer consumes the preceding layer's
actual residual and coefficients for the same token. Routes stay in original
layer order. Cache accessors accept an explicit source ID of 2 or 8, defaulting
to 2; inactive or invalid sources are rejected.

All caller buffer aliases, including source-8 caches, and the common group
capacity are checked before layer 0 executes. Reset clears all consumers before
both owners and the earlier layers, retaining expert caches and counters. Any
execution/I/O failure poisons the entire prefix instance and publishes no
completed call. Consumers are destroyed before their borrowed sources.

### Validation on the RTX 2080 Ti

The fourteen-layer synthetic suite passed **296 checks**. It distinguishes
per-layer routes and same-token coefficients, compares against separate buffers
for every decoder, and checks prefill, decode, both source histories, rejected
inputs, reset/replay and injected source failure at every layer 0–13. The owner
suite also tests both directions of cross-group rejection for every consumer
3–7 and 9–13, preserving output/state and then accepting the correct source.

The real-weight suite passed **227 checks** using the fixed revision and
tokens `[0,455,1805,1,20852]`: three-token prefill and two single-token decode
calls. It repeats after reset with six and two expert slots per layer, and Engram
row caching enabled/disabled. Output residuals, delayed coefficients, all original
top-six IDs/weights, every window history and both compressed-KV/index histories
match explicit native decoder calls bit-for-bit. Those control calls use a
separate destination per layer and verify source immutability after consumers.
The first nine layers retain their prior original expert selections.

The per-layer expert counts for these inputs are `[27, 28, 29, 28, 26, 26, 24, 28, 26, 26, 24, 24, 24, 21]`. All selected
expert payloads were independently inverted from GGUF MXFP4 back to the original
packed FP4 codes and E8M0 scales and hash-checked. Existing fixtures, including the
26 source-8 experts from 6o, are reused. This gate adds 119 supplemental
original experts for layers 9–13. Missing coverage aborts before any expert read;
it is never substituted with zero weights or a different expert ID.

The real control validates native orchestration and ownership. It is not an
official fourteen-block CPU/FP8 oracle. The earlier component references and
native Q8_1 activation baseline still define the numerical scope. Layer-0 input
is seeded; embedding, the output head, layer-14 Engram and generation remain
outside this gate.

Further checks completed:

- 39 selected regression CTests, plus the separate 296-check fourteen-layer suite.
- The original four- and eight-layer real suites, with unchanged owned device
  allocation and scratch measurements in their original configurations.
- 130 attention-weight loader checks under ASan/UBSan, including original
  source-8 profiles, missing topology and wrong source/ratio/identity inputs.
- Fourteen-layer synthetic and real CUDA memcheck: zero errors and zero leaks.
- Both suites under CUDA racecheck: zero errors and zero warnings, without kernel
  filters; synchronization limit 16 and two racecheck workers.
- Strict C++ warnings for seven affected translation units and offline Python syntax.
- Feature OFF/ON builds keep the three default Strata archives byte-identical
  to 6o. Main checkout remains clean; cgroup OOM counters remain zero.

### Resources and fixture preparation

These are prefix-owned device allocations on the RTX 2080 Ti reporting 22528 MiB.
Both sources have capacity for three compressed groups. CUDA/cuBLAS, mapped
sources, host staging and caller/control buffers are excluded. The measurements
are for this five-token test, not full-model capacity, a 20 GB cap or throughput.

| Expert slots per layer | Prefill capacity | Owned device bytes | GiB | Prefix scratch bytes |
| --- | --- | --- | --- | --- |
| 6 | 3 | 6736562640 | 6.273913 | 492032 |
| 2 | 7 | 5622450640 | 5.236315 | 1147392 |

For a fixed prefill capacity, prefix scratch is identical to the eight-layer
entry point. The new layers add their own weights, window state and bounded
expert caches; consumers do not duplicate either source's compressed cache.

`prepare_shared_prefix_weights.py` now accepts consumers 4–7 and 9–13, preserving
their source-specific profile and checking the fixed revision/config, shapes,
dtypes and hashes of all 29 source tensors per layer. `prepare_prefix_experts.py`
accepts original layers 0–13 and retains its limit of 32 explicit experts per
call. Source downloads remain bounded fixed-revision byte-range reads. Neither
tool is part of the native runtime; Torch is confined to offline preparation.

```sh
python3 tools/deepseek41/prepare_shared_prefix_weights.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer9 \
  --layer 9 --output build-deepseek-cuda/real-layer9-decoder.gguf
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 14
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 14 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

Preparation requires fresh output paths and previously fetched local source
tensors. Ignored `build-deepseek-cuda/step6p-*` files contain exact commands,
source/fixture hashes, actual route coverage, independent expert verification,
sanitizer logs and build-isolation evidence. `step6p-validation.json` is the
combined record. This gate stops at original layer 13.

## Step 6q: layer-14 Engram layout and lookup

`EngramHash` and `EngramTable` now accept the original layer ID, exactly 1 or 14.
They reuse the step-6j implementation, including the two Strata `PleReader`
instances, direct file access, bounded row caches and FP8-to-BF16 conversion.
`Layer1EngramHash` and `Layer1EngramTable` remain compatibility aliases; their
existing constructors still select layer 1. No projection, decoder or numerical
kernel changes are included. The connected text prefix remains layers 0–13.

### Original layer identity and sequence state

The pinned `EngramLayout` allocates distinct prime buckets in layer order
`[1,14]`; layer 14 must not reuse the first layer's primes. Its four multipliers
are `[67716810739261,51510806800915,30921347202721,82619226485591]`.
The 24 prime moduli in `engram.hpp` sum to **384016682** rows. Both constant sets
were compared with values generated by the hash-pinned original source.

Both layers consume the same original tokenizer IDs and the same compressed
vocabulary mapping from step 6j. Each instance owns its three-token history and
position. Token zero and EOS retain their text semantics. Dead-token barriers,
contiguous prefill/decode chunks, reset, the 1048576-position bound and validation
before state mutation remain unchanged. Interleaved layer-1 and layer-14 calls
and separate resets preserve each other's state. Equal row numbers in separate
tables retain distinct data and cache statistics.

The map loader checks the requested layer's hash profile, and the table loader
checks its table profile, original tensor names and exact row count. Defaults
remain layer 1. Unsupported layers and cross-layer fixtures are rejected.
Layer 14 uses `1017-layer14-compressed-4gram` and `1017-layer14-fp8-row32`.
Guarded partial tables retain explicit opt-in and sorted original IDs in
`blk.14.engram_populated_rows`. Missing rows are rejected before I/O.

| Original layer-14 tensor | Source shape | Native artifact |
| --- | --- | --- |
| `layers.14.engram.embed.weight` | `[384016682,256]`, FP8 E4M3FN | `blk.14.engram_embed.weight`, I8 `[256,384016682]` |
| `layers.14.engram.embed.scale` | `[384016682,8]`, E8M0 | `blk.14.engram_embed.scale`, I8 `[8,384016682]` |

Each row retains eight scales, one per 32 values. The original two tensor
payloads total **101380404048 bytes (94.417859 GiB)**. They stay on disk; lookup
restores only one token's 24 rows into 6144 BF16 values. Per-instance map payload
is still 517120 bytes, history is 12 bytes, and raw/restored staging is 18624
bytes, excluding object overhead and the readers' configured caches/I/O buffers.
All GGUF mappings are released before direct row reads. This host-only extension
adds no GPU allocation or sequence-length-sized state.

### Bounded original data and independent reference

The source remains revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Layer 14 is in `model-00048-of-00048.safetensors`. Its 664-byte JSON header has
SHA256 `5d8fa4697d7a071c31baeee6947d493c08ea3243261c5d0f79c355ad191b8222`.
Original row `r` starts at file byte `672 + r * 256` for values and
`98308271264 + r * 8` for scales. This differs from layer 1's payload start.

Only **220 original rows / 58080 payload bytes** were fetched using bounded
HTTP 206 ranges from that revision. They cover a nine-token masked sequence and
rows 0, 15, 16 and 384016681. Source offsets and per-row SHA256 records are checked
before preparation. Every populated fixture row was then read back and compared
byte-for-byte with the cached original values and scales.

The existing offline preparer now accepts `--layer 1|14`, defaulting to 1.
It executes the original `NgramHashState` and `ParallelEngramEmbedding.forward`;
only the embedding backing storage maps global IDs to downloaded rows. It selects
the second original hash column for layer 14. Python/Torch remains offline only.
The layer-1 default discovery output is unchanged apart from an explicit `layer`
field. The projection preparer's imported layer-1 constants remain unchanged.

The native hash matches the original reference for prefill lengths 1/3/9,
one-token decode, masked/unmasked input and reset/replay. A separate 257-token
masked oracle matches chunk sizes 1/7/31/257. All downloaded rows, including the
last table row, restore to exactly the original BF16 bits. The reference fixture
is 841088 bytes, SHA256
`94afa09867902b03cb55430c69ba6b4be5ca7a3a4188d4eaf9ed29c15947d154`.
The guarded table has 101380405584 logical bytes but allocates only **1789952
bytes** on the test filesystem. It is not a complete checkpoint.

Measured layer-14 reader counts for the nine-token case, per pass:

| Cache rows per reader | Cold value/scales reads | Replay value/scales reads | Replay hits per reader | Reader bytes on replay |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 216 / 216 | 0 / 0 | 216 | 0 |
| 512 | 216 / 216 | 0 / 0 | 216 | 0 |
| 8 | 216 / 216 | 216 / 216 | 0 | 1769472 |
| 0 | 216 / 216 | 216 / 216 | 0 | 1769472 |

Cold reads also account for 1769472 bytes. These counters include page-sized
I/O, not just row payloads. The 512-row result is specific to this sample's cache
set occupancy; layer 1 retains its previously measured set conflicts. This is
correctness/cache accounting, not a full-model throughput measurement.

### Validation and scope

Checks ran with `STRATA_IO_THREADS=2`, the existing GCC 13.3 builds, and the
RTX 2080 Ti/SM75 CUDA 12.8.93 build for existing GPU consumers. The offline
reference used Torch 2.9.1+cu128. No new CUDA kernel was introduced.

| Check | Result |
| --- | --- |
| Synthetic lookup/state | 208 checks pass across both layers, including independent reset/cache state, wrong-layer files, exact tail bounds, split shards, corrupt rows, short reads and failure poisoning |
| Real layer 14 | 44 checks pass; original hashes and all 220 BF16 rows match exactly at cache capacities 4096/512/8/0 |
| Real layer 1 | 44 checks pass using the existing original fixture |
| CPU ASan/UBSan | Synthetic 208, real layer 1 44 and real layer 14 44 checks pass; no sanitizer reports |
| Related regression | 6 CTests pass: lookup, projection weights, CUDA Engram, layer-1 decoder, four-layer text prefix and Strata `ple_reader_selftest` |
| Real layer-1 decoder | Existing 111 checks pass, including actual Engram projection/decoder integration and cache eviction |
| Strict checks | Both changed C++ translation units pass `-Wall -Wextra -Werror`; offline Python syntax passes |
| Feature isolation | Default core/kernels/engine archives match step 6p before, with the feature OFF and ON; main checkout clean |

The initial synthetic run exposed one test-fixture assumption: truncating the
aligned file by one byte removed trailing padding for layer 14, leaving its
scale tensor complete. The test now accepts omitted padding and rejects removal
of the final payload byte. Runtime bounds logic was unchanged. The failed log
and exact 16-byte padding diagnosis are retained in the step-6q records.

Reproduce with the pinned local sources and provenance-checked row cache:

```sh
python3 tools/deepseek41/prepare_engram_fixture.py \
  --reference-root build-deepseek-cuda/reference-cache --layer 14 --discover
python3 tools/deepseek41/prepare_engram_fixture.py \
  --reference-root build-deepseek-cuda/reference-cache --layer 14 \
  --output build-deepseek-cuda/real-engram14-reference.gguf \
  --table-output build-deepseek-cuda/real-engram14-table.gguf
cmake --build build-deepseek-cuda --target deepseek41_engram_test -j 4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_test
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_test \
  --layer 14 --real build-deepseek-cuda/real-engram14-reference.gguf \
  build-deepseek-cuda/real-engram14-table.gguf
```

Preparation refuses existing output paths. Ignored `build-deepseek-cuda/step6q-*`
records retain the source/row/fixture hashes, commands, test results and feature
isolation checks. Layer-14 projection, decoder/source assembly, longer prefix
integration and text generation remain later review gates. GPU sanitizer runs
and the full 8/14-layer prefix tests from step 6p were not repeated for this
host-only lookup change; the related GPU consumers above were rebuilt and tested.

## Step 6r: layer-14 Engram projection and residual gate

The existing projection and gate now serve original Engram layers 1 and 14.
`read_engram_weights(model, layer)` requires the matching fixed profile and
original tensor names. Its `EngramWeights` views carry the layer ID into
`CudaEngram`. Unsupported IDs are rejected before GPU allocation. Both layers
use the same dimensions, startup conversion, activation quantization, GEMV,
normalization and residual kernel from step 6k. The CUDA diff contains only the
shared type names, identity storage/check and an error message; compute code,
launch dimensions and allocation sizes are unchanged.

`Layer1EngramWeights`, `read_layer1_engram_weights` and `CudaLayer1Engram` retain
source compatibility. Existing aggregate weights default to layer 1. The layer-1
loader rejects layer-14 artifacts. The existing layer-1 decoder and text-prefix
constructors also reject layer-14 Engram weights before allocating their GPU
components or opening the row table. These constructor guards are their only
behavior changes. No layer-14 decoder or source-14 attention is assembled here.

### Original weights and reference boundary

The source remains revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Four projection/gate tensors, totaling **157521920 bytes**, were fetched with
HTTP 206 requests of at most 8 MiB from `model-00048-of-00048.safetensors`.
The original source offsets, dtypes, shapes and header SHA256 use the pinned
layer-14 header from step 6q. All four artifact payloads were subsequently
compared byte-for-byte with their verified download cache.

| Original tensor | Shape and type | SHA256 |
| --- | --- | --- |
| `layers.14.engram.q_weight` | `[4,5120]` BF16 | `e63616e48cda06fcecafc472287e765fcc27f0423537371c88f8647cb0a1c601` |
| `layers.14.engram.k_weight` | `[4,5120]` BF16 | `994c959b8a16b7090c9463b897c792487ba38884a5d57cd1d0b6da379e6da5ee` |
| `layers.14.engram.wkv.weight` | `[25600,6144]` FP8 E4M3FN | `48fcb9eea4edf7fa27d9612b4ff810bdba73b97945471f60c5b9144592e88137` |
| `layers.14.engram.wkv.scale` | `[800,192]` E8M0 | `4977ea51a108dde2552feafae630000463f99be4035621e31eeba92fcd2523af` |

The artifact uses profile `1017-layer14-wkv-fp8-block32` and tensor names
`blk.14.engram.*`. Projection scales remain 32x32 blocks. The row-local embedding
scales and 220-row guarded table from step 6q are reused without changes or new
row downloads. Only 216 of those rows are needed by the projection sequence.

The existing offline preparer accepts `--layer 1|14`, defaulting to 1. It executes
the hash-pinned original `Engram.forward` and `ParallelEngramEmbedding.forward`,
selecting the original layer-14 hash column and weights. The linear reference
remains the independent NumPy FP8 block-32 implementation from step 6k. It does
not reproduce TileLang's reduction order. The layer-1 cache was recovered from
its retained verified artifacts, and its default preparation output remains
byte-identical to step 6k.

The real native path runs `EngramHash(14)` and `EngramTable(14)` through the shared
CUDA projection/gate. It uses the nine original tokens, including token zero,
EOS, normalization collisions and a dead-token barrier. Residuals are seeded BF16
vectors with different stream magnitudes and a zero-residual case. They are
standalone component inputs, not the actual layer-13 decoder output.

Prefill 3 plus single-token decode, prefill 9, reset/replay and row-cache
capacities 4096/0 cover 36 complete calls per layer. Original row IDs, looked-up
BF16 values and restored FP8 activations match their references exactly.
Additional synthetic checks keep both projections resident, give them distinct
q weights and explicit streams, and interleave calls. Each retains its own
weights, scratch, returned views and repeatable result.

### Numerical and resource results

The step-6k limits were retained before running the new fixture: projection and
residual relative L2 <= 0.003 with maximum absolute error <=
`1e-5 + 0.02 * peak(reference)`; end-to-end gate absolute error <= `2e-4`;
gates on identical projected keys use `2e-6 + 2e-5 * abs(reference)` pointwise.
All gates pass. Layer 14 has small differences from the reference, so sampled
bitwise equality is reported separately and is not claimed for that layer.

| Layer-14 boundary | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Projected keys/value | `8.18462273e-5` | `0.015625` |
| End-to-end gates | `1.01470265e-7` | `1.1920929e-7` |
| Gates on identical native projected keys | `7.70021955e-8` | `8.94069672e-8` |
| Final residual | `6.49471257e-6` | `0.00390625` |

Native layer-14 outputs remain byte-identical across chunking, reset and enabled
or disabled row caching. Layer 1 retains its step-6k numerical results, including
sampled output equality to the independent reference in all 36 calls.

One projection still owns **314769424 bytes (300.187515 MiB)** on the GPU:
a 300 MiB BF16 matrix and 196624 bytes of product/activation/output scratch.
Startup expands at most 4096 rows at a time, with a 48 MiB temporary host chunk.
There is no per-forward allocation. Row caches, host mappings, caller buffers,
CUDA context and other model components are excluded. These component bytes do
not demonstrate the complete 64 GB RAM/20 GB GPU target.

Checks ran on the RTX 2080 Ti/SM75 with CUDA 12.8.93, GCC 13.3,
`STRATA_IO_THREADS=2` and Torch 2.9.1+cu128 for the offline reference:

| Check | Result |
| --- | --- |
| Weight layout and identity | 48 checks pass across both layers, including missing/wrong-shaped/wrong-typed/wrong-layer tensors and truncation |
| Synthetic CUDA | 94 checks pass, including distinct resident instances, all original analytic gate cases, aliases, streams and replay |
| Real layer 14 | 264 checks pass across 36 lookup/projection/gate calls |
| Real layer 1 | 264 checks pass using the retained original fixture |
| CUDA memcheck | Synthetic 94 and real layer-14 264 checks pass; 0 errors and 0 leaked bytes |
| CUDA racecheck | Both cases pass; 0 hazards, errors or warnings; no kernel filters |
| CPU ASan/UBSan | All 48 weight-layout checks pass without diagnostics |
| Existing integration | Layer-1 decoder and four-layer text-prefix CTests pass; existing real layer-1 decoder passes 111 checks |
| Strict checks | Seven changed host C++ translation units pass `-Wall -Wextra -Werror`; offline preparer syntax passes |
| Offline compatibility | Default layer-1 preparation remains byte-identical; all four layer-14 payloads match the verified original bytes |
| Feature isolation | Default core/kernels/engine archives match step 6q before, OFF and ON; main checkout clean |

The new reference/weight fixture is 160253792 bytes, SHA256
`eba0e241a1975dda778486edebd704b0fa98fc6019d02218c391c6e663523c8f`.
The native prefix still covers original layers 0–13. The full 8/14-layer prefix
runs were not repeated for these constructor guards; the shared prefix code was
rebuilt and its four-layer integration, including wrong Engram identity, passed.

### Reproduce step 6r

Use the pinned local sources, provenance-checked projection cache and existing
step-6q row cache. Preparation is local-only and refuses an existing output path.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_engram_projection_fixture.py \
  --reference-root build-deepseek-cuda/reference-cache --layer 14 \
  --output build-deepseek-cuda/real-engram14-projection.gguf
cmake --build build-deepseek-cuda \
  --target deepseek41_engram_projection_test deepseek41_engram_cuda_test -j 4
build-deepseek-cuda/src/deepseek41/deepseek41_engram_projection_test
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_cuda_test
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_engram_cuda_test \
  --layer 14 --real build-deepseek-cuda/real-engram14-projection.gguf \
  build-deepseek-cuda/real-engram14-reference.gguf \
  build-deepseek-cuda/real-engram14-table.gguf
```

Ignored `build-deepseek-cuda/step6r-*` records retain commands, source/fixture
hashes, errors, sanitizer results and isolation checks. The next review gate
will establish the source-14 KV/index path before joining it with Engram in a
complete decoder. Full-model generation, quality and throughput remain later work.

## Step 6s: layer-14 KV/index source

The existing `CudaSourceAttention` now accepts original source 14 alongside
sources 2 and 8. It uses the same projections, ratio-2 compressor, key/query
indexer, bounded caches and publication implementation. The CUDA production
change is limited to the supported-owner check and its error text. No compute
kernel, accumulation order, allocation formula or numeric tolerance changed.
The four weight readers require layer-14 profiles/tensor names, original
self-owned KV/index topology and no candidate source. Their old source-2/8
entry points and default source-2 constructor remain available.

Source-14 publications expire on successful advance, reset or destruction.
Rejected calls preserve the previous publication, caches and caller output.
Independent source-8/14 instances retain separate state. Existing consumers
3 and 9 reject a source-14 batch even at matching sequence positions.
`CudaSourceDecoder` still rejects layer 14 before expert I/O; a regression check
now pins this boundary. The existing source-8 test's obsolete "owner 14 is not
enabled" attention assertion now checks the still unsupported source 20.
Engram, the full layer-14 decoder and consumers 15–19 are not joined in this gate.

### Fixed weights and test boundary

Revision remains `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The 21 attention/compressor/index tensors total **142936832 bytes**, downloaded
from `model-00017-of-00048.safetensors` using verified HTTP 206 ranges of at most
8 MiB. Source dtype, shape, offset and SHA256 are retained in
`build-deepseek-cuda/reference-cache/prefix-layer14/provenance.json`.
The local preparer verifies the pinned model/kernel/config hashes and each
payload hash before executing the original Attention/Compressor/Indexer graph
with the existing independent CPU arithmetic.

`real-layer14-attention.gguf` is 152231040 bytes, SHA256
`450da5fbcfc813d3477846cf9385a049d6c7eacb8086d7a5dcad3b81279bf6db`.
Its seeded BF16 inputs cover prefill/decode plans `(1,15)`, `(5,15)`, `(6,15)`,
`(13,15)` and `(129,131)`, including odd pending groups and the 128-slot window
wrap. These are standalone attention inputs, not layer-13 decoder outputs.
The source-14 arena reports **287738112 bytes** with 66 compressed groups and
publication capacity 129, excluding caller buffers and host fixture storage.
This is not a full-model RAM/VRAM estimate or a throughput result.

### Original block-32 discrepancy and isolated cause

The original block-32 reference run finishes **375 checks with 3 failures**: position 6 in plans
0, 1 and 2 has output relative L2 **0.00548263550**, exceeding the unchanged
**0.005** limit. Maximum absolute error is 0.015625. All compared window KV,
compressed KV and index-key histories are byte-identical to the CPU reference.
All other real checks, including publication identity, partition-independent
histories and reset/replay, pass. The failed real command continues to return 1.

Diagnostic copies of the CUDA source captured intermediate tensors without
changing the production build. Input, normalized query, normalized KV and
selected indices match. The first BF16 difference is one Q projection element
(row 1340 of 32768) in head 2:

| Quantity | Value |
| --- | --- |
| Exact dot, checked with integer-scaled products | 0.11596677452325821 |
| Native FP32 accumulation | 0.1159667819738388 |
| CPU block-32 accumulation / BF16 midpoint | 0.115966796875 |
| Native BF16 / exact-dot BF16 rounding | 0.11572265625 |
| CPU block-32 BF16 rounding | 0.1162109375 |

The native result is closer to the exact dot here. The reference's accumulation
lands exactly on a BF16 midpoint and rounds to the other neighbor. Three
attention FP8 values then differ; after the output projections, 112 quantized
low-rank values differ. Identical-input checks confirm both native FP8
quantizers match the independent CPU implementation exactly.

Two diagnostic controls establish causality: replacing just that Q boundary
in the CPU calculation, and independently recomputing all Q dots for that token
in FP64, both reduce final output relative L2 to **6.63423156e-7**, with maximum
absolute error **3.05175781e-5** and one differing output value. Neither control
replaces the formal fixture or changes its pass criteria. They do not prove
bit parity with the original TileLang GPU implementation or full-model quality.

### Initial validation, before the reference follow-up

Validation ran on the RTX 2080 Ti (SM75, reported 22528 MiB), CUDA 12.8 and
Torch 2.9.1+cu128 for offline CPU reference preparation.

| Check | Observed result |
| --- | --- |
| Weight readers | 261 pass: attention 140, compressor 41, index key 39, index query 41 |
| CPU ASan/UBSan | The same 261 pass without diagnostics |
| Source-14 synthetic/publication CTest | 74 pass, including cross-group rejection and publication lifetime |
| Real source 14 | 375 checks, 3 numerical failures; process exits 1 |
| CUDA memcheck on the same real test | 0 errors, 0 leaked bytes; all 3 numerical failures remain |
| CUDA racecheck on the same real test | 0 hazards, errors or warnings; all 3 numerical failures remain; no kernel filters |
| Existing source-2 attention | Real fixture passes 251 checks through the legacy default entry point |
| Existing source-8 attention | Synthetic fixture passes 37 checks |
| Decoder/shared/prefix regressions | Shared attention, layer-2 decoder, layer-3 decoder and four-layer text prefix CTests pass; source-8 decoder passes 112 checks after correcting the obsolete owner-14 assertion |
| Strict compilation | Ten changed host C++ translation units pass `-Wall -Wextra -Werror`; Python preparer syntax passes |
| Feature isolation | Default core/kernels/engine archive hashes match step 6r before, OFF and ON; feature restored ON; main checkout clean |

The initial source-8 regression failure and the corrected rerun are both retained
in the logs. Full 8/14-layer real prefix controls were not repeated for this
standalone source extension; decoder/prefix production code is unchanged.
Layer-14 Engram/decoder integration and original TileLang GPU parity were not
tested. The clean memory/race diagnostics did not resolve the numerical
acceptance issue. The user then authorized the general reference follow-up
below. The original block-32 artifacts and their failing comparisons are kept;
no token-specific correction or tolerance increase is applied.

### Reproduce step 6s

The preparer uses only the verified local cache and refuses an existing output.
The last command below is expected to report the three documented failures.

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_attention_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer14 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 14 --output /tmp/source14-attention.gguf
cmake --build build-deepseek-cuda \
  --target deepseek41_layer2_attention_cuda_test -j 4
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_source14_attention_cuda_test$'
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_layer2_attention_cuda_test \
  --layer 14 --real /tmp/source14-attention.gguf
```

Ignored `build-deepseek-cuda/step6s-*` records preserve initial failures,
verification logs, provenance, source changes and diagnostic evidence. In
`step6s-diagnostic`, `comparison.json`, `dot-comparison.json` and
`control-comparison.json` retain the measured causal chain. Diagnostic helper
scripts are `diagnose-step6s-{native,reference,dot,control}.py` in the build
folder. The production CUDA file still has only the owner-check change above.

### Reference follow-up: explicit FP64 linear accumulation

Step 6s now passes the declared independent CPU reference with **unchanged
numeric thresholds**. `--linear-oracle fp64` selects one uniform rule for all
Linear modules in the attention oracle: `wq_a`, `wq_b`, `wkv`, `wo_b`, the
indexer's `wq_b`, `wk` and `weights_proj`, and compressor `wkv`/`wgate`.

- Keep the original weights, shapes and input FP8 quantization where specified.
  The BF16/F32 dense projections do not acquire FP8 input quantization.
- Accumulate each full-K dot in FP64, cast once to FP32, then to the original
  output type. Compressor projections remain FP32; the other linears return BF16.
  The explicit FP32 boundary is intentional; this is not direct FP64-to-BF16 rounding.
- Preserve grouped BF16 `wo_a` einsum, normalization, softmax, RoPE, sparse attention,
  compression positions and FP4/FP8 conversion. There are no layer-, token- or
  element-specific adjustments and no selection of whichever reference passes.

The FP64 weight scratch uses 256-row tiles, without splitting the K reduction.
The fixture records `fixture.linear_oracle=fp64-linears-fp32-boundaries-v1`;
the native test reports this tag and rejects unknown explicit oracle tags.
The default remains the original `block32` mode. Existing callers and fixtures
retain their old arithmetic. No production C++/CUDA code, kernel, comparison
formula or tolerance changed in this follow-up.

The new mode is a mathematical accuracy control for these component fixtures.
It does not claim the original FP8 GPU accumulation order. Both the old and new
references execute the pinned original graph and retain its typed boundaries.
Offline NumPy/Torch reference work is outside the native inference path.

### Additional source-8 boundary found by broader coverage

The first follow-up changed only FP8 linears. Source 14 then passed 375 checks,
but the newly prepared standalone source-8 fixture exposed a separate compressor
boundary: two latent BF16 values differed at the second compressed group. They
propagated into one index-key value, 0.125 versus 0.1875, and consequently **39
cache comparisons** failed as the cache grew. Its attention output comparisons
passed. The old block-32 and FP8-only FP64 references had identical key failures.
This seed/input suite extends the earlier source-8 decoder and synthetic checks.

Isolated capture showed that, given the same latent, key projection and
normalization matched. Recomputing all compressor projection dots in FP64,
with the original FP32 output, removed both latent differences and the cached
key difference. This is why the final mode uniformly covers all Linear modules,
including the compressor's FP32 projections. The final source-8 key histories
match the native cache byte-for-byte. The diagnostic copies and initial failed
runs remain under `build-deepseek-cuda/step6s2-*`.

### Final numerical acceptance

All five original prefill/decode plans and the seeded 131-token input are kept.
For each source, all 21 weight tensors and the input are byte-identical between
the two reference modes. All **124 cache comparisons per source** against the
final FP64 reference are byte-identical, including window, compressed KV and
index keys. The existing publication, rejection and replay checks also pass.

| Original source | FP64-reference native checks | Largest output relative L2 |
| --- | --- | --- |
| 2 | 251 pass | 0.00304960047 |
| 8 | 338 pass | 0.00100886719 |
| 14 | 375 pass | 0.00242325163 |

Output relative L2 remains bounded by 0.005, with the existing maximum-absolute,
finite and BF16 checks. Compressed/key comparisons retain their 0.003 and
maximum-absolute limits. Eight model-independent Python tests cover exact
rational dots, cancellation, both BF16 midpoint directions, the FP32 boundary,
dense F32/BF16 output types, unquantized dense inputs, row-tile tails and invalid
oracle/bias rejection. Strict compilation of the changed C++ test and syntax
checks of the changed Python tools pass.

The original source-2 and source-14 fixtures reproduce their pre-follow-up
SHA256 exactly; the newly introduced source-8 block-32 artifact also reproduces
its first-run hash. Complete native logs against all three block-32 artifacts
are byte-identical before and after the uniform reference implementation.
Those controls still report source 2 passing, source 8 with 39 key differences,
and source 14 with the original 3 output differences. They are recorded as
legacy-reference discrepancies, not silently converted to passing tests.

The final FP64 artifacts are:

| Source | Bytes | SHA256 |
| --- | --- | --- |
| 2 | 152231072 | `40558dea52eb416ecf65741cb4ccf5b795a4c0a96173eb478219955139f35c41` |
| 8 | 152231072 | `f5b5a19b04e6fc313d0c18a06844b809d72a735461e190261d069c95e2793b3f` |
| 14 | 152231136 | `97c128129e0a2ad7e5ff032b4e62b1d4644cadf6df6b9fadcef5365cf0fb5c18` |

All production source/header files and the default core/kernels/engine archive
hashes match the pre-follow-up snapshot. The main checkout remains clean.
GPU sanitizers, CPU reader sanitizers and decoder/prefix tests were not repeated:
this follow-up changes offline references and the test's metadata reporting;
production execution, weights and inputs are unchanged. Their previous step-6s
results above remain separately recorded. Layer-14 decoder/Engram integration,
full-model quality, original GPU bit parity and throughput are still untested.

### Reproduce the accepted reference

```sh
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover \
  -s tests/deepseek41 -p test_reference_linear.py -v
OPENBLAS_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_attention_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/prefix-layer14 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 14 --linear-oracle fp64 --output /tmp/source14-attention-fp64.gguf
cmake --build build-deepseek-cuda \
  --target deepseek41_layer2_attention_cuda_test -j 4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_layer2_attention_cuda_test \
  --layer 14 --real /tmp/source14-attention-fp64.gguf
```

The same commands use `--layer 8` with `reference-cache/prefix-layer8`, or
`--layer 2` with `reference-cache/attention2-restored`. Source-2's 21 original
payloads were restored losslessly from the existing GGUF and each checked
against its recorded original tensor hash; no new checkpoint download was needed.
The final artifacts are `step6s2-final-layer{2,8,14}-fp64.gguf`. Preparation still
refuses existing outputs. `step6s2-final-{fixtures,native-validation,fixture-audit}.json`
and `step6s2-validation.json` retain the measured results, versions and hashes.

## Step 6t: standalone layer-14 decoder with Engram

The user's next continuation confirmed step 6s and authorized this single gate.
`CudaLayer14Decoder` combines the existing layer-14 Engram lookup/projection,
source attention, mHC and native MoE. It introduces no new mathematical kernel.
It uses original layer-14 experts and Strata's bounded row and expert caches.
The plain `CudaSourceDecoder` still accepts only sources 2 and 8, so layer 14
cannot silently skip Engram.

### Execution and state

The original source-attention prefill needs all compressed groups before its
attention pass. The decoder therefore applies Engram twice: first to supply
normalized compressor inputs, then to supply the attention and FFN residual path.
Two startup-owned hash objects consume identical original token sequences.
They share one Engram table reader and projection instance. This avoids a
prompt-sized `[T,4,5120]` intermediate; the decoder's own scratch is **208896
bytes**, independent of prompt length. Attention's compressed cache and optional
publication storage remain explicitly capacity-bounded. Repeated Engram work
is an accepted correctness-first cost; its performance has not been benchmarked.

All token IDs, positions, capacities, streams and buffer aliases are checked
before either hash advances or row/expert I/O starts. Only a complete residual
and MoE call publishes source-14 state. Rejected arguments preserve a completed
publication. Advance, reset, destruction or execution failure expire it; an I/O
or execution failure requires reconstructing the decoder. Reset clears both hash
histories and all attention state, while retaining row/expert caches and counters.
Consumers 3 and 9 reject source 14. Consumers 15–19 are still unavailable.

### Reference and results

The standalone reference uses the first five all-text tokens of the confirmed
layer-14 Engram projection fixture: `[0,455,1805,1,20852]`, partitioned as a
three-token prefill and two single-token decodes. Its residual is seeded; it is
not an actual layer-13 output. The Engram fixture SHA-256 is
`eba0e241a1975dda778486edebd704b0fa98fc6019d02218c391c6e663523c8f`.
The pinned original Block graph supplies mHC, attention, FFN inputs, routes and
residual-join coefficients. Attention uses step 6s's explicit
`fp64-linears-fp32-boundaries-v1` CPU reference. The final join uses the declared
native Q8_1 MoE baseline, as in earlier decoder gates. No thresholds were widened.

The original graph selects 29 distinct experts. Preparation and an independent
local audit verify all **211 source tensors / 731465944 payload bytes**, including
174 tensors for those experts. Missing experts remain errors. The sparse routed
fixture occupies 549519360 disk bytes for a 7223419648-byte logical address space.
The decoder fixture has 184473664 bytes and SHA-256
`3b642dc8eeabfc74da2d9fcd5f54439930b0ed615aa12d909f8e37c8569af5cf`.

Measured on the RTX 2080 Ti (SM75, reported 22528 MiB), CUDA 12.8, with
Torch 2.9.1+cu128 used only by the offline CPU reference:

| Check | Observed result |
| --- | --- |
| Synthetic composition and boundaries | 112 checks pass: 129-token prefill plus decode, ring wrap, odd pending group/reset, capacity rejection, aliases, identities, publication lifetime, row/expert failure |
| Real complete layer | 147 checks pass with 6/2 expert slots, row cache enabled/disabled and reset/replay |
| Complete output against declared composition baseline | Maximum relative L2 **0.000354727532**, maximum absolute error **0.0625**; existing limits pass |
| Strict same-input next-layer mHC coefficients | Maximum absolute error **8.94069672e-8**; original propagated-input bound also passes |
| Window KV, compressed KV and index keys | All 36 real cache comparisons are bit-exact |
| Reset and cache policy invariance | Output and delayed coefficients are bit-exact across both replay and cache configurations |
| Existing native paths | Four CTests pass: layer 1, source 2, source 8 and four-layer prefix; real source-14 attention 375, source-2 decoder 154, source-8 decoder 154 and layer-1 decoder 111 checks pass |
| Offline compatibility | Layer-8 decoder fixture regenerated byte-identically; source-2 default discovery is byte-identical to the pre-step script using verified local tensors |
| Host checks | New C++ implementation/test pass `-Wall -Wextra -Werror`; changed Python syntax and all 13 existing offline tests pass |
| Feature isolation | Feature OFF/ON builds preserve all three default Strata archive hashes from step 6s; feature restored ON; main checkout clean |
| GPU memcheck | Synthetic 112 and real 147 checks pass; both report 0 errors and 0 leaked bytes |
| GPU racecheck | Real 147 checks pass; 0 hazards, errors or warnings, with no kernel filters |

The real test performs two five-token replays per configuration. Six slots use
**801123888 device bytes**, with 58 expert misses, 2 hits and 52 evictions. The
4096-row cache per table half performs 240 row reads and 720 hits, reading
983040 bytes. Two expert slots with row caching disabled use **721497648 device
bytes**, with 60 misses and 58 evictions; 960 row reads transfer 3932160 bytes.
The two Engram passes account for 480 requests per table half in either case.
These are owned device allocations and counters for this standalone layer with
three compressed-group slots and publication capacity three. They exclude CUDA,
cuBLAS, caller buffers, host mappings and test/reference storage. They are not a
full-model memory estimate or a throughput measurement.

### Reproduce step 6t

The local preparer refuses existing output paths. It requires the verified
layer-14 cache, including all discovered experts.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/deepseek41/prepare_layer2_decoder_fixture.py \
  --layer 14 --model-dir build-deepseek-cuda/reference-cache/prefix-layer14 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --engram-fixture build-deepseek-cuda/real-engram14-projection.gguf \
  --output /tmp/layer14-decoder.gguf --routed-output /tmp/layer14-routed.gguf
cmake --build build-deepseek-cuda --target deepseek41_layer14_decoder_cuda_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_layer14_decoder_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_layer14_decoder_cuda_test --real \
  /tmp/layer14-decoder.gguf /tmp/layer14-routed.gguf \
  build-deepseek-cuda/real-engram14-projection.gguf \
  build-deepseek-cuda/real-engram14-reference.gguf \
  build-deepseek-cuda/real-engram14-table.gguf
```

Per-step commands, hashes, numerical metrics, cache counters and validation logs
are retained under `build-deepseek-cuda/step6t-*`; `step6t-validation.json` records
the completed gate. The connected prefix remains
original layers 0–13. This gate does not establish official TileLang GPU parity,
full-model quality, generation, throughput or the 64 GB RAM/20 GB GPU target.
CPU reader sanitizers and full 8/14-layer real prefix controls were not repeated;
their production implementations are unchanged.

## Step 6u: connect the native prefix through layer 14

The user's next continuation confirmed step 6t and authorized this integration
gate. `CudaTextPrefix` now accepts exactly **4, 8, 14 or 15** original layers;
the default remains four. The fifteen-layer path calls the confirmed
`CudaLayer14Decoder` after layer 13, passing its actual four-stream residual,
same-token delayed mHC coefficients and the original text token IDs.

### Ownership and bounded execution

`TextPrefixWeights` adds explicit `engram14`, `compressor14`, `key14` and
`query14` fields. Fifteen-layer construction also requires the final
`token_map14` and `table_path14` arguments. The layer order, source-14 topology,
weight identities and presence of these inputs are checked before layer-1 table
I/O. Existing four/eight/fourteen-layer calls require no new arguments.

Layers 1 and 14 own separate Engram hash histories and row caches. Sources 2,
8 and 14 own separate compressed KV/index state. Source 2 still feeds consumers
3–7 and source 8 feeds consumers 9–13. Layer 14 keeps its confirmed two-pass
Engram/source-attention execution. Consumers 15–19 are not enabled in this gate.
The prefix retains the same two residual buffers and two coefficient buffers;
layer 14 writes the caller's final output. No new math kernel is introduced.

Preflight checks now cover source-14 cache aliases as well as the common group
capacity, all original tokens and all earlier caches. A rejection occurs before
layer 0 executes. `engram_stats(layer)` accepts active Engram layers 1 and 14,
defaulting to 1; compressed-cache accessors accept active sources 2, 8 and 14,
defaulting to 2. No shared publication escapes the prefix owner. Reset clears
both Engram histories and all three sources with their consumers, preserving
bounded weight caches and accounting. Failure in any layer, including an Engram
row read at layer 14, poisons the whole prefix and publishes no completed call.

### Reference scope and validation

The synthetic fixture uses different token maps for the two Engram layers, so
incorrectly substituting layer 1's input is observable. Its explicit native
control allocates a different output buffer for every layer. The production
prefix instead uses two reusable buffers. Tests cover same-token coefficients,
original route order, three source histories, capacity and alias rejection,
reset/replay, injected expert failure at every original layer 0–14 and a
layer-14 row-I/O failure after earlier layers have executed.

The real control uses fixed revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` and original tokens
`[0,455,1805,1,20852]`: three-token prefill, then two single-token decodes.
Layer 0 receives its earlier seeded input. Every later layer receives the
preceding native decoder's actual residual and coefficients. Layer-14 routes
are discovered from that handoff; the standalone seeded layer-14 fixture's
expert selection is not substituted for them. Missing original experts abort
before any expert read, and only the required fixed-revision payloads are added.

The synthetic fifteen-layer suite passes **344 checks**. The real suite passes
**256 checks** with six and two expert slots per layer, Engram row caching
enabled/disabled and reset/replay. Final residuals, same-token coefficients,
every original route ID/weight, all fifteen window histories and all three
compressed-KV/index histories are bit-exact against the separate-buffer control.
The full-prefix synthetic CUDA memcheck passes all 344 checks with **0 errors
and 0 leaked bytes**. Real CUDA memcheck passes all 256 checks with **0 errors
and 0 leaked bytes**. Real CUDA racecheck passes all 256 checks with **0 hazards,
errors or warnings**, without kernel filters.

Existing 4/8/14-layer synthetic suites pass 138/190/300 checks. Their real suites
pass 110/146/227 checks; the fourteen-layer device/scratch and row-I/O resource
measurements exactly match step 6p. Three affected C++ translation units pass
`-Wall -Wextra -Werror`; the offline preparer syntax and all 13 existing Python
tests pass. Feature OFF/ON builds preserve the default core/kernels/engine
archive hashes from step 6t, with the feature restored ON and main checkout clean.

An initial synthetic setup attempt changed the mandatory padding entry while
making the two token maps distinct. The existing `2 -> 2` check rejected it
before any forward call. Only that test input was corrected; the initial log is
retained and the complete rerun passes. The new CTest has a 1200-second limit
following the measured 682-second passing run under the initial 720-second limit.
No numerical threshold or production input check was relaxed. CPU reader
sanitizers were not repeated because their production code is unchanged;
synthetic memcheck and real full-prefix memcheck/racecheck cover this integration.

Layer 14 selects 25 distinct original experts, six already covered by the
standalone fixture and **19 supplemental experts / 357212160 payload bytes**.
The first fourteen layers retain exactly their earlier selected expert sets. All
**386 actually selected experts** across the fifteen layers were independently
inverted from GGUF MXFP4 to original FP4 code bytes and E8M0 scales and hash-checked.
The per-layer counts are `[27,28,29,28,26,26,24,28,26,26,24,24,24,21,25]`.

Measured owned device allocations on the RTX 2080 Ti (SM75, reported 22528 MiB),
with three compressed groups per source:

| Expert slots per layer | Prefill capacity | Owned device bytes | Prefix scratch bytes |
| --- | --- | --- | --- |
| 6 | 3 | 7537686528 | 492032 |
| 2 | 7 | 6343948288 | 1147392 |

The two scratch buffers retain the earlier size at each prefill capacity.
Layer-14 Engram accounts for 960 total row requests across both table halves
and both five-token replays. With row caching enabled it performs 240 reads and
720 hits, transferring 983040 bytes; without caching it performs 960 reads,
transferring 3932160 bytes. Layer 1 keeps its independent accounting. These
measurements exclude CUDA/cuBLAS, caller/control buffers, host mappings and
staging; they are not a full-model capacity or throughput result.

This is an independent orchestration control using the confirmed native
components. It is not an official fifteen-block CPU/FP8 or TileLang GPU oracle.
The earlier component references and native Q8_1 MoE baseline still define the
numerical scope. Full-model quality, embedding, the output head, generation,
throughput and the 64 GB RAM/20 GB GPU target remain unvalidated.

### Reproduce step 6u

The fixture directory must contain all earlier decoder fixtures plus the
confirmed layer-14 decoder, routed-expert and Engram fixtures. Supplemental
experts must cover the original IDs discovered by the probe. The offline
`prepare_prefix_experts.py` now accepts original layers 0–14; its existing
32-expert bound, revision/config checks and refusal to overwrite outputs remain.

```sh
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 15
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 15 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

Per-step commands and evidence are retained under `build-deepseek-cuda/step6u-*`;
`step6u-validation.json` records the completed gate.
This gate stops at original layer 14; extending the source-14 consumer group is
a separate review gate.

## Step 6v: extend the source-14 consumer group through layer 19

The user's continuation confirmed step 6u and authorized this gate. The pinned
1017 configuration assigns original consumers 15–19 to source 14 with compression
ratio 2 and no candidate blocks. Layer 20 starts a different ratio-1 group and
is outside this change.

`CudaSharedAttention`, `CudaSharedDecoder` and the attention reader now accept
explicit original consumers 15–19. Their source is 14. Their own projections,
window rings, mHC and MoE remain unchanged. The new consumers borrow the completed
source-14 compressed KV and per-query indices; they allocate no compressor,
indexer or duplicate compressed/key cache. Source layers 8 and 14 and future
layer 20 still cannot be constructed as shared consumers. Profiles and tensor
names preserve the original layer IDs.

`CudaTextPrefix` accepts exactly 4, 8, 14, 15 or 20 layers; its default remains
four. In the twenty-layer path, layer 14 writes its actual residual and same-token
delayed coefficients into the next reusable buffer and publishes its shared
batch. Layers 15–19 consume that batch in original order. Layer 19 writes the
caller output. The existing two residual buffers and two coefficient buffers
retain their size for a given prefill capacity. Reset clears the consumers
before their owners. Preflight, whole-prefix progress and execution-failure
rules extend to all twenty layers. No math kernel was added or changed.

The separate-buffer native control now executes all twenty decoders and checks
that consumers leave the three sources' histories unchanged. Synthetic tests
cover distinct layer-1/layer-14 token maps, actual handoffs, all original routes,
cache alias and capacity rejection, reset/replay, and expert failure at every
layer. A layer-14 table-read failure still invalidates the whole call. A separate
three-source test checks wrong-source rejection for every consumer, publication
expiry and source-14 sequence changes without changing consumer output/state.

Real fixtures retain the same fixed revision and five original tokens used in
step 6u. The five new dense fixtures contain original attention, mHC, shared
expert and router weights. Missing routed experts are discovered from the actual
native handoffs and prepared with the existing bounded, hash-checked tooling.
Normal verification rejects missing coverage before expert reads. The explicit
`--prepare` test mode pauses on the exact missing original route. An external
preparer fetches and hash-checks bounded fixed-revision expert files, then sends
`READY`. The test verifies all acknowledged file lengths before any expert read
and resumes that same call; it does not restart earlier layers or change routing.
No expert is replaced or fabricated. This verifies native orchestration and ownership. It does not provide an
official twenty-block FP8/TileLang oracle or full-model generation/quality result.

### Completed validation

The twenty-layer synthetic suite passes **456 checks**, including failure
injection at every original layer. The real suite passes **331 checks** with
six and two expert slots per layer, row caching enabled/disabled and reset/replay.
Final residuals, same-token coefficients, every original route ID/weight, all
twenty window rings and all three compressed-KV/index histories are bit-exact
against the separate-buffer native control.

Synthetic CUDA memcheck passes all 456 checks with **0 errors and 0 leaked
bytes**. Real CUDA memcheck passes all 331 checks with **0 errors and 0 leaked
bytes**. Real CUDA racecheck passes all 331 checks with **0 hazards, errors or
warnings**, without kernel filters. These are correctness checks, not throughput
measurements.

The 4/8/14/15-layer synthetic regressions pass 140/192/302/346 checks. The real fifteen-layer
control passes 256 checks; its device/scratch and complete expert/Engram accounting
exactly match step 6u. Component regressions pass: `deepseek41_attention_weights_test` (191 checks), `deepseek41_shared_attention_cuda_test` (51 checks), `deepseek41_layer3_decoder_cuda_test` (94 checks), `deepseek41_layer8_decoder_cuda_test` (149 checks).
The changed attention reader also passes 191 checks under CPU ASan/UBSan
with leak detection. Seven affected C++ translation units pass strict warning
checks; the two users of the fixture were rechecked after its alignment fix and explicit
preparation protocol change.
Both changed Python preparers parse, and all 13 existing offline Python tests pass.
Feature OFF/ON builds preserve the default core/kernels/engine archive hashes
from step 6u; the feature is restored ON and the main checkout remains clean.

The five new dense fixtures contain **145 tensors**. Their actual GGUF names,
shapes, types and payload hashes were independently checked against the original
cached safetensors and fixed-revision provenance. The new layers selected
**125 original experts / 2350080000 payload bytes**.
All **511 actually selected experts** across the twenty layers were independently
inverted from standard GGUF MXFP4 into original FP4 code bytes and E8M0 scale
hashes. The first fifteen layers retain their earlier selected expert sets.
Per-layer counts are `[27, 28, 29, 28, 26, 26, 24, 28, 26, 26, 24, 24, 24, 21, 25, 25, 25, 25, 22, 28]`.

Measured owned device allocations on the RTX 2080 Ti (SM75, reported 22528 MiB),
with three compressed groups per source:

| Expert slots per layer | Prefill capacity | Owned device bytes | Prefix scratch bytes |
| --- | --- | --- | --- |
| 6 | 3 | 9808949408 | 492032 |
| 2 | 7 | 8217079968 | 1147392 |

Prefix scratch stays at its previous size for the same prefill capacity. These
figures exclude CUDA/cuBLAS, caller/control buffers, host mappings and staging.
They do not establish full-model capacity under the 64 GB RAM/20 GB GPU target.

The initial synthetic attempt stopped before any forward call: the independent
control's fixed 4096-byte alignment allowance was 64 bytes short for twenty-one
residual/coefficient pairs. Only the test control's arena sizing was corrected
to align each pair explicitly. Production arena sizing, numerical thresholds
and input validation were unchanged. The initial failure and passing rerun are
both retained under `build-deepseek-cuda/step6v-*`.

### Reproduce step 6v

Use the same earlier fixtures, the five new original dense fixtures, and guarded
expert blobs covering the IDs recorded by the probe. The preparers preserve
fixed revision/config checks, source hashes and refusal to overwrite outputs.
The expert preparer now accepts original layers 0–19 with the same 32-expert
bound; the dense preparer additionally accepts original consumers 15–19.

```sh
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test -j4
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 20
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 20 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

Commands, hashes, initial failures and completed checks are recorded in
`build-deepseek-cuda/step6v-validation.json` and the adjacent per-step logs.
Layer 20, later groups, embedding, the output head, generation, full-model
quality and throughput remain outside this gate.


## Step 6w: source-20 ratio-1 and candidate contract

Complete; user confirmed. Step 6v was confirmed.
The native prefix still executes original layers 0–19. This gate adds a portable
source-20 contract and pinned reference fixtures; it does not execute a native
layer-20 compressor, indexer, attention or decoder.

The fixed config has **40 backbone layers (0–39)** and **3 additional MTP
prediction layers**. MTP is outside the initial text-generation path. The
remaining backbone work starts at original layer 20, which is the 21st layer.

### Ratio-1 boundary

The source-20 compressor uses original BF16 `wkv` weights `[512,5120]` and BF16
RMSNorm weights `[512]`. It projects each input token to one 512-value latent.
The projection rounds to BF16 before the reference RMSNorm with epsilon `1e-20`;
the output is BF16 and has not yet received RoPE. There is no `wgate`, pooled
pair, pending odd token, `kv_state` or `score_state`.

`Source20CompressorWeights` is a separate mapped type. It reuses Strata's
`GgufModel` and bounds checks, requires the original source-20 topology and
profile, and rejects malformed shapes/types, truncation and an unexpected gate.
The existing ratio-2 reader and CUDA implementation remain unchanged.

`Source20Batch` validates bounded initial prefill followed by contiguous
single-token decode. A call at `start` with `tokens` inputs emits `tokens`
latents at `start + i`; cache width becomes `start + tokens`. Query `i` sees
`start + i + 1` keys, including its own token. It validates the expected next
position and context/prefill limits before exposing dimensions. It describes a
call, and does not mutate a cache or publish runtime state.

### Candidate representation and ownership

The pinned candidate function takes scores that are already causally masked.
It uses the maximum score of each eight-position block, pins the newest visible
block to positive infinity, and retains at most 2048 reachable blocks. A block
whose remaining score is negative infinity is dropped. A tie at the cutoff
allows any valid tied subset; tests do not assume PyTorch's tie ordering.

The proposed compact row stores sorted original block IDs followed by `-1`.
Its stride is `min(2048, ceil(key_count / 8))`. The host validator checks ordering,
padding, reachability, the forced newest block, cardinality and the block-score
cutoff. It is a fixture/debug validator, not a CPU selection fallback. No GPU
candidate kernel or runtime storage is introduced in this gate.

A selected block includes all of its positions in the reference mask. A partial
newest block can therefore include future positions. Consumers must separately
apply the causal boundary before their final top-512 selection. Packed IDs
reconstruct every bit of the reference mask, including those future bits.

The subsequent native implementation must preserve the reference ownership:

- Layer 20 generates index keys from the unrotated latent before attention
  applies KV RoPE and quantization in place. It owns KV and index-key histories
  reused through layer 39.
- Layer 20 publishes candidate rows for the current call. It computes its own
  top-512 directly from its original causal scores, without consuming its own
  candidate filter. Layers 21–23 reuse its selected indices.
- Index-source layers 24, 28, 32 and 36 reuse source-20 keys and candidates,
  compute their own query scores, and select their own top-512. Their following
  consumers reuse that selection; these sources do not generate new index keys.
- Candidate rows belong to one call, unlike the persistent key histories.
  They must remain valid until all downstream consumers of that call finish.
  Reset invalidates the publication. Runtime ownership and stale-publication
  checks remain for the native integration gate.

### Validation and limits

Only two original compressor tensors were fetched from the fixed revision:
**5243904 payload bytes**. An independent safetensors header/offset parse and
GGUF payload comparison verified their shapes, BF16 types and every source byte.
The fixed checkpoint index also confirms that source 20 has no compressor gate.

The offline preparer extracts the unmodified pinned `Compressor`, `RMSNorm` and
`select_candidate_blocks` definitions. With PyTorch `2.9.1+cu128` on CPU, it
exports **9 compressor plans and 58 calls**, including odd/even prefills,
continuous decode, zero/tiny/large inputs and 58 position-invariance checks.
Both the intermediate projection and output are stored as BF16. These outputs
are a reference for a later native implementation; this gate has not compared
native source-20 arithmetic against them.

The candidate fixture has **20 synthetic score rows through the official
function**, including widths 7/8/9, 16384/16385/16393, partial blocks, unreachable
interior blocks, forced newest blocks and cutoff ties. One width-1048576 row
checks the model's maximum context boundary. It does not allocate a quadratic
full-prefill mask or establish a full-context memory budget. Candidate scores
are synthetic; they are not a real full-indexer execution.

Completed checks:

- Portable C++ tests: **134 checks, 0 failures**.
- Pinned fixture interface and candidate-mask checks: **113 checks, 0 failures**.
  These cover output shape/type and candidate semantics, not native compressor
  numerical parity.
- Both suites pass CPU AddressSanitizer/UndefinedBehaviorSanitizer with leak
  detection. The original artifact and ratio-2 weight-reader regressions pass
  in both the Release build and the sanitizer build.
- Both new C++ units pass `-Wall -Wextra -Werror`. Existing Python tests:
  **13 passed**. Fixture identity, overwrite and encoding guards: **11 passed**.
- Feature-OFF and feature-ON builds leave default `strata_core`,
  `strata_kernels` and `strata_engine` archives byte-identical to step 6v.
  The main checkout stays clean. Existing CUDA execution sources are unchanged;
  GPU integration and Compute Sanitizer were not rerun for this host-only gate.

Reproduce with the bounded local source-20 cache and pinned reference:

```sh
python3 tools/deepseek41/prepare_source20_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --output /tmp/source20-fixture.gguf
cmake --build build-deepseek-cuda --target deepseek41_source20_contract_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_source20_contract_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_contract_test \
  --real /tmp/source20-fixture.gguf
```

The preparer is offline and refuses to overwrite its output. Evidence is in
`build-deepseek-cuda/step6w-validation.json` and adjacent per-step logs.
This gate was confirmed. Step **6x** below implements the native source-20
ratio-1 compressor against these exported outputs. Candidate execution,
source-20 decoder assembly and a longer text prefix remain later review gates.

## Step 6x: native source-20 ratio-1 compressor

Complete; user confirmed. Step 6w was confirmed.
`CudaSource20Compressor` now executes the standalone original layer-20
projection and RMSNorm on the RTX 2080 Ti. The connected text prefix remains
layers 0–19; layer-20 KV post-processing, indexer, candidates, attention and
decoder assembly remain separate gates.

### Reuse and execution contract

The new component is a C++ caller of existing Strata kernels. It adds no CUDA
kernel and changes no existing matrix, conversion or normalization kernel:

1. Convert F32 inputs to BF16 using `f32_to_bf16_bulk`.
2. Project with existing `bf16_gemv`, retaining the original BF16 weight matrix.
3. Round the F32 accumulation result to BF16 using the existing DeepSeek rounding
   helper, before normalization.
4. Run `native_qsa_rms_norm_weighted` with epsilon `1e-20` and round its output
   to BF16 again. The external output stores these values in F32.

Each token emits one pre-RoPE latent, including an odd final prefill token. The
returned `Source20Batch` carries ratio-1 positions and cannot be passed as an
old ratio-2 `CompressedBatch`. The component uses the confirmed source-20 weight
view and Strata's `DeviceArena`. There is no gate, incomplete pair, KV history
or candidate allocation.

Construction fixes context and prefill limits. Calls require one explicit
stream and aligned, disjoint external device buffers. The initial call is a
bounded prefill; later calls are contiguous single-token decode. Reset keeps
the same stream and invalidates the diagnostic last-projection view. Scratch
is overwritten on each token. Successful calls allocate no additional buffers
and synchronize before returning. Rejected arguments preserve state and output.

The read-only `last_projection()` view exposes the last token's BF16-valued
projection before RMSNorm. It is null initially and after reset, and is useful
for checking the rounding boundary independently of normalized output.

### Measured validation

Validation used the existing, unchanged step-6w fixture from revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, with 9 plans and 58 calls. The CPU
reference runs the pinned official BF16 projection and RMSNorm definitions.

| Comparison | Maximum per-call relative L2 | Maximum absolute error | Differing comparison values |
| --- | ---: | ---: | ---: |
| Last-token BF16 projection | 6.13636038178e-6 | 0.000244140625 | 2 / 29696 |
| All emitted pre-RoPE outputs | 6.56031542162e-6 | 0.0001220703125 | 6 / 55296 |

The native and CPU implementations have different F32 accumulation orders.
These results satisfy the declared numerical checks; they are not bitwise
official GPU parity. Before execution, projection thresholds were set to
`relL2 <= 0.001` and `max_abs <= 1e-6 + 0.01 * peak`; normalized-output
thresholds were `relL2 <= 0.003` and `max_abs <= 1e-5 + 0.02 * peak`.

Completed checks:

- **217 synthetic checks**, including an independent scalar projection/RMSNorm
  reference, explicit BF16 ties-to-even and input-rounding witnesses, odd/even
  prefill partitions, continuous decode, reset, capacity, overlap and alignment
  rejection, unchanged input/output guards and fixed memory use. Numeric outputs
  in these synthetic cases matched the scalar reference exactly.
- **216 pinned-weight checks** across all 58 reference calls. Native outputs
  are byte-identical across the six equivalent prefill/decode partitions.
  Zero, tiny and large inputs also pass.
- Synthetic and real Compute Sanitizer memcheck: **0 errors, 0 leaked bytes**.
  Real racecheck: **0 errors, 0 warnings**.
- Existing ratio-2 compressor, index-key and source-20 contract tests pass.
  The real ratio-2 fixture also passes **177 checks**.
- Both new C++ units pass `-Wall -Wextra -Werror`. Feature-OFF and feature-ON
  builds leave default `strata_core`, `strata_kernels` and `strata_engine`
  archives byte-identical to step 6w. The main checkout remains clean.

Measured owned device allocation is **5257216 bytes (5.013671875 MiB)**:
5242880 bytes for the BF16 projection, 2048 for the F32 norm weights, and 12288
for one BF16 input row plus one F32 projection row. Changing context/prefill
limits to 1048576 does not expand this allocation. Caller buffers and CUDA
runtime resources are excluded; this is not a full-model memory measurement.

Reproduce with the existing fixed source-20 fixture:

```sh
cmake --build build-deepseek-cuda --target deepseek41_source20_compressor_cuda_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_source20_compressor_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_compressor_cuda_test \
  --real build-deepseek-cuda/source20-fixture.gguf
```

Commands, numerical results, sanitizer logs and source hashes are recorded in
`build-deepseek-cuda/step6x-validation.json` and adjacent per-step evidence.
No full-prefix GPU rerun, multi-GPU test, injected hardware failure, throughput
benchmark or full-model quality/capacity test was performed in this gate.
Existing prefix and CUDA kernel sources are unchanged.

This gate was confirmed. Step **6y** below adds source-20 ratio-1 compressed-KV
post-processing and a bounded cache. Index keys, candidates and decoder
integration remain later gates.

## Step 6y: source-20 ratio-1 compressed KV

Implemented, validated and user-confirmed. Step 6x was confirmed.
`postprocess_source20_kv` and `CudaSource20KvCache` now consume the standalone
native source-20 compressor output. The complete native prefix remains layers
0–19. Source-20 index keys, candidates, attention and decoder assembly are not
connected by this gate.

### Reused computation and bounded state

The existing compressed-KV kernel already takes a position-step argument.
The new entry point passes step 1, while ratio-2 KV/index keys retain step 2
and index queries retain step 0. Only the host launch bounds were extended.
The frequency calculation, RoPE kernel and FP4 conversion kernel are unchanged;
their source block is byte-identical to the step-6x snapshot.

The transform applies adjacent-pair YaRN RoPE to the last 64 channels, then
group-16 FP4/E2M1 quantization with E4M3 scales, and restores BF16 values.
Positions are `first_position + i`, including odd positions and position
1048575. The F32 pre-RoPE latent remains untouched for the future index-key path.

The separate source-20 cache accepts only `Source20Batch`, keeping ratio-1
calls distinct from ratio-2 emission descriptors. It reuses `DeviceArena` and
the existing transform, with an explicit startup capacity and no eviction.
Append requires the expected next position and rejects insufficient capacity
before writing. Reset clears the complete allocation without reallocating.
One explicit stream is retained across calls. A later owner must preflight
cache capacity before advancing its compressor; this cache cannot roll back
an already advanced producer. Shared publication/epoch tracking remains later work.

Storage is **1024 bytes per cached token**: 512 restored BF16 values, not packed
FP4 codes. Allocations for capacities 1, 5, 17 and 33 were checked. The reference
sequence uses a 17-token cache of **17408 bytes**; one native compressor plus
that cache owns **5274624 bytes**, excluding caller buffers and CUDA resources.
The maximum RoPE position was checked with small raw-transform batches. No
1048576-token cache or full-context prefill was allocated or benchmarked.

### Reference and validation

The new offline fixture uses the same two fixed source-20 BF16 weight tensors.
It extracts unmodified pinned `Compressor`, `RMSNorm`, RoPE and
`Attention._compress_kv` definitions. Index selection is replaced by a capture
of the pre-RoPE latent; the established independent NumPy FP4/E4M3 grid oracle
represents quantization. This is a component reference, not official GPU
attention/indexer parity.

The fixture covers **19 raw post-processing cases**, including odd positions,
7/8/9 boundaries, the YaRN context boundary, and the final model position.
Nine fixed-weight sequence plans make **58 calls**, covering odd/even prefill,
single-token decode, zero/tiny/large inputs and reset. Tests compare both
the native compressor/cache chain and a cache fed the captured reference latent.

Completed checks on the RTX 2080 Ti (SM75, reported 22528 MiB):

- **75 synthetic checks, 0 failures**. FP4 midpoint/scale rounding, signed zero,
  position bounds, invalid buffers, fixed allocation, full-capacity rejection,
  history preservation and reset are covered. **45568 BF16 value comparisons**
  match bit-for-bit.
- **515 reference checks, 0 failures**. All raw cases and both sequence paths
  match the fixture bit-for-bit: **667648 cumulative BF16 value comparisons**
  with no differing values. This count includes repeated history comparisons.
  Native cache contents match across the six equivalent prefill partitions.
- Real and synthetic Compute Sanitizer memcheck: **0 errors, 0 leaked bytes**.
  Real racecheck: **0 errors, 0 warnings**.
- Existing ratio-2 KV, index-key and index-query tests pass **40, 46 and 58
  checks** respectively. The real ratio-2 KV fixture passes **138 checks**.
- Existing Python tests: **13 passed**. Fixture identity, source-hash and
  overwrite guards: **7 passed**.
- Both new C++ units pass `-Wall -Wextra -Werror`. The normal CUDA build passes.
  Additional NVCC host-warning checks reproduce two generated member-pointer
  parentheses warnings in unchanged `gguf_reader.hpp`, also reproduced with
  the step-6x source. Keeping those warnings visible with only
  `-Wno-error=parentheses` allows the additional check to pass; the check asserts
  that no other warnings occur. No default reader code was changed.
- Feature-OFF and feature-ON builds leave default `strata_core`,
  `strata_kernels` and `strata_engine` archives byte-identical to step 6x.
  The main checkout remains clean.

Reproduce using the bounded local source cache:

```sh
python3 tools/deepseek41/prepare_source20_kv_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --output /tmp/source20-kv-fixture.gguf
cmake --build build-deepseek-cuda --target deepseek41_source20_kv_cuda_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_source20_kv_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_kv_cuda_test \
  --real /tmp/source20-kv-fixture.gguf build-deepseek-cuda/source20-fixture.gguf
```

The preparer is offline and refuses to overwrite output. Evidence, exact
commands, source hashes, the initial strict-warning failure and its diagnosis
are recorded in `build-deepseek-cuda/step6y-validation.json` and adjacent logs.

Full-prefix execution was not rerun in this gate: the shared device math is
unchanged and its existing ratio-2/indexer component paths were regressed.
Multi-GPU behavior, injected hardware failures, full-model generation, quality,
throughput and whole-model RAM/VRAM fit remain untested here.

This gate was confirmed. Step **6z** below adds source-20 index-key generation
and its bounded ratio-1 cache using the preserved pre-RoPE latent. Candidate
selection and decoder integration remain later gates.

## Step 6z: source-20 ratio-1 index keys

Implemented, validated and user-confirmed. Step 6y was confirmed.
The standalone source-20 compressor can now feed both KV and index-key caches.
The connected decoder prefix still ends at original layer 19.

### Shared implementation

`Source20IndexKeyWeights` reads the original layer-20 `wk` and `k_norm` BF16
weights, with fixed tensor names, shapes and source ownership checks.
`CudaSource20IndexKeyCache` accepts only `Source20Batch`; it cannot accidentally
consume a ratio-2 emission descriptor. The existing ratio-2 public API remains.
Both cache types use the same internal allocation, append and reset code, with
position steps 1 and 2 respectively. This is one key pipeline used by different
source structures, not another layer-specific operator implementation.

The pipeline reuses Strata BF16 GEMV and RMSNorm, with the existing BF16 rounding
boundaries. The existing RoPE/FP4 kernel applies adjacent RoPE to the last 64
channels, followed by group-32 E8M0-scaled FP4/E2M1 roundtrip to BF16. Only a
step-1 launch wrapper was added. The frequency/device-math block and the
ratio-2 KV cache implementation are byte-identical to step 6y. No CUDA kernel
was added. The pinned checkpoint index has these key tensors only at source 20,
not at later index sources 24, 28, 32 or 36.

The key cache requires contiguous append, checks the complete batch before
writing, never evicts and clears all slots on reset. It retains one explicit
stream and its allocation. Both consumers leave the pre-RoPE latent untouched.
The later owning attention component must preflight both cache capacities before
advancing its compressor; shared publication and joint failure handling are
still pending. `last_normalized()` exposes the existing final-row scratch for
component diagnostics without another device allocation.

### Fixed reference and measured results

The offline fixture uses the pinned official CPU Compressor/RMSNorm/RoPE and
Indexer key prefix. It stops before query projection and reuses the independent
NumPy E8M0/FP4 reference. All four original source tensors are checked against
their cached provenance hashes. This is not official GPU parity or a full
attention reference.

The fixture is **2930144 bytes**, SHA-256
`8e8493e9af4946f943b9b8a0cef93a60a894555a72733789fcebd9a8a1f2aec4`.
It covers 19 raw cases, odd/even positions, 7/8/9, the YaRN boundary and position
1048575. Nine plans make 58 calls using the same inputs as gates 6w and 6y:
whole/partitioned prefill, single-token decode, reset, zero, tiny and large input.
The tests alternate whether KV or index keys consume the latent first.

On the RTX 2080 Ti, CUDA 12.8 / SM75:

- Portable weight-reader validation: **61 checks**, Release and ASan/UBSan pass.
- Synthetic GPU validation: **78 checks**; **11392** compared BF16 values match.
- Original-weight GPU validation: **756 checks**; **488704** cumulative BF16
  comparisons match exactly across raw transforms, native/official-latent key
  cache histories and paired KV histories. Native sequence partitions match
  each other byte for byte after reset.
- The diagnostic last normalized rows compare **14848** values, with 10 numeric
  differences. Maximum relative L2 is **9.45174732418e-10**, maximum absolute
  error **7.45058059692e-9**. The predeclared normalized tolerance is relative
  L2 <= 0.003 and max absolute <= 1e-5 + 0.02 times the reference peak. Final
  quantized cache comparisons require exact BF16 bytes and were not relaxed.
- Synthetic and original-weight memcheck: **0 errors, 0 bytes leaked**.
  Original-weight racecheck: **0 errors, 0 warnings**.
- Existing ratio-2 key/KV/query and source-20 KV synthetic regressions pass.
  Original-weight ratio-2 key **185 checks**, ratio-2 KV **138 checks** and
  source-20 KV **515 checks** also pass.
- **13 Python tests** and **10 fixture guard/reproducibility checks** pass.
  The final preparer regenerates the same fixture bytes and refuses wrong
  source/config/kernel/revision/tensor hashes or an existing output.
- Four changed C++ units pass `-Wall -Wextra -Werror`. CUDA strict validation
  retains the same two NVCC-generated member-pointer parentheses warnings in
  the unchanged GGUF reader, documented in step 6y. The validation-only
  `-Wno-error=parentheses` exception asserts exactly those two warnings and no
  others. Normal CUDA builds pass; no default reader change was made.
- Feature-OFF/ON default core, kernels and engine archives remain byte-identical
  to step 6y. The main checkout remains clean.

Owned key storage is **133120 + 256 times capacity** bytes: fixed original
weights and one-row scratch plus 128 restored BF16 values per token. At the
17-token test capacity it is **137472 bytes**. One compressor plus both caches
owns **5412096 bytes (5.161376953125 MiB)**. These are component allocations,
excluding caller buffers and CUDA resources; keys are not stored as packed FP4.
Maximum raw RoPE positions were tested with small batches, not a full-context
cache allocation or full-model memory measurement.

Reproduce with the local pinned source cache:

```sh
python3 tools/deepseek41/prepare_source20_index_key_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --output /tmp/source20-index-key-fixture.gguf
cmake --build build-deepseek-cuda --target deepseek41_source20_index_key_cuda_test deepseek41_index_key_weights_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_index_key_weights_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_index_key_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_index_key_cuda_test \
  --real /tmp/source20-index-key-fixture.gguf \
  build-deepseek-cuda/source20-fixture.gguf build-deepseek-cuda/source20-kv-fixture.gguf
```

The source cache must contain the original source-20 compressor and index-key
weights. The preparer performs no network access and refuses to overwrite.
Commands, hashes, sanitizer output and the isolated gate diff are recorded in
`build-deepseek-cuda/step6z-validation.json` and adjacent evidence.

No full-prefix rerun, multi-GPU test, hardware-failure injection, model-quality
measurement, throughput benchmark or whole-model RAM/VRAM fit test was done in
this gate. This gate was confirmed. Step **6aa** below connects source-20
queries, scoring and candidate selection.

## Step 6aa: source-20 queries, scores and candidate blocks

Implemented and validated; awaiting user confirmation. Step 6z was confirmed.
The standalone source-20 path now projects index queries, computes causal
scores, selects its own top-512 positions and selects candidate blocks for later
index sources. The connected native decoder prefix still ends at layer 19.

### Reused computation and candidate rules

`Source20IndexQueryWeights` reads the original source-20 FP8/E8M0 block-32
`wq_b` and BF16 `weights_proj` tensors. Shape, source identity, ratio and
ownership checks are separate from the existing ratio-2 reader.
`CudaSource20IndexQuery` uses the same internal query projection, RoPE/FP4,
weighted score accumulation and CUB sort implementation as `CudaIndexQuery`.
Visibility is one key per token, passed through the confirmed `Source20Batch`.
The existing scale, score-accumulation and index-shift kernels are unchanged.

The one new kernel computes candidate-block maxima over eight positions and
pins the block containing the newest visible key to positive infinity. It
marks unreachable blocks for padding. The existing CUB stable score sort,
position sort and index-shift kernel then produce at most 2048 block IDs,
ascending, followed by -1 padding. Equal scores prefer lower original IDs;
Torch top-k may choose a different equally scored subset.

`CudaSource20Candidates` also exposes this component for direct score-row
validation. It masks future positions before taking block maxima. A selected
partial block still represents all of its positions, including future positions
inside that block. Downstream consumers must also apply the causal mask.
Source 20's own top-512 uses its causal scores without candidate filtering;
its candidate block IDs have no attention offset.

Both selectors preallocate scratch and CUB workspace. Calls do not allocate
device memory and preserve their input buffers. They enforce capacity, query
row, pointer, offset and single-stream/device rules before changing output.
An execution failure requires reconstruction. Results are one-query views,
valid until the next successful call; a later attention owner must copy prefill
rows and publish them with the appropriate position/epoch. No shared state is
published in this gate, and no later index-source consumer was connected.

### Head-projection cancellation found and resolved

The first original-weight run reported one differing head weight and 25
resulting score differences in one long-history case. Query/key BF16 outputs
and both selected sets already agreed. The initial failed run remains in
`step6aa-before-head-fix-real.log`.

A standalone diagnostic tested all 118 query inputs against FP64 dot products,
Torch and both existing Strata BF16 GEMV entry points. With 32 outputs, the
legacy entry uses a serial FP32 sum. For selection case 0, head 18:

| Unscaled dot | Value |
| --- | --- |
| FP64 control | -0.024107805726089282 |
| Existing serial FP32 GEMV | -0.024109378457069397 |
| Existing 32-lane GEMV | -0.024107933044433594 |
| BF16 rounding midpoint | -0.02410888671875 |

Cancellation pushed the serial sum across that midpoint. After the declared
BF16 rounding and scaling, the existing 32-lane entry agrees with both FP64
and Torch for all **3776 head values**. Source-20 now calls
`bf16_gemv_split(..., 32)` for this head projection. Ratio-2 preserves its
previous accumulation order. The shared Strata kernel sources are unchanged;
this is an existing-kernel selection inside the opt-in source-20 path.

A synthetic regression also checks the exact sum
`2^24 + 5118 ones - 2^24 = 5118`, with cancellation terms placed on the same
warp lane. BF16 rounding followed by division by 64 must give 80; the serial
sum gives 79.5. No fixture values or comparison tolerances were changed to
make this pass. The diagnosis, commands and before/after results are preserved
in `build-deepseek-cuda/step6aa-head-diagnosis.json` and adjacent logs.

### Fixed reference and validation

The offline fixture runs the pinned CPU Compressor, Indexer and
`select_candidate_blocks` graph with seven original weight/scale tensors.
It reuses the established independent FP8 and FP4 rounding references. This is
a component oracle, not official TileLang GPU parity or a full decoder run.

Nine plans cover the same 58 compressor/key-cache calls as gates 6w–6z.
Together with decode cases and selection cases, there are **118 query rows**.
The long selection cases repeat 640 original-weight keys into a controlled
16393-key history to exercise actual pruning above the 2048-block threshold.
They are not a 16K-token original compressor run. Separately, all 20 pinned
candidate-rule fixtures from gate 6w run on the GPU, including the maximum
1048576-position score row, newest-block pinning, holes, ties and partial blocks.
Six raw query RoPE cases cover odd/even and maximum token positions.

The new fixture is **18825792 bytes**, SHA-256
`961541ba7c7f02926018acddd067f4815076838f23cbb6a6eecdc56e16077a1c`.
It is byte-identical before and after the head-projection fix.

On the RTX 2080 Ti, CUDA 12.8 / SM75:

- Portable reader: **65 checks**, Release and ASan/UBSan pass.
- Synthetic GPU: **334 checks** pass, including analytical cancellation,
  causal top-512, candidate pruning and argument failures preserving results.
- Original-weight GPU: **1376 checks** pass. All **670080** compared BF16
  query/key values and **40613** compared head/score values match exactly.
  Top-512 and candidate sets match the independent canonical tie rule and
  Torch's selected score membership; tied IDs need not be identical.
- Synthetic and original-weight memcheck: **0 errors, 0 bytes leaked**.
  Original-weight racecheck: **0 errors, 0 warnings**.
- Existing ratio-2 query and source-20 key synthetic tests pass. Original-weight
  ratio-2 query **126 checks** and source-20 key **756 checks** also pass.
- **13 Python tests** and **13 fixture provenance/reproducibility guards** pass.
  The final generator reproduces the fixture hash, rejects wrong pinned inputs
  and tensor hashes, and refuses an existing output.
- Three C++ units pass `-Wall -Wextra -Werror`. CUDA strict validation uses the
  previously documented validation-only `-Wno-error=parentheses` exception and
  confirms exactly the same two NVCC-generated GGUF-reader warnings, no others.
  Normal CUDA builds pass; no default reader change was made.
- Feature-OFF/ON default core, kernels and engine archives remain byte-identical
  to step 6z. The main checkout remains clean.

At the tested 16393-key capacity, query plus candidate processing owns
**11294208 bytes**, including **138240 bytes** of requested CUB workspace.
The independent candidate selector at maximum context owns **3170816 bytes**,
including **1065471 bytes** of requested workspace. Arena alignment is included
in owned totals. These figures exclude caller score/key/input buffers and CUDA
resources. No full-context query/key cache or full model was allocated, and no
throughput or whole-model 64 GB RAM/20 GB GPU fit claim follows from these tests.

Reproduce with the local pinned compressor, key and query source tensors:

```sh
python3 tools/deepseek41/prepare_source20_index_query_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --output /tmp/source20-index-query-fixture.gguf
cmake --build build-deepseek-cuda --target deepseek41_source20_index_query_cuda_test deepseek41_index_query_weights_test -j4
build-deepseek-cuda/src/deepseek41/deepseek41_index_query_weights_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_index_query_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_index_query_cuda_test \
  --real /tmp/source20-index-query-fixture.gguf build-deepseek-cuda/source20-fixture.gguf
```

Commands, source hashes, complete sanitizer results and the gate diff are in
`build-deepseek-cuda/step6aa-validation.json` and adjacent evidence. Full-prefix
execution, multi-GPU behavior, injected hardware failures, model quality and
throughput were not tested here. This gate was confirmed. Step **6ab** below
assembles source-20 attention and its shared KV/index/candidate publication.

## Step 6ab: complete source-20 attention and publication

`CudaSource20Attention` combines the confirmed ratio-1 compressor, compressed
KV cache, index keys and query/candidate selection with the existing attention
projection and sparse-core arithmetic. No CUDA math kernel was added or changed.
The existing ratio-2 owners and raw compressed-core API retain their behavior.
The weight reader now accepts the explicit original layer-20 ratio-1 profile.

Context and prefill capacities are explicit in tokens. All argument checks run
before any component advances or an existing publication expires. Prefill uses
two passes: preserve every original compressor latent for both caches, then
project and attend one query at a time against the whole prefill layout. Decode
uses one token per call. The source's own top-512 is unfiltered by candidates.
Execution failures invalidate the complete owner; reset is not error recovery.

A distinct `Source20AttentionBatch` borrows the source's KV/key caches and retains
per-query index/candidate rows in startup storage. It checks call position,
token count, device and stream. Handles expire on advance, reset, failure or
source destruction; rejected calls preserve the last completed publication.
Reset changes the sequence identity and clears all window/KV/key history.
Raw diagnostic views do not extend lifetime. Candidate rows contain sorted
block IDs followed by -1; a selected partial block can include future positions,
so future consumers must also apply their causal mask. Downstream consumers and
cross-owner consumer binding are not implemented in this gate.

The existing pinned attention fixture builder now accepts source 20. It verifies
20 original tensors and executes the original attention, compressor, indexer
and candidate-selection graph with independent CPU arithmetic. Acceptance uses the existing step-6s2 uniform FP64-linear control, selected
before the first native run. The legacy block-32 FP32 reference is also run
as a separate diagnostic, with its failures retained below.
Five prefill/decode plans make 42 calls and include 129-token prefill, window
wrap, odd/even partitions and decode through position 130. The integrated test
has at most 131 keys; top-512 pruning and the 2048-block boundary retain their
separate step-6aa coverage. This does not establish official GPU bit parity.

For the FP64-linear control on the RTX 2080 Ti:

| Comparison | Calls | Maximum relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Attention output | 42 | 0.00260730571 | 0.0078125 |
| Window ring | 42 | 0 | 0 |
| Compressed KV | 42 | 0 | 0 |
| Index keys | 42 | 0 | 0 |

The pre-existing gates remain unchanged: output/ring relative L2 <=0.005 and
maximum absolute error <=`1e-5 + 0.03 * reference_peak`; compressed KV/index keys
<=0.003 and `1e-5 + 0.02 * reference_peak`. All outputs must be finite and BF16
valued. Published indices and candidate blocks match each reference exactly.

The source-20 block-32 diagnostic reports **429 checks, 3 output failures**:
decode position 8 in three prefill plans, relative L2 0.00581589812 against the
unchanged 0.005 limit. Its window/KV/key checks and all publication rows pass.
Two `wq_b` BF16 outputs differ between the CPU reference modes at this token;
`wq_a`, normalized/quantized query input and window KV agree. A diagnostic copy
changing only `wq_b` to the existing FP64 accumulation rule reduces the two
CPU references' output relative L2 from 0.00581559905 to 0.00000116949709.
Original weights and input hashes are identical. No production, reference,
fixture or acceptance-tolerance changes were made in response to this diagnostic.

A direct call to the existing native query GEMV agrees with the FP64 control
on those two elements. It also has one separate BF16 difference out of 32768
query elements versus FP64 (three versus block-32); the exploratory exact-query
check therefore exits 1 and is retained, not presented as passing. The complete
native attention output at this token equals all 5120 FP64-reference values.
This confirms an accumulation/quantization boundary in the diagnostic control,
not intermediate bit parity. Details and the causal intervention are in
`step6ab-reference-stage-diagnosis.json` and `step6ab-query-diagnosis.log`.


Owned device storage is **271885056 bytes** at context 9/prefill 8 and
**272119296 bytes** at context 131/prefill 129. These include attention weights,
all component arenas, caches and publication buffers; they exclude cuBLAS,
CUDA runtime and caller buffers. Allocation stays fixed across tested calls.
They are component measurements, not a full-model resident-memory estimate.

The initial source-14 regression accidentally used the legacy block-32 fixture.
It reported the same three output discrepancies recorded in step 6s2; the entire
native log is byte-identical to `step6s2-final-real14-block32.log`. The accepted
source-14 FP64 fixture is verified against its previously recorded hash and
passes all 375 checks. Both runs remain in the evidence. No production logic,
reference arithmetic or comparison threshold changed to handle this control.


Validation includes 67 synthetic checks, 429 passing FP64-reference checks, 204
weight-reader checks in Release and ASan/UBSan, CUDA memcheck on synthetic and
real cases, real racecheck, and existing attention/source/shared-consumer
regressions. The prior source-2 reference fixture is reproduced byte-for-byte.
All 13 existing Python tests pass. Strict compilation passes with the two
previously documented NVCC-generated `gguf_reader.hpp` parentheses warnings
remaining the only validation exception. Default core/kernel/engine archives
remain byte-identical with the feature off and on; the main checkout is clean.

```sh
python3 tools/deepseek41/prepare_layer2_attention_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 20 --linear-oracle fp64 \
  --output build-deepseek-cuda/source20-attention-fp64-fixture.gguf
build-deepseek-cuda/src/deepseek41/deepseek41_source20_attention_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_attention_cuda_test \
  --real build-deepseek-cuda/source20-attention-fp64-fixture.gguf
```

The fixture builder refuses an existing output. The block-32 reference uses
`--linear-oracle block32` and a separate output path. Evidence is recorded in
`build-deepseek-cuda/step6ab-validation.json`, its patch and adjacent logs.
Step **6ab** was confirmed. The native decoder prefix remains 0–19;
step 6ac below assembles the standalone source-20 decoder. No embedding, output
head, text generation, memory-target or throughput result is claimed here.

## Step 6ac: standalone source-20 decoder

`CudaSource20Decoder` joins the confirmed source-20 attention to existing mHC,
text MoE, native expert caching and residual computation. It requires the original
layer-20 identity and ratio-1 KV/index ownership. All math uses the confirmed
Strata/native components; this gate adds decoder orchestration and validation.

The attention owner now exposes private staged operations to the decoder.
Prefill first runs the existing mHC collapse/normalization and fills both
ratio-1 caches. A second pass completes attention, its residual join, FFN mHC,
top-6/shared-expert computation and the final residual for each token. Scratch
contains one token's intermediates. Context and publication limits remain
explicit startup allocations; caller input/output buffers are separate.
Same-token delayed coefficients pass to the next layer, not the next token.

Only successful completion of the whole decoder call commits its position and
shared KV/key/index/candidate publication. Preflight checks protect all four
residual/coefficient buffers, exposed caches and publication storage. Rejected
calls preserve the previous publication. An expert-read failure after cache
advance invalidates the publication and poisons the layer; reset cannot recover
partial execution. Reset clears sequence caches and retains immutable expert
weights and cumulative cache statistics.

### Validation and limits

The synthetic suite passes **70 checks**, including 129-token prefill followed
by decode, window wrap, explicit capacity limits, aliased buffers, original
expert identity, same-token coefficients, reset/replay and failure injection.
Its long-path reference combines standalone attention with independent CPU
residual joins and the existing shared expert. Routed experts are zero in this
synthetic control. A callback during actual expert reads verifies that no
publication becomes visible before the enclosing decoder commits.

The real suite passes **198 checks** using 198 hash-verified tensors from the
pinned 1017 revision and **27 original experts**, selected by the original gate.
It runs a three-token prefill and two single-token decodes, with six and two
expert slots, then repeats after reset. Original top-6 IDs/weights, KV/index
histories, publication rows and cache accounting are checked. Decoder outputs
and delayed coefficients are byte-identical across slot counts and reset/replay.

The reference retains the accepted step-6s2 FP64 attention linears and original
CPU Block through FFN inputs, coefficients and residual joins. Previously
validated native Q8_1 MoE supplies expert arithmetic. This is an independent
attention/Block plus native-MoE composition baseline; it is not a complete
official quantized-decoder or GPU-bit-parity result. Output/cache tolerances,
strict same-input mHC coefficient checks and the separately measured propagation
bound are unchanged from prior decoder gates. Step 6ab's block-32 diagnostic
limitations remain documented there; reference arithmetic was not changed here.

The first real run recorded **166 checks with four failures**, all for route
weights at decode position 3, repeated across cache sizes and reset. The initial
check compared weights from different FFN inputs with a fixed `1e-4` threshold.
All original expert IDs, output and cache checks passed. Independent attention/
mHC composition reproduced the decoder routes exactly. At this position the
FFN inputs differ from the CPU graph at 1556 of 5120 elements, with relative L2
0.00268476 and maximum absolute difference 0.0078125. The attention residual and
FFN-input boundaries both satisfy the existing output tolerance.

Running the pinned original `Gate.forward` on the captured native FFN inputs
reduces the maximum same-input weight difference to approximately `3.02e-8` over
all cases. The original-input weight difference reaches `0.000136122`, so the
initial failure is retained rather than reported as a passing `1e-4` check.
Runtime arithmetic and fixture contents were not changed. The test now checks
original IDs on both inputs, uses an independent FP64 CPU gate with the existing
strict `2e-6` same-input allowance, and separately checks input-error propagation.
For `s(z)=sqrt(softplus(z))`, `|s'(z)| <= 1/2`; this bounds score changes from the
measured input difference through the router weights. Normalization gives a
per-route bound, with `2e-6` numerical allowance for each endpoint. The maximum
observed fraction of that bound is approximately 0.476. This is a revised,
explicit acceptance criterion for differing inputs, not official whole-decoder
numerical parity. Original failed logs and the pinned-Gate diagnosis are saved
in `step6ac-before-route-fix-real.log` and `step6ac-gate-diagnosis.json`.

CUDA memcheck passes synthetic and real suites without errors or leaks; real
racecheck reports zero errors or warnings. The refactored standalone source-20
attention passes 67 synthetic and 429 real checks with **complete logs identical
to step 6ab**. Existing source-2/source-8 decoders and mHC regressions pass.
Source-2 offline discovery retains its original step-6h route IDs. All 13 Python
tests pass. Strict checks pass for both new C++ files and the changed CUDA unit,
with only the previously documented pair of NVCC-generated parentheses warnings.
Default core/kernel/engine archives remain byte-identical with the feature off
and on; the main checkout is clean.

### Bounded resources and reproduction

Synthetic context 130/prefill 129 with six expert slots owns **470743072 device
bytes**. The real suite uses context 5/prefill 3; cumulative traffic below covers
two five-token runs per instance, including reset. Owned bytes exclude cuBLAS,
CUDA runtime, host weights/staging and caller buffers.

| Expert slots | Owned device bytes | Hits | Misses | Evictions | Expert H2D bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 470505248 | 2 | 58 | 52 | 1154580480 |
| 2 | 390879008 | 0 | 60 | 58 | 1194393600 |

These are standalone-component allocations and measured transfer counts, not
whole-model memory or throughput results. Only the 27 selected experts were
fetched and populated in the sparse fixture; the other original-ID entries are
holes and the test rejects access to them.

```sh
python3 tools/deepseek41/prepare_layer2_decoder_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 20 --discover
# After caching the reported original expert tensors, use fresh output paths:
python3 tools/deepseek41/prepare_layer2_decoder_fixture.py \
  --model-dir build-deepseek-cuda/reference-cache/source20 \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --reference-kernel build-deepseek-cuda/reference-cache/inference_kernel.py \
  --layer 20 --output /tmp/source20-decoder.gguf \
  --routed-output /tmp/source20-decoder-routed.gguf
build-deepseek-cuda/src/deepseek41/deepseek41_source20_decoder_cuda_test
build-deepseek-cuda/src/deepseek41/deepseek41_source20_decoder_cuda_test \
  --real /tmp/source20-decoder.gguf /tmp/source20-decoder-routed.gguf
```

Evidence and the exact gate diff are in `build-deepseek-cuda/step6ac-validation.json`
and adjacent files. Step **6ac** was confirmed. Step 6ad below connects source
20 and its shared consumers 21–23 into the prefix. The limitations and original
failed cross-input route checks above remain part of the numerical record.

## Step 6ad: connect original layers 20–23 into the prefix

`CudaTextPrefix` now runs original layers 0–23 with the existing two residual
buffers and two coefficient buffers. Source 20 receives the actual layer-19
output and same-token coefficients. Consumers 21–23 run their own projections,
mHC and MoE while borrowing source-20 KV and per-query top-512 indices. They do
not run an indexer or consume candidate blocks; the next index source, 24, is a
separate gate. Layer-1 and layer-14 Engram remain independent parts of the chain.

`CudaSharedAttention` and `CudaSharedDecoder` are reused for all these consumers.
Their named `forward_source20` entry accepts the distinct source-20 publication
and delegates to the existing execution path. This preserves the earlier
`forward` API, including empty-publication argument validation. There is no
new GPU math kernel, no KV/index copy and no duplicate source cache. Publication
ownership, stream/device checks, source identity, reset sequence and invalidation
now use the same private lifetime record for both source families; the public
source-20 view retains its keys and candidate blocks for future index sources.

The ratio-2 capacity argument keeps its original meaning. Source 20 reserves
`min(1048576, 2 * capacity_groups + 1)` entries, one per token, including the
last incomplete pair permitted by earlier sources. `capacity(source_layer)`
reports entries for that source; `capacity()` still means source-2 groups.
Prefill scratch is bounded by the smaller of prefill and context capacity.
All capacity, token and alias checks run before layer 0 executes, including the
last entries of source-20 KV and key storage. Only a successful full-chain call
advances the prefix position. Failure at any decoder poisons the chain, which
must then be reconstructed. Reset on a healthy chain clears all consumer bindings
before resetting their owners and retains immutable expert/Engram row caches
and cumulative counters.

### Validation and reference boundary

The standalone source-20 sharing suite passes **87 checks**. A 129-token prefill
crosses the window boundary, followed by decode and reset/replay. With identical
projection weights and inputs, consumers 21–23 match source-20 attention exactly.
Tests verify independent window rings, unchanged source KV/keys/indices/candidates,
expired publications, wrong owners, wrong source families, stream/shape mismatch,
and reset-sequence mismatch at the same position. The long native and memcheck
runs pass with zero errors or leaks. Racecheck passes the same 87 lifecycle
checks with a nine-token prefill crossing the
candidate-block boundary. An initial repeated long-prefill racecheck was stopped
after about eleven minutes of instrumentation and is recorded as incomplete,
not passed; the complete real-prefix racecheck below covers integration. No
kernel filters or numerical thresholds were changed.

The complete 24-layer synthetic suite passes **494 checks**, including whole-chain
preflight, real residual/coefficient handoffs, original layer order, all caches,
reset, and failures at layers 0, 19, 20, 21, 22 and 23. Earlier gates retain their
per-layer failure coverage for 1–18. Layer-14 table-read failure remains tested.
The initial synthetic run completed 459 checks before running out of GPU memory
while constructing a second full prefix for fault injection. The test retained
the already-validated successful instance unnecessarily. Its lifetime now ends
before independent fault-injection instances are created, preserving all planned
failure cases and runtime behavior. The original failed log is retained as
`step6ad-before-memory-fix-prefix24-synthetic.log`. Synthetic memcheck passes
without errors or leaks.

The real suite passes **415 checks** under both memcheck and racecheck. It uses a
three-token prefill plus two single-token decodes, six and two expert slots per
layer, and replay after reset. The production two-buffer chain is compared with
explicit individual decoder calls using a separate destination for every layer.
Outputs, same-token coefficients, original top-6 IDs/weights, all window rings
and all four source KV/key histories match byte-for-byte. Consumers leave every
source history unchanged. This validates native composition, ownership and
handoffs; it is **not an independent official 24-block quantized oracle**, a
logit-quality test, or a generation/performance result. The declared component
references and numerical limits from earlier gates remain unchanged.

Real preparation uses the pinned 1017 revision. Layers 21–23 each contain 29
hash-verified dense tensors. Original source-20 weights reuse step 6ac's fixture
without copying or changing its bytes. Actual upstream outputs select
**26, 25, 26, 26 distinct experts** in layers 20, 21, 22 and 23,
respectively. Only **100 additional selected experts** were downloaded and
packed. All **614 actually routed experts** across the 24
layers are independently inverted from the packed test format and checked
against original FP4-code and E8M0-scale hashes. Missing weights are never
substituted; the test-only preparation mode pauses the exact route until the
required verified files exist. Normal validation runs without that mode.

Source-20 standalone attention (67 synthetic/429 real checks) and decoder (198
real checks) retain **complete logs identical to step 6ac**. The original
20-layer real prefix retains the complete native log from step 6v. The default
four-layer real prefix, earlier shared attention/decoder tests, 234 reader checks
in Release and ASan/UBSan, and 13 Python tests pass. Strict checks pass for all
touched compilation units and the shared fixture's consumer test, with only the
previously documented two NVCC-generated parentheses warnings. Default core,
kernels and engine archives remain byte-identical with the feature OFF and ON;
the main Strata checkout is clean.

### Measured allocations and reproduction

The real suite uses prefill capacities 3 and 7, three ratio-2 groups and seven
source-20 entries. Owned device bytes below exclude CUDA/cuBLAS runtime
allocation, host weights/staging, file cache and caller buffers. These small-context component
allocations do not establish full-model fit on 64 GB RAM and a 20 GB GPU.

| Expert slots per layer | Owned device bytes | Prefix workspace bytes | Prefill capacity |
| --- | ---: | ---: | ---: |
| 6 | 11642214944 | 492032 | 3 |
| 2 | 9731840544 | 1147392 | 7 |

```sh
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test \
  deepseek41_source20_shared_cuda_test deepseek41_attention_weights_test
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(source20_shared_cuda|text_prefix24_cuda)_test$'
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 24 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The real command requires the bounded pinned fixtures and selected-expert files
recorded in `build-deepseek-cuda/step6ad-validation.json`. Adjacent logs, coverage,
hash records and `step6ad.patch` preserve the validation and exact gate diff.
Step **6ad** was confirmed. Its scope stopped at original layer 23. Gate 6ae
below adds candidate-consuming source 24 and connects its 24–27 group.

## Step 6ae: source-24 candidate indexing and original layers 24–27

Implementation and the validation below are complete, and the user confirmed
this gate. Its scope stopped at original layer 27. It added no embedding,
output head, generation loop or performance optimization.

### Reuse and state ownership

`CudaCandidateIndexQuery` uses the existing `CudaIndexQuery::Impl` ratio-1
projection, FP4 query conversion, score accumulation and CUB selection. Two
small kernels validate the compact candidate row and mask excluded scores.
The existing projection, score-reduction and selection arithmetic is unchanged.
Original layer 24 has its own query/head weights and owns its top-512 indices;
it allocates no duplicate source-20 KV or index-key cache.

`CudaSharedAttention` and `CudaSharedDecoder` accept that optional index owner.
The same projection, sparse attention, mHC, MoE and residual loops serve both
the index owner and ordinary consumers. The prefix retains its two residual
buffers and two coefficient buffers.

| Original layer | KV and keys | Indices and candidates |
| --- | --- | --- |
| 20 | Own ratio-1 caches | Own top-512 and top-2048 candidate blocks |
| 24 | Borrow source-20 caches | Own query/top-512 inside source-20 candidates |
| 25–27 | Borrow source-20 KV | Borrow source-24 top-512 |

A source-24 publication depends on both its own completed call and the exact
source-20 publication it borrowed. Advancing, resetting, failing or destroying
either owner invalidates the dependent view. Consumers protect both owners'
storage from output aliasing. Only a completed decoder, including its MoE and
residual join, publishes indices. Invalid arguments leave completed results
unchanged; execution failures invalidate state.

### Independent reference and current checks

The new CPU fixture executes the pinned official candidate-consuming
`Indexer.forward`, using original layer-24 query/head weights and the accepted
source-20 keys/candidate masks. Linear dots use the previously established FP64
CPU oracle and original BF16 boundaries; FP4 rounding is independent of native
CUDA code. All 27 query rows pass exact query-bit, head-weight and score
comparisons plus canonical top-512 and Torch score-membership checks: **148
checks**. Two controlled histories reach 16,385 and 16,393 positions, crossing
the candidate-block limit. They repeat accepted original-weight source-20 keys;
they are not full-model long-context runs or official GPU parity measurements.

The analytical index suite passes **87 checks**, including deliberate exclusion
of higher-scoring positions, negative/zero weights, causal padding and malformed
candidate rows. Shared-state tests pass **83 checks**. Their 513-position control
uses different query weights with identical source-20 KV/keys, proves that the
selected indices and attention output change, and checks that consumer 25 uses
source-24 indices. All three consumers 25–27 are covered by the 77-check
lifecycle subset with a 9-token prefill.
This shared control exercises top-512 selection, while the separate index tests
exercise the 2048-block candidate cap.

The connected synthetic prefix passes **589 checks**. Attention and query
weight readers pass **282 and 96 checks** in Release and ASan/UBSan; all 13 Python
tests pass. Twelve affected translation units pass strict compilation. Each of
the two CUDA units has only the same two previously documented NVCC-generated
header warnings. Re-preparing the previous layer-21 dense fixture produces
byte-identical contents.

Original layer-24 dense preparation preserves 32 original tensors; layers
25–27 preserve 29 each. All are tied to the fixed revision and tensor hashes.
The actual connected prefix requested 110 additional original experts from
layers 24–27. All 724 distinct layer/expert pairs used across the 28 layers
were independently inverted from their packed GGUF layout and verified against
the original FP4 code bytes and E8M0 scale hashes. Original layers 0–23 choose
the same expert-ID sets as gate 6ad.

The original-weight prefix passes **499 checks** under both CUDA memcheck
and racecheck. Residuals, delayed coefficients, routes, window/KV histories,
reset and cache-budget variants match the explicit native composition. Memcheck
also passes the 589-check synthetic prefix, both candidate-index suites (87 and
148 checks), and the complete 83-check shared suite. Candidate real-query
racecheck passes all 148 checks; shared lifecycle racecheck passes 77 checks
with a 9-token prefill. All completed memchecks report zero errors and zero
leaked bytes; all completed racechecks report zero hazards. The 513-position
shared control ran natively and under memcheck; its long racecheck was not run.

The previous original 0–23 real prefix passes 415 checks with a byte-identical
native log to gate 6ad after removing sanitizer annotations. The default
four-layer real prefix passes 110 checks. The old ratio-2 and source-20 query,
shared attention/decoder and source-20 attention/decoder regressions pass.
Source-20 attention and decoder logs remain byte-identical to gate 6ad. Default
core, kernels and engine archives match gate 6ad with the feature OFF and ON;
the main Strata checkout remains clean.

The full-prefix control uses explicit separate buffers with the same confirmed
native components. It validates handoffs and ownership; it is not an independent
official 28-block reference, logit-quality test or whole-model capacity test.
No result here establishes generation speed or fit under 64 GB RAM/20 GB GPU.

### Measured allocations and reproduction

The real suite uses three ratio-2 groups, seven source-20 entries, and prefill
capacities 3 and 7. These are owned device bytes; they exclude CUDA/cuBLAS runtime,
host weights/staging, file cache and caller buffers. The synthetic prefix owns
13,470,562,720 device bytes, including 983,552 workspace bytes, with two ratio-2
groups, five source-20 entries and a prefill capacity of six.

| Expert slots per layer | Owned device bytes | Prefix workspace bytes | Prefill capacity |
| --- | ---: | ---: | ---: |
| 6 | 13470077600 | 492032 | 3 |
| 2 | 11241198240 | 1147392 | 7 |

```sh
cmake --build build-deepseek-cuda --target deepseek41_text_prefix_cuda_test \
  deepseek41_candidate_index_query_cuda_test deepseek41_source24_shared_cuda_test
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(candidate_index_query_cuda|source24_shared_cuda|text_prefix28_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_candidate_index_query_cuda_test \
  --real build-deepseek-cuda/candidate-index-query-fixture.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 28 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The real commands need the pinned fixtures and selected-expert files recorded
in `build-deepseek-cuda/step6ae-validation.json`. Adjacent logs, coverage and hash
records retain the evidence; `step6ae.patch` isolates this gate from earlier
uncommitted work. That gate rejected original layer 28. After user confirmation,
gate **6af** below reuses the same path for original layers 28–31.

## Step 6af: reuse the candidate index path for original layers 28–31

Implementation and the validation below are complete, and the user confirmed
this gate. Its scope stopped at original layer 31. Gate 6ag below adds the next
repeated group after that confirmation.

The existing candidate index query, shared attention and shared decoder accept
original source 28 with its own weights. No new CUDA arithmetic kernels or
decoder loops were added. Source 28 borrows source-20 KV, keys and candidate
blocks directly; it does not consume source-24 indices. Consumers 29–31 use
source-28 indices. The prefix retains two residual and two coefficient buffers.

Both groups can publish independently for the same source-20 batch. Resetting
24 leaves 28 current, and resetting 28 leaves 24 current. Resetting the common
source 20 invalidates both. Cross-group publications are rejected before a
consumer advances. The new shared suite passes 91 checks, including eight
checks of independent group ownership and a 513-position distinct-query
control for consumer 29. The 9-token lifecycle cases cover consumers 29–31.

The independently generated source-28 CPU fixture uses the pinned official
candidate-consuming Indexer.forward, FP64 linear dots and the existing BF16/FP4
boundaries. All 27 query rows pass 148 native checks, including two controlled
pruning histories at 16,385 and 16,393 positions. The analytical index suite
passes 87 checks; the connected synthetic prefix passes 697 checks.

A first offline fixture attempt exposed a remaining hardcoded layer-24 tensor
name. The name now follows the selected original layer. The failure log and
cause are retained; no reference arithmetic or comparison tolerance changed.
Re-preparing the previous layer-24 dense fixture produces byte-identical bytes.

Original layer-28 dense preparation preserves 32 original tensors; layers
29–31 preserve 29 each. All use the fixed revision and original tensor hashes.
The actual connected prefix requested 106 additional original experts from
layers 28–31. All 830 distinct layer/expert pairs used across 32 layers were
inverted from their native packed GGUF layout and verified against original
FP4 code bytes and E8M0 scale hashes. Original layers 0–27 keep the same routed
expert-ID sets as gate 6ae.

The original-weight prefix passes **595 checks** under both CUDA memcheck
and racecheck. Residuals, delayed coefficients, routes, source publications,
window/KV histories, reset and cache-budget variants match the explicit native
composition. Memcheck also passes the 697-check synthetic prefix, 87-check
analytical index, 148-check original-weight index and full 91-check shared
suite. The real index passes racecheck; the shared lifecycle and independent
group cases pass 85 racecheck checks with a 9-token lifecycle prefill. All these
memchecks report zero errors and zero leaked bytes; all these racechecks report
zero hazards. The 513-position shared control ran natively and under memcheck;
its long racecheck was not run.

Attention and query weight readers pass 330 and 127 checks in both Release and
ASan/UBSan. All 13 Python tests and 12 strict translation-unit builds pass.
Each CUDA unit reports only the same two established NVCC-generated header
warnings. Earlier component regressions pass, including the original source-24
candidate query and shared path. The real 28-element prefix passes 499 checks
with a byte-identical native log to gate 6ae after stripping sanitizer lines;
the default four-layer prefix passes 110 checks. Source-20 attention/decoder
logs remain byte-identical. Default core, kernels and engine archives match
gate 6ae with the feature OFF and ON; the main Strata checkout remains clean.

### Allocations, reproduction and limits

On the RTX 2080 Ti reporting 22528 MiB, the synthetic prefix owns 15,298,425,376
device bytes, including 983,552 workspace bytes, with ratio-2 capacity 2,
source-20 capacity 5 and prefill capacity 6. The original-weight variants below
use ratio-2 capacity 3 and source-20 capacity 7. Owned bytes exclude CUDA/cuBLAS
runtime, host weights/staging, file cache and caller buffers.

| Expert slots per layer | Owned device bytes | Prefix workspace bytes | Prefill capacity |
| --- | ---: | ---: | ---: |
| 6 | 15297940256 | 492032 | 3 |
| 2 | 12750555936 | 1147392 | 7 |

```sh
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(candidate_index_query28_cuda|source28_shared_cuda|text_prefix32_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_candidate_index_query_cuda_test \
  --layer 28 --real build-deepseek-cuda/candidate28-index-query-fixture.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 32 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The real commands require the pinned fixtures and selected experts recorded in
`build-deepseek-cuda/step6af-validation.json`. Adjacent logs retain the failure
diagnosis, reference and hash evidence; `step6af.patch` isolates this gate.

The full-prefix comparison is against separate-buffer native composition,
not an independent official 32-block oracle. Long index histories use controlled
repeated source-20 keys. Full-prefix contexts remain small. This gate does not
demonstrate complete generation, logits/quality, throughput or full-model fit
within 64 GB RAM and a 20 GB GPU. The user confirmed this gate before 6ag
began connecting original layers 32–35.

## Step 6ag: reuse the candidate index path for original layers 32–35

Implementation and the validation below are complete. The user confirmed
this gate before 6ah started. Its scope stopped at original layer 35; the final
backbone group is handled in 6ah below.

Source 32 reuses the candidate index query, shared attention and shared decoder
with its own original weights. It borrows source-20 KV, keys and candidate
blocks directly. Consumers 33–35 use source-32 indices. The existing decoder
loop and the two residual/two coefficient buffers are reused. One small
feature-local head-projection kernel was necessary for the precision issue
below; the general Strata kernels and attention arithmetic are unchanged.

### Head-weight precision correction

The first original-weight query run failed one of 148 checks: case 0, row 16,
head 22 produced `0.00201416015625` instead of `0.0019989013671875`.
All query bits, scores and selected-index checks passed. An independent dot
calculation gives `0.12841790614766069`; the BF16 midpoint is `0.12841796875`.
The original FP32 warp dot returned `0.1284182071685791`, crossing that midpoint
after cancellation. FP64 and long-double CPU sums agree. The existing serial
sum and all six tested split widths (32–1024 lanes) fail this same head; merely
changing the reduction width cannot fix the reference contract.

The shared ratio-1 path now accumulates only the 32-by-5120 head projection in
FP64, then rounds to FP32 before the existing BF16 projection and 1/64 scale
boundaries. It applies uniformly to sources 20, 24, 28 and 32, with no new
allocation. Ratio-2 accumulation, query FP8/FP4 processing, scoring, selection
and default Strata compute remain unchanged. No reference values or tolerances
were changed. This is a correctness fix, with no throughput claim.

An independent signed-cancellation test covers both sides of a BF16 midpoint:
it fails 16 of 32 head values before the fix and passes afterward. The original
source-32 fixture now passes all 148 checks. Initial failure and diagnosis
remain in `step6ag-initial-index-real.log`, `step6ag-rounding-repro-before.log`
and `step6ag-head-diagnosis.json` under `build-deepseek-cuda`.

The native shared suite passes 97 checks. Fourteen new checks keep index owners
24, 28 and 32 alive together: consumer 33 rejects either earlier publication;
resetting each index owner preserves the other two; resetting common source 20
invalidates all three. The 9-token lifecycle cases cover consumers 33–35.
The 513-position distinct-query control directly exercises consumer 33.

The analytical index suite passes 88 checks. A separate source-32 CPU fixture
uses pinned official Indexer.forward, FP64 linear dots and the established
BF16/FP4 boundaries. All 27 query rows pass 148 native checks, including two
controlled pruning histories at 16,385 and 16,393 positions. Re-preparing the
previous layer-28 dense fixture produces byte-identical output.

Original layer-32 dense preparation preserves 32 original tensors; layers
33–35 preserve 29 each. The connected prefix requested 106 additional
original experts from this group. All 936 distinct layer/expert pairs used
across 36 layers were independently inverted from the native GGUF layout and
verified against original FP4 codes and E8M0 scale hashes. Original layers 0–31
keep exactly the same routed expert-ID sets as gate 6af.

The original-weight prefix passes **703 checks** under both CUDA memcheck and
racecheck. Residuals, delayed coefficients, routes, source publications,
window/KV histories, reset and cache-budget variants match the separate-buffer
native composition. The synthetic prefix passes **820 checks** natively and
under memcheck, including failures in original layers 31–35 and layer 0.
Memcheck also passes the analytical index (88), original-weight index (148)
and full shared suite (97). Index racecheck passes 148 checks; shared lifecycle
and three-group racecheck passes 91 checks. These memchecks report zero errors
and zero leaked bytes; these racechecks report zero hazards. The 513-position
shared control ran natively and under memcheck; its long racecheck was not run.

Attention and query readers pass 378 and 161 checks in both Release and
ASan/UBSan. All 13 Python tests and 12 strict translation-unit builds pass.
Each CUDA unit retains only the same two established NVCC-generated header
warnings. The previous 32-element real prefix passes 595 checks with a native
log byte-identical to gate 6af after stripping sanitizer lines. The default
four-layer prefix passes 110 checks. Earlier component regressions, including
candidate groups 24 and 28, pass. Source-20 attention/decoder logs remain
byte-identical. Default core, kernels and engine archives match gate 6af with
the feature OFF and ON; the main Strata checkout remains clean.

### Allocations, reproduction and limits

On the RTX 2080 Ti reporting 22528 MiB, the synthetic prefix owns
17,126,288,032 device bytes, including 983,552 workspace bytes, with ratio-2
capacity 2, source-20 capacity 5 and prefill capacity 6. The real variants below
use ratio-2 capacity 3 and source-20 capacity 7. Owned device bytes exclude
CUDA/cuBLAS runtime, host weights/staging, file cache and caller buffers.

| Expert slots per layer | Owned device bytes | Prefix workspace bytes | Prefill capacity |
| --- | ---: | ---: | ---: |
| 6 | 17125802912 | 492032 | 3 |
| 2 | 14259913632 | 1147392 | 7 |

```sh
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(candidate_index_query32_cuda|source32_shared_cuda|text_prefix36_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_candidate_index_query_cuda_test \
  --layer 32 --real build-deepseek-cuda/candidate32-index-query-fixture.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 36 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The real commands require pinned fixtures and selected experts recorded in
`build-deepseek-cuda/step6ag-validation.json`. Adjacent logs retain the reference
and hash evidence; `step6ag.patch` isolates this gate from earlier work.

The full-prefix control is a native composition, not an independent official
36-block oracle. Long index histories use controlled repeated source-20 keys;
full-prefix contexts remain small. This gate does not demonstrate full text
generation, logits/quality, throughput or full-model fit within 64 GB RAM and a
20 GB GPU. The final original group 36–39 and its DSpark boundaries are
handled in 6ah below. Embedding and the output head remain later work.

## Step 6ah: complete the original 40-layer backbone

Implementation and the validation below are complete. The user confirmed this
gate before 6ai began. It connected original layers 36–39. Full backbone
coverage alone does not provide text generation.

### Pinned DSpark boundary

The hash-pinned reference keeps all backbone layers 0–39 at 384 routed experts
and top-6. `dspark_target_layer_ids=[37,38,39]` names ordinary backbone layers;
it does not turn those layers into DSpark blocks. Actual `DSparkBlock` objects
are separate `mtp.0..2`, corresponding to original IDs 40–42 and 128/top-3 MoE.

`Transformer.forward` captures `h.mean(dim=2)` **before** each target block,
then concatenates the three captures on the feature axis. This is an
unweighted mean of four residual streams, not a block output or the weighted
delayed `hc_pre` collapse. At full geometry the concatenated BF16 tensor has
shape `[B,T,15360]`. Ordinary forward does not execute MTP; draft execution is
in `forward_spec`. The reference does not include a speculative scheduling loop.

Two controlled executions of the unmodified pinned `Transformer.forward` AST,
with distinguishable tiny per-stream block values, verify prefill/decode order,
pre-layer mean capture, final weighted collapse and zero draft invocations.
The same audit executes `ModelArgs.get_moe_config` for original layers 36–42.
This is a boundary/order control, not a real-weight model parity test. Evidence:
`build-deepseek-cuda/step6ah-boundary.json` and its adjacent script.

`CudaTextPrefix` continues to return layer-39 residuals and delayed coefficients.
It does not capture DSpark target means or execute draft layers. A later output
head must apply the final weighted collapse, normalization and vocabulary
projection. DSpark capture, MTP execution and speculative scheduling remain
separate optimization work in step 7.

### Reuse and validation

Source 36 reuses the candidate index query, shared attention and shared decoder
with its original weights. It directly borrows source-20 KV, keys and candidate
blocks. Consumers 37–39 use source-36 indices. All four candidate groups keep
independent publication lifetimes. The existing decoder loop still uses two
residual buffers and two coefficient buffers; no new CUDA arithmetic kernel
is introduced. Gate 6ag's head-weight precision correction is preserved.

The four-group lifecycle test rejects cross-group consumption at matching
positions, preserves sibling publications when one group resets and expires
all publications when common source 20 resets. The 40-layer tests also reject
MTP layers and draft MoE geometry at the backbone boundary. All validation
below is complete.

### Exact scale117 storage adaptation

The first original-weight run stopped while preparing layer39 expert94. Its
down projection contains five blocks with E8M0 byte117 (`2^-10`); expert174
has six more. The existing adapter accepted only bytes118..135. Source hashes
were intact. Both failures were reproduced with the unmodified native expert
test; the initial logs are retained in `step6ah-initial-scale-failure`.

A per-projection global scale was considered using Strata's existing NVFP4
layout. Inspection found a smaller exact conversion for those eleven
blocks: none contains E2M1 values +/-0.5 or +/-1.5. The adapter raises only a
scale117 block to118 and halves its codes exactly. For example, a code for
`3 * 2^-10` becomes a code for `1.5 * 2^-9`. Zero signs are preserved. This
changes storage codes while preserving every decoded weight value; it is not
lossy re-quantization. Original MXFP4 fixtures retain all source codes/scales.
Expert cache layout, sizes and CUDA kernels remain unchanged.

Scale117 blocks containing +/-0.5 or +/-1.5, other unsupported scales and
nonfinite scales still fail. This extends the demonstrated lossless range; it
does not establish all-checkpoint scale coverage or arbitrary E8M0 support.
The user approved addressing this compatibility issue within 6ah.

Independent storage comparison checks all 23,592,960 values of the two affected
down matrices with zero differences, including zero signs. The CPU adapter
suite covers 768 accepted code/position cases and 256 rejected inexact cases,
in addition to its existing 256-scale sweep. Release and ASan/UBSan both pass.
Each affected original expert passes 41 native reference checks and the same 41
under memcheck/racecheck, with zero errors, leaked bytes or race hazards. The
reference uses original MXFP4 weights and the existing checked Q8 activation
baseline with FP64 dots; its tolerances were not relaxed. Evidence is recorded
in `build-deepseek-cuda/step6ah-scale-diagnosis.json`.

The analytical index suite passes 88 checks, including the existing signed
cancellation regression. The separate source-36 CPU fixture uses pinned
official `Indexer.forward`, FP64 linear dots and the established BF16/FP4
boundaries. All 27 query rows pass 148 checks, including two controlled pruning
histories at 16,385 and 16,393 positions. Re-preparing the previous layer-32
dense fixture produces byte-identical output.

Original layer-36 dense preparation preserves 32 original tensors; layers
37–39 preserve 29 each. The connected prefix requested 103 additional
original experts from this group. All 1039 distinct layer/expert pairs used
across 40 layers were independently inverted from the native GGUF layout and
verified against original FP4 codes and E8M0 scale hashes. Original layers 0–35
keep exactly the same routed expert-ID sets as gate 6ag. The exact scale117
path covers 27 blocks in 9 original layer39 experts.

The original-weight 40-layer prefix passes **823 checks** under both CUDA
memcheck and racecheck. Residuals, delayed coefficients, routes, source
publications, window/KV histories, reset and cache-budget variants match the
separate-buffer native composition. The synthetic prefix passes **964
checks** natively and under memcheck, including failures in original layers
35–39, layer 0 and the layer-14 Engram path.

The full shared suite passes 101 checks natively and under memcheck, including
18 independent four-group lifecycle checks and the 513-position source-36 /
consumer-37 query control. Shared lifecycle/four-group racecheck passes 95
checks. Analytical index memcheck passes 88; original index memcheck and
racecheck each pass 148. These memchecks report zero errors and zero leaked
bytes; these racechecks report zero hazards. The long 513-position shared
control was not run under racecheck.

Attention and query readers pass 426 and 197 checks in both Release and
ASan/UBSan. All 16 Python tests and 14 strict translation-unit builds pass.
Each CUDA unit retains only the same two established NVCC-generated header
warnings. The previous 36-element real prefix passes 703 checks with its native
log byte-identical to gate 6ag after stripping sanitizer lines. The default
four-layer prefix passes 110 checks. Eleven earlier component CTests and
original-weight candidate groups 24, 28 and 32 pass. Source-20 attention/decoder
logs remain byte-identical. Default core, kernels and engine archives match
gate 6ag with the feature OFF and ON; the main Strata checkout remains clean.

### Allocations, reproduction and limits

On the RTX 2080 Ti reporting 22528 MiB, the synthetic prefix owns
18,954,150,688 device bytes, including 983,552 workspace bytes, with ratio-2
capacity 2, source-20 capacity 5 and prefill capacity 6. The real variants below
use ratio-2 capacity 3 and source-20 capacity 7. Owned device bytes exclude
CUDA/cuBLAS runtime, host weights/staging, file cache and caller buffers.

| Expert slots per layer | Owned device bytes | Prefix workspace bytes | Prefill capacity |
| --- | ---: | ---: | ---: |
| 6 | 18953665568 | 492032 | 3 |
| 2 | 15769271328 | 1147392 | 7 |

```sh
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(candidate_index_query36_cuda|source36_shared_cuda|text_prefix40_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_candidate_index_query_cuda_test \
  --layer 36 --real build-deepseek-cuda/candidate36-index-query-fixture.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 40 --real build-deepseek-cuda build-deepseek-cuda/prefix-experts
```

The real commands require pinned fixtures and selected experts recorded in
`build-deepseek-cuda/step6ah-validation.json`. Adjacent logs retain source,
boundary and hash evidence; `step6ah.patch` isolates this gate from earlier work.

The full-prefix control is a native composition, not an independent official
40-block oracle. Long index histories use controlled repeated source-20 keys;
full-prefix contexts remain small. This gate does not demonstrate embedding,
logits, text generation, model quality, throughput or full-model fit within
64 GB RAM and a 20 GB GPU. It adds no DSpark capture or draft execution.


## Step 6ai: token embedding and the last-token output head

Implementation and validation are complete. The user confirmed this gate
before 6aj began. It adds the token embedding and last-token output boundaries.

### Fixed reference and reuse

The pinned checkpoint has vocabulary size 129,280 and hidden width 5,120.
`embed.weight`, `norm.weight` and the untied `head.weight` are BF16. Ordinary
`Transformer.forward` copies each embedding into four residual streams and
starts each token's delayed coefficients at `[1,0,0,0]`. After layer 39 it
uses the final coefficients for `hc_pre`, followed by RMSNorm with epsilon
`1e-20`. `ParallelHead` selects the last token and computes FP32 logits; it
does not round the output logits to BF16.

`read_text_io_weights` checks the pinned profile/revision, BF16 types, shapes,
alignment and bounds through Strata's mapped GGUF reader. `CudaTextIo` keeps
the embedding mapped on the host and gathers only requested rows into bounded
host staging. Four stream copies and initial coefficients use CUDA copies.
The output head retains its original BF16 device storage. Its arithmetic
uses the existing mHC collapse, native RMSNorm, BF16 rounding and Strata
`bf16_gemv_fp32_mmvf`; no new CUDA arithmetic kernel is introduced. The existing
mHC kernel itself is unchanged and is called through a shared internal helper.

Calls are synchronous, batch size one, on one explicit stream. Embedding
checks every original token ID and selected row before writing output.
Invalid dimensions, capacities, overlap and stream changes are rejected.
Execution failure requires reconstruction. The embedding's mapped file must
outlive `CudaTextIo`. Staging and GPU scratch are fixed at construction.

The caller composes the boundaries with the existing 40-layer prefix:

```cpp
io.embed(tokens, residual_in, initial_pre, stream);
prefix.forward(tokens, residual_in, initial_pre,
               residual_out, final_pre, start_pos, stream);
io.head(residual_out, final_pre, int(tokens.size()), logits, stream);
```

This boundary API returns one vocabulary-sized FP32 logits row for the last
token. It has no tokenizer, sampler or generation loop. It does not capture
DSpark states or run MTP.

### Validation

The reader passes **31 checks** in both Release and ASan/UBSan. Analytical
CUDA input/output passes **41 checks** natively, under memcheck and under
racecheck. Six changed translation units pass strict compilation; the CUDA
unit retains only the two existing generated-header warnings. An initial
strict-check harness invocation wrongly combined NVCC linking with host
syntax-only mode; the corrected compile-only command passes. That harness
failure and diagnosis are retained in `step6ai-strict-harness-diagnosis.json`.
The first original-boundary attempt stopped before numerical checks because
test IDs used an unsupported I32 tensor. They now use the reader's existing
U32 metadata arrays. All 45 numerical payloads and the token IDs were verified
unchanged; the diagnosis and initial logs are retained in
`step6ai-reference-format-diagnosis.json` and `step6ai-initial-io-format`.

All three original BF16 tensors were fetched from the pinned revision and
independently verified after GGUF packing: **2,647,664,640 bytes** are preserved,
including distinct input and output matrices. Three embedding cases cover
seven token rows, including ID 0, ID 129279 and repeated IDs, with bit-exact
four-stream values and initial coefficients. Six head cases compare all
129,280 logits each against the pinned CPU graph and FP64 dot controls. They
include multi-token last-position selection, zero/tiny/large inputs, signed
coefficients and actual embedding outputs. Original boundary tests pass
**35 checks** natively and under both CUDA checkers; the maximum-logit ID agrees
in every case, and nonzero cases retain FP32 output values.

The 40-layer composition passes **507 checks** with analytical boundaries and
**507 checks** with original embedding/head boundaries, each natively, under
memcheck and under racecheck. Both versions use the synthetic backbone. The
prefill of three tokens and two one-token decode calls repeat after reset.
Every final residual, delayed coefficient and routed expert matches the
separate-buffer native composition; all logits match independent CPU final
collapse/normalization/FP64 dot controls, and reset/replay logits are bit-exact.
The original-boundary standalone maximum relative L2 error is
`2.16251592e-07`; the original-boundary composition maximum is
`1.03621651e-07`. The declared comparison bounds are
relative L2 <= `2e-4` and maximum absolute error <= `2e-4 * (1 + reference_peak)`;
these are numerical comparisons, not bit-exact CPU/GPU logits claims.

All listed memchecks report zero errors and zero leaked bytes; all listed
racechecks report zero hazards. Original mHC regression passes
**151 checks**. The default real four-layer prefix passes **110 checks**
with its complete native log byte-identical to 6ah. Default core, kernel and
engine archives are byte-identical with the feature OFF and ON, the feature
is restored ON, and the main Strata checkout remains clean.

On the RTX 2080 Ti reporting 22528 MiB, standalone original text IO owns
**1,323,950,080 device bytes** and **81,984 host staging bytes**
for prefill capacity 4. The mapped embedding is not included in host staging.
The original-boundary/synthetic-backbone composition uses two expert slots per
layer, prefill capacity 3, ratio-2 capacity 2 and source-20 capacity 5:

| Allocation | Bytes |
| --- | ---: |
| Synthetic 40-layer prefix owned device storage | 15768609568 |
| Original text IO owned device storage | 1323950080 |
| Caller input/output device buffers | 1010784 |
| Text IO host staging | 61488 |

Owned device counts exclude CUDA/cuBLAS runtime allocations. These small-context
controls do not measure full-model peak memory or real-weight expert residency.

```sh
ctest --test-dir build-deepseek-cuda --output-on-failure \
  -R '^deepseek41_(text_io_weights|text_io_cuda|text_io_prefix40_cuda)_test$'
build-deepseek-cuda/src/deepseek41/deepseek41_text_io_cuda_test \
  --real build-deepseek-cuda/real-text-io.gguf
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 40 --text-io build-deepseek-cuda/real-text-io.gguf
```

The original commands require the pinned fixture recorded in
`build-deepseek-cuda/step6ai-validation.json`. Adjacent logs contain the executed
commands, source hashes and numeric errors. `step6ai.patch` isolates this gate.


The independent boundary fixture executes the unmodified pinned
`ParallelEmbedding`, `make_identity_pre_mix`, `Block.hc_pre`, `RMSNorm` and
`ParallelHead` on CPU. It also records FP64 output-dot controls. This validates
all vocabulary logits for controlled residuals, not official end-to-end
40-block parity. The connected test uses a synthetic backbone with distinct
coefficients, two Engram histories and all KV/index groups; optional original
embedding/head weights change only its two boundaries. A real-token run with
all original backbone weights, complete expert coverage and text generation
remains subsequent work. No throughput, quality or 64 GB RAM/20 GB GPU fit is
claimed by these controls.

## Step 6aj: original-weight token-to-logits integration

The user confirmed 6ai and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 6aj and authorized 6ak. The original token route exposed
a required expert-packing compatibility fix described below.

### Fixed integration control

The new opt-in `--text-io-real` mode uses the original BF16 embedding/head and
all 40 original backbone layers through the existing quantized native path.
The fixed tokens are `[0,455]` for prefill followed by `[1805]` for decode.
These are token IDs for a bounded integration control, not a generated answer.
The existing guarded Engram tables supply the exact rows for both layers.

An explicit composition uses a separate destination for every decoder. The
production prefix uses its two alternating buffers. The test compares every
route ID and weight, final residual/coefficient values, all 40 window caches
and all shared KV/index histories. It checks six versus two expert slots per
layer, Engram row caching enabled versus disabled, and reset/replay. Both
configurations return the full 129,280-element FP32 logits row.

`--text-io-probe` is read-only and rejects missing original experts.
`--text-io-prepare` uses the existing explicit preparation protocol: pause
before reading any expert in the selected route, then resume only after the
preparer supplies and verifies the bounded missing files. The preparer retains
original layer/expert identities, pinned revision and source hashes. Temporary
downloads created by this gate are removed only after the retained MXFP4
payload has been inverted and matched to the original code/scale bytes.
Existing source files and fixtures are preserved.

### Exact tiny-weight compatibility

The original route reaches layer 39 expert 163. Its down projection contains
four E8M0 scale-117 blocks, including six FP4 codes that the earlier code-halving
adapter cannot represent. Silently rounding those weights would change the
pinned model. The whole projection can instead shift its block scales by one
and use an exact global factor of 0.5. All 11,796,480 original values, including
signed zeros, match an independent full-matrix reconstruction.

The DeepSeek cache now uses Strata's existing NVFP4 layout:
`[gate | up | down | s_gate,s_up,s_down,0]`. Each projection keeps the earlier
payload and scale 1 when its direct conversion is exact. Only projections
requiring the extra range use scale 0.5, and only when every block is in
117..134. Unsupported ranges still fail explicitly. The original on-disk
MXFP4 bytes and source identities remain unchanged.

`CudaExpert` applies Strata's existing `scale_inplace` after GEMV, before any
BF16 rounding or activation. Host scale metadata follows each expert cache
slot through admission, hits and replacement. The cache adds 16 device bytes
per slot and 16 transfer bytes per miss; its host scale array uses three floats
per slot. Ordinary scale-1 projections launch no extra scaling operation.
No new CUDA arithmetic kernel is introduced.

### Validation

The original-weight integration passes **588 checks** natively, under
CUDA memcheck and under racecheck. It compares prefill and decode, repeats both
after reset, and uses six and two expert slots per layer. All 129,280 logits
are compared at each call. Residuals, coefficients, routes and cache histories
match the separate-buffer native control. Reset and cache-size changes preserve
logits bit-for-bit.

The selected coverage is **638 layer/expert pairs** across all 40 layers.
Every used expert was independently inverted to its original FP4 code and
E8M0 scale hashes. This gate added **410** supplemental experts
(**7,708,262,400 bytes**). All 47 dense, Engram projection/reference and text IO
fixture files match hashes from earlier preparation records. These counts
describe this fixed three-token route, not all 384 experts in every layer.

The CPU capture verifier passes both calls using the unmodified pinned
embedding and head graph, plus FP64 output-dot controls. Maximum relative L2
error against the official CPU head is `1.55203909e-07`; the independent C++
head control reports `6.63534479e-08`. Bounds remain
relative L2 <= `2e-4` and maximum absolute error <= `2e-4 * (1 + reference_peak)`,
with the maximum-logit ID also required to agree. This establishes numerical
head agreement, not bit-exact CPU/GPU logits or official 40-block parity.

The exact-scale fix passes full-value host decoder controls under Release
and ASan/UBSan, four Python guard tests, and the real expert-163 CUDA oracle.
Each of the three projection roles is tested with global scaling. The mixed
scaled/unscaled cache test covers cold/warm use, eviction and invalid metadata.
Expert 163 and the cache test pass memcheck and racecheck. Original experts
94 and 174 each pass 95 regression checks. The decoder and synthetic 40-layer
text IO checks also pass.

The complete original path reports zero memcheck errors/leaked bytes and zero
racecheck hazards. The earlier original four-layer test passes 110 checks.
Its log matches 6ai after subtracting exactly the expected 16-byte tail per
cache slot and miss; every other field remains identical. Eight changed
translation units pass strict compilation; NVCC retains only the same two
pre-existing generated-header parentheses warnings recorded in 6y. Python
syntax checks pass. Default core, kernel and engine archives are byte-identical
with the feature OFF and ON, and the feature is restored ON. Main checkout
is clean. Only CUDA was built and executed for this opt-in DeepSeek path.

Two preparation failures are retained in the validation evidence. A temporary
index first retained names of already converted raw downloads; pruning only
absent temporary entries fixed that helper. Later, original expert 163 exposed
the exact-scale compatibility issue above. Neither fix changes original
weight bytes or numerical tolerances.

On the RTX 2080 Ti reporting 22528 MiB, measured owned allocations are:

| Expert slots per layer | Prefix device bytes | Text IO device bytes | Caller device bytes | IO host staging bytes |
| --- | ---: | ---: | ---: | ---: |
| 6 | 18953499168 | 1323950080 | 846912 | 40992 |
| 2 | 15768774688 | 1323950080 | 846912 | 40992 |

Ratio-2 capacity is two groups, source-20 capacity is five tokens, and IO
prefill capacity is two. Prefix prefill capacities are two and four for six
and two slots respectively. Device counts exclude CUDA/cuBLAS runtime
allocations. Host staging excludes mapped weights and expert/Engram caches.
These figures do not establish full-model peak memory or a 20 GB fit.

```sh
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 40 --text-io-real build-deepseek-cuda \
  build-deepseek-cuda/prefix-experts build-deepseek-cuda/real-text-io.gguf \
  build-deepseek-cuda/token-logits-captures
python3 tools/deepseek41/verify_text_prefix_logits.py \
  --model-dir build-deepseek-cuda/reference-cache/text-io \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --capture-dir build-deepseek-cuda/token-logits-captures
```

These commands require the pinned local fixtures; the optional capture
directory must be new. Omitting it runs the native/CPU FP64 integration
checks without writing captures. `build-deepseek-cuda/step6aj-validation.json`
records commands, source identities, coverage, errors and resource counts.
`step6aj.patch` isolates this gate from earlier uncommitted work.


The C++ control compares all logits with independent CPU collapse,
normalization and FP64 dot products. Optional captures additionally run the
unmodified pinned `ParallelEmbedding`, initialization, `Block.hc_pre`,
`RMSNorm` and `ParallelHead` on CPU via `verify_text_prefix_logits.py`.
This is independent boundary validation on native final states. The backbone
control uses the same native layer implementations with separate buffers;
it does not establish independent official full-model parity.

The fixed three-token path exercises selected original experts and Engram
rows. It is not a complete model package for arbitrary prompts. Tokenization,
sampling, a generation loop, full-model quality, throughput and the
64 GB RAM/20 GB GPU target remain later work. This gate changes expert cache
metadata and projection scaling, while reusing the existing CUDA arithmetic
kernels and leaving the default Strata path unchanged.

## Step 6ak: pinned tokenization and bounded greedy completion

The user confirmed 6aj and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 6ak and authorized 7a. No new CUDA
arithmetic kernel is introduced. `CudaGreedyGeneration` composes `CudaTextIo`, `CudaTextPrefix` and
Strata's existing `sample_tokens` with greedy sampling enabled.

### Request and sequence contract

The request is batch one, raw completion without a chat template. At the Python
entry boundary, the pinned `tokenizer.json` is loaded using `tokenizers`; both
its SHA-256 and the pinned configuration hash must match. The entry prepends
BOS 0 to the tokenizer body with automatic special tokens disabled. It decodes
the entire generated sequence at once, preserving UTF-8 split across tokens.
The entry imports no Torch. Python launches the native runner once per request;
all model forwards, sampling and token feedback run in C++.

Each positive-budget request resets sequence histories while retaining weights
and expert caches. The initial forward evaluates the whole prompt. Subsequent
forwards consume the actual previously sampled token. EOS 1 is returned once
and stops the loop. The final sampled token is returned without another
forward, so the evaluated length is `prompt_length + max_new_tokens - 1`.
Zero budget returns an empty completion and leaves an existing session intact.
Invalid IDs, prompt/budget overflow and a changed/null stream are rejected
before resetting state. Execution failures return no partial completion and
require reconstruction, consistent with the underlying components.

Workspace is allocated once at construction. The sampler returns one integer
to the host per step; the runtime does not copy the full vocabulary logits to
Python or host memory. Existing router transfers inside the model remain.
The guarded runner uses two expert slots per layer, two ratio-2 group slots and
a five-evaluated-token capacity. Missing original experts or Engram rows fail
explicitly. This runner uses the existing test fixture loader; it is not yet a
general checkpoint loader or packaged server.

### Original-weight control and independent checks

The bounded prompt is `" The"`, encoded as `[0,455]`, with a budget of two new
tokens. All 40 original layers, both original Engram paths and the original
embedding/output head participate. The first generated token is fed through
all 40 layers to obtain the second. Only missing source rows and actually
selected experts are downloaded at the pinned revision. Existing fixtures are
preserved. New partial Engram files list populated rows and retain original
logical offsets; sparse holes are never treated as valid weights.

A separate explicit `embed → prefix → head → CPU argmax` loop controls the GPU
sampling loop. It compares output IDs, finish reason, all 40 window histories,
all shared KV/index histories, original expert coverage, cache accounting and
fixed allocations. Captures additionally use the pinned CPU embedding,
initialization, collapse, normalization and head via
`verify_text_prefix_logits.py --generation`. The second capture must consume
the first independent CPU-reference maximum. This verifies the boundaries and
feedback contract; it is not independent official full-backbone parity.

Synthetic controls exercise a tied greedy maximum, the exact context limit,
zero budget, invalid requests before mutation, reset/replay, a tied positive
EOS maximum, and execution-failure propagation. They use the real GPU sampler.
Tokenizer tests cover pinned sources, BOS handling, Unicode sequence decoding,
length/EOS results and malformed or partial native results.

### Measured validation on 2026-10-25

The original completion is `" The" → " The distance between"`. Generated IDs
are `[6520,1469]`; the second forward consumes ID 6520, and the final ID 1469 is
returned without an extra forward. The request finishes at its two-token
length limit. Both the native control and the Python text entry return this
same result. The first-forward captures remain byte-identical to 6aj.

| Check | Result |
| --- | --- |
| Native generated-token feedback across all 40 original layers | 46 checks, 0 failures; GPU sampler and CPU argmax IDs, all histories, original coverage and accounting agree |
| Synthetic generation state machine | 20 checks, 0 failures |
| Synthetic generation memcheck | 20 checks; 0 errors and 0 leaked bytes |
| Synthetic generation racecheck | 20 checks; 0 hazards, errors or warnings |
| Existing four-layer prefix regression | 140 checks, 0 failures |
| Pinned tokenizer and process-result contract | 5 tests pass; runtime entry imports no Torch |
| Independent pinned CPU boundaries | 2 cases pass; largest relative L2 error 1.30713432e-07 for the official head and 5.99503611e-08 for FP64 dots; greedy IDs agree |
| Previous 6aj CPU validation | Case results, source hashes and capture hashes unchanged |
| Strict C++ compilation | Both affected translation units pass `-Wall -Wextra -Werror` |
| Default Strata path | Core, kernels and engine archives identical to 6aj with the feature OFF and ON; main checkout clean |

The original route covers **648 unique layer/expert pairs**. This gate adds
**150 lossless original MXFP4 payloads**, totaling **2,820,096,000 bytes**,
after inverse code/scale hash verification. It adds 24 original Engram rows per
table: 48 new rows, or 12,672 payload bytes. The two new guarded sparse tables
contain all 144 required rows for this sequence and use 1,216,512 physical
file bytes. Their large logical extents do not mean complete tables are present.

On the RTX 2080 Ti reporting 22,528 MiB, the counted device arenas are
15,768,447,008 bytes for the 40-layer prefix,
1,323,950,080 for embedding/head boundaries and
845,316 for the generation workspace: **17,093,242,404
bytes (15.919 GiB)** total. The native control's measured process maximum
RSS is 9,414,440 KiB; the text entry run's maximum RSS is
9,394,516 KiB. These are owned device allocations and host
process RSS for this fixed control. They do not establish full-device peak,
whole-system RAM use, an enforced 20 GB GPU fit, general prompt coverage,
throughput or official full-backbone quality parity. GPU sanitizers cover the
new generation state machine with synthetic weights; the original generated
sequence is validated natively and with independent CPU boundary checks.

`build-deepseek-cuda/step6ak-validation.json` records the commands, sources,
routes, errors and resource measurements. `step6ak.patch` isolates the gate
from earlier uncommitted work. The gate changes 10 source/doc/test/build files;
all 83 pre-existing DeepSeek runtime/header files remain unchanged.

### Reproduction with the prepared local fixtures

```sh
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test --generation

python3 tests/deepseek41/test_generation.py

python3 tools/deepseek41/generate.py \
  --reference-root build-deepseek-cuda/reference-cache \
  --fixture-dir build-deepseek-cuda \
  --supplemental-dir build-deepseek-cuda/prefix-experts \
  --boundary-fixture build-deepseek-cuda/real-text-io.gguf \
  --engram-dir build-deepseek-cuda/step6ak-engram \
  --runner build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --prompt ' The' --max-new-tokens 2
```

To capture the original native control and check its CPU boundaries, choose a
new capture directory:

```sh
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --layers 40 --generation-real build-deepseek-cuda \
  build-deepseek-cuda/prefix-experts build-deepseek-cuda/real-text-io.gguf \
  build-deepseek-cuda/step6ak-engram build-deepseek-cuda/generation-captures

python3 tools/deepseek41/verify_text_prefix_logits.py \
  --model-dir build-deepseek-cuda/reference-cache/text-io \
  --reference-model build-deepseek-cuda/reference-cache/inference_model.py \
  --capture-dir build-deepseek-cuda/generation-captures --generation
```

The runner uses only local fixtures and never downloads during completion.
`prepare_generation_rows.py --discover` computes required rows using the pinned
official hash state. After the bounded source-row cache is populated, its
`--output-dir` mode writes a new guarded table directory. Torch is needed only
by offline preparation and reference checks. The validated fixed completion
does not imply coverage for arbitrary prompts, chat, long context or MTP.

## Step 7a: common startup CPU/GPU memory admission

The user confirmed 6ak and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 7a and authorized 7b. The text
entry checks both a configured allowance and live memory before opening any
model weight file.
A rejected request reports the limiting resource and returns no completion.

### Scope and reuse

`short_generation_memory` describes the current **1017-full40-context5**
fixture: all 40 original layers, at most five evaluated input tokens, ratio-2
capacity two, ratio-1 capacity five, and existing Engram reader defaults
(4096 cached rows per reader, 24 in-flight reads). The configurable expert slot
count is per layer; its cost is charged across all 40 layers. Unsupported
prompt capacities or slot counts are rejected. This profile is a conservative
startup bound, not a general context planner or an OS/allocator hard cap.

Admission reuses Strata's `device_info`, `sample_host_memory` and
`conversation_memory_admit`. Host availability includes the existing
MemAvailable/cgroup constraints; unknown availability fails closed. Existing
`DeviceArena`, `ExpertCache`, weight conversion and all model arithmetic stay
unchanged. No new CUDA arithmetic kernel or checkpoint download is introduced.
The pure CPU plan is part of the opt-in DeepSeek library; the budgeted entry
still uses the guarded test-fixture loader.

Device accounting includes dense weights, all expert slots and bounded
workspaces, with a separate 2 GiB CUDA/cuBLAS allowance. Host accounting charges
dense inputs conservatively at their larger device representation, the CPU
embedding, copied routers/maps, expert source mappings/staging and bounded
conversion/Engram scratch, with 4 GiB system/library headroom. The current
source allows at most 64 populated experts in each of seven mapped indices;
its staging is charged once per source, not once per GPU cache slot. A future
general source must supply new accounting. Full Engram tables are not resident.

For the default two-slot profile:

| Category | Bytes |
| --- | ---: |
| Dense device weights | 15,385,272,768 |
| GPU expert slots, all 40 layers | 1,592,526,080 |
| Device workspace bound | 337,641,472 |
| Device total before CUDA headroom | 17,315,440,320 (16.126 GiB) |
| Host dense-input/embedding/map bound | 16,977,535,424 |
| Host expert-source/staging bound | 8,600,739,968 |
| Host workspace bound | 536,870,912 |
| Host total before system headroom | 26,115,146,304 (24.322 GiB) |

Including headroom, startup requires **18.126 GiB GPU** and **28.322 GiB host**
allowance and live availability. These are conservative bounds, not measured
resident peaks. The code checks actual owned device arenas against the plan
and samples live GPU/host availability after construction and generation.
Sampling does not reserve memory against concurrent processes or measure a
continuous GPU peak; it does not impose a process RSS limit.

### Entry and measured validation on 2026-10-25

`generate.py` now defaults to `--gpu-budget-mib 20480`,
`--host-budget-mib 65536` and `--expert-slots 2`. `--plan` returns the bounds,
live availability and admission result without loading model weights. It
checks memory feasibility, not whether the supplied weight paths are complete.
The completion JSON retains its token/text fields and adds the admitted plan
and measured allocation accounting. Legacy native reference modes remain
available for their existing controls.

Both original-weight runs below produce `" The distance between"` with IDs
`[6520,1469]`, identical to 6ak. Both use
`STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB=20480`: the test runner holds the physical
card's excess memory in a Strata device arena for the entire request. Thus the
22 GiB card has at most 20 GiB left for CUDA allocations throughout generation.
This is a capacity simulation on the existing card, not a physical 20 GiB board.

| Measured quantity | Two slots/layer | One slot/layer |
| --- | ---: | ---: |
| Configured GPU startup allowance | 20 GiB | 18 GiB |
| Capacity left after test reservation | 20 GiB | 20 GiB |
| Counted prefix + text IO + generation arenas | 17,093,242,404 bytes | 16,296,979,364 bytes |
| Largest GPU allocation delta in two live samples | 18,373,148,672 bytes | 17,618,173,952 bytes |
| `/usr/bin/time -v` maximum RSS | 9,396,396 KiB | 9,396,068 KiB |

The one-slot configuration saves **796,263,040 bytes (759.376 MiB)** of counted
device allocations and returns the same tokens. It passes the 18 GiB startup
allowance, while two slots are rejected under that allowance. Both experiments
retain the 20 GiB capacity simulation; this is not an 18 GiB card test. Timings
include model loading and conversion and are not decode-throughput results.

| Check | Result |
| --- | --- |
| CPU admission boundaries and unknown/zero probes | 19 checks pass natively and with ASan/UBSan |
| Process-level preflight with nonexistent weight paths | Five cases pass: GPU/host/slot rejection and one-slot admission; rejection occurs before opening weights |
| Python tokenizer/result/admission contracts | Six tests pass |
| Existing GPU generation and four-layer prefix regressions | 20 and 140 checks pass; all three focused CTests pass |
| Strict compilation | All three affected C++ translation units pass `-Wall -Wextra -Werror` |
| Default Strata isolation | Core/kernels/engine archive hashes match 6ak with DeepSeek OFF and ON; main checkout clean |

This gate adds three files and modifies six source/build/test/doc files. All
85 pre-existing DeepSeek runtime/header implementation files remain unchanged.
`build-deepseek-cuda/step7a-validation.json` records commands and measurements;
`step7a.patch` isolates this gate from earlier uncommitted work. CUDA sanitizer
results from 6ak remain historical evidence for unchanged generation arithmetic;
7a does not claim a new GPU sanitizer run.

### Reproduction with the prepared local fixtures

Preview admission without opening model weights:

```sh
python3 tools/deepseek41/generate.py \
  --reference-root build-deepseek-cuda/reference-cache \
  --fixture-dir /not-loaded --supplemental-dir /not-loaded \
  --boundary-fixture /not-loaded --engram-dir /not-loaded \
  --runner build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --prompt ' The' --max-new-tokens 2 --plan
```

Run the fixed completion with the 20 GiB capacity simulation:

```sh
STRATA_IO_THREADS=2 STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB=20480 \
  python3 tools/deepseek41/generate.py \
  --reference-root build-deepseek-cuda/reference-cache \
  --fixture-dir build-deepseek-cuda \
  --supplemental-dir build-deepseek-cuda/prefix-experts \
  --boundary-fixture build-deepseek-cuda/real-text-io.gguf \
  --engram-dir build-deepseek-cuda/step6ak-engram \
  --runner build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --prompt ' The' --max-new-tokens 2 \
  --gpu-budget-mib 20480 --host-budget-mib 65536 --expert-slots 2
```

For the one-slot control, change the GPU allowance to `18432` and expert slots
to `1`; retain the same test-cap environment variable. General checkpoint
loading, arbitrary prompts, longer context, whole-system memory and throughput
remain unvalidated. This gate proposed 7b: bounded CPU expert caching and SSD
reads with cold/warm/eviction and byte-accounting checks. The user subsequently
confirmed 7a and authorized that implementation.

## Step 7b: bounded CPU expert cache and direct demand reads

The user confirmed 7a and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 7b and authorized 7c. A GPU expert miss can be served
from one bounded CPU cache shared by all 40 layers. Cache hits reuse the exact
native weight bytes and avoid an upstream payload read and conversion.

### Contract, reuse and bounds

`HostExpertCache` keys entries by original `(layer, expert)` and uses LRU
replacement. A fixed lookup table and linked slot metadata keep lookup and
replacement independent of cache size. Its allowance includes payloads and
all heap-allocated lookup/LRU metadata; entries never accumulate beyond that
allocation. Zero disables the CPU cache. A positive allowance must hold at
least one complete expert plus metadata, and larger allowances are capped at
the number of unique backbone experts. LRU is a correctness baseline; no
optimal-policy claim follows from this gate.

`CachedExpertSource` implements the existing Strata `ExpertSource` contract.
It forwards route validation and fills the host cache only after a successful
upstream read. `blob()` has one caller and returns evictable storage. Retained
consumers use `copy_blob()`; asynchronous consumers finish or notify their read
before the next call. `copy_blob()` serializes access to the upstream staging
buffer. Lifetime-stable pointers are explicitly unsupported. Existing
`ExpertCache`, GPU arithmetic and the lossless MXFP4 adapter remain unchanged.

The budgeted runner reads supplemental original MXFP4 files using Strata's
`DirectFile`. One aligned buffer, one open file and one demand read are active
at a time. Exact file size, successful completion, exact returned bytes and
lossless conversion are required before publication. There is no buffered
fallback or expert read-ahead. The Linux run uses the existing `O_DIRECT`
implementation. The source's single raw staging buffer is replaced by the
aligned buffer. Legacy reference/preparation modes retain their existing reader.

`--cpu-expert-cache-mib` defaults to **0**. With **512 MiB**, the original shape
fits **26 global native expert slots**: **517,632,728 bytes**
including metadata, below the 536,870,912-byte allowance. This is charged to
the common host budget before loading weights. The fixed-profile host bound
becomes **26,632,779,032 bytes**, plus the existing 4 GiB
headroom. The direct-read staging allocation is **18,800,640 bytes**, already
covered by the source bound. GPU memory requirements stay unchanged.

The underlying fixture still has seven bounded mapped expert indices; the
new direct reader covers supplemental files. Cache misses from those indices
still use the existing adapter. Consequently the reported direct-read bytes
cover supplemental payloads, not every source mapping, filesystem metadata,
or physical SSD-controller traffic. General checkpoint loading remains 7c.

### Measured validation on 2026-10-25

On the existing RTX 2080 Ti (22,528 MiB, CUDA 12.8, SM75), both original runs
reserve the card's excess memory so at most **20 GiB** remains available to
CUDA allocations. Both use two GPU expert slots per layer and the prompt
`" The"`, returning **`" The distance between"`**, IDs `[6520,1469]`, identical
to 7a. Counted device arenas remain **17,093,242,404 bytes** in both runs.

| Fixed short-generation measurement | CPU cache off | 512 MiB allowance |
| --- | ---: | ---: |
| CPU cache hits | 0 | 20 |
| Upstream expert reads/conversions | 720 | 700 |
| CPU cache evictions | 0 | 674 |
| Supplemental direct payload reads | 698 | 680 |
| Completed direct payload bytes | 13,122,846,720 | 12,784,435,200 |
| Maximum process RSS (`time -v`, KiB) | 9,395,572 | 9,901,424 |
| Elapsed time including load/conversion | 224.24 s | 226.97 s |

For this sequence, caching avoids **20 upstream
reads/conversions**, including **18 direct file reads
(338,411,520 payload bytes)**. The cache starts empty
in each process. This pair shows fewer demand reads but no improvement in
load-inclusive elapsed time. These are single-run measurements for this fixed
sequence, not steady-state decode-throughput results or a general speedup.

| Check | Result |
| --- | --- |
| Host cache/reader boundaries and failures | 34 checks pass, including ASan/UBSan |
| Three hash-verified original 5120/2304 expert payloads | 50 checks pass; cold/warm/eviction replay returns byte-identical native weights; six requests require four reads |
| Common memory admission | 23 checks pass, including ASan/UBSan; CPU cache cannot bypass the host allowance |
| Process preflight with nonexistent weight paths | Six cases pass; invalid/cache-over-budget requests fail before weight loading; Python plan preserves the native rejection reason |
| GPU source/cache integration | 48 checks pass; two-slot GPU eviction with a six-slot CPU cache preserves bitwise routed output |
| GPU memcheck and racecheck | The 48-check source/cache test passes both; zero leaked bytes, errors or race hazards |
| Existing generation and four-layer prefix | 20 and 140 checks pass; all five focused CTests pass |
| Python and strict C++ | Six Python tests and eight translation units with `-Wall -Wextra -Werror` pass |
| Default Strata isolation | Core/kernels/engine archives match 7a with DeepSeek OFF and ON; main checkout clean |

The initial GPU validation script could not find `compute-sanitizer` on PATH.
Using the existing CUDA 12.8 absolute path resolved this before any GPU test
ran; the initial log and final successful runs are retained. No runtime fix
was needed for that environment error. No weights were downloaded or modified.
CPU and CUDA validation ran on Linux; Windows, HIP and SYCL were not tested.

### Reproduction and remaining scope

Use the prepared fixtures with an explicit global CPU cache allowance:

```sh
STRATA_IO_THREADS=2 STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB=20480 \
  python3 tools/deepseek41/generate.py \
  --reference-root build-deepseek-cuda/reference-cache \
  --fixture-dir build-deepseek-cuda \
  --supplemental-dir build-deepseek-cuda/prefix-experts \
  --boundary-fixture build-deepseek-cuda/real-text-io.gguf \
  --engram-dir build-deepseek-cuda/step6ak-engram \
  --runner build-deepseek-cuda/src/deepseek41/deepseek41_text_prefix_cuda_test \
  --prompt ' The' --max-new-tokens 2 \
  --gpu-budget-mib 20480 --host-budget-mib 65536 --expert-slots 2 \
  --cpu-expert-cache-mib 512
```

Add `--plan` to inspect admission before loading weights. Set the CPU cache to
`0` for the uncached control. `step7b-validation.json` records commands,
sources and limits, and `step7b.patch` isolates this gate from prior work.
The gate changes 17 source/build/test/doc files. Model computation and the
underlying Strata allocator, GPU cache and direct-file implementation stay
unchanged; 83 pre-existing DeepSeek runtime/header files are unchanged.

The current fixture has bounded expert/Engram coverage and a five-input-token
capacity. Arbitrary prompts, complete checkpoint residency/I/O behavior, long
context, transfer overlap and whole-model quality remain unvalidated. The next
proposed gate, 7c, connects a common checkpoint weight index/source to these
bounded reads and caches. The user subsequently confirmed 7b and authorized
that next gate.

## Step 7c: common original-ID expert file index and source

The user confirmed 7b and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 7c and authorized 7d. Expert GGUF ranges and standalone MXFP4
files now feed the same bounded source, CPU cache and existing GPU execution.
This is the expert-loading part of general checkpoint support. Dense weights,
Engram packaging and the generation runner still depend on prepared fixtures.

### Source contract and reuse

`ExpertFileIndex` stores `(original layer, expert)` locations for gate, up and
down, including file identity, offset and expected file size. It reuses Strata's
`GgufModel` split discovery and the confirmed `ExpertIndex` shape/type/bounds
validation. The index owns metadata only; mappings used to construct it may
close before payload reads. Runtime access no longer branches on fixture names
or particular layer numbers. The fixture adapter registers existing guarded
GGUFs first and supplemental raw files second, preserving previous precedence.

The source accepts the pinned 1017 revision and executable MXFP4 role layout.
Missing IDs, duplicate registrations, malformed coverage, inconsistent shard
guards, unsupported role types and out-of-bounds ranges are rejected. Populated
coverage takes precedence over the legacy selected-ID guard, across all shards.
The complete expert-GGUF entry requires every original backbone expert:
**40 × 384 = 15,360 entries** at the production shape. Revision metadata is a
provenance agreement, not cryptographic payload verification. Source files must
remain immutable during inference; each opened file must still have its indexed
size. Existing offline payload provenance remains required.

`IndexedExpertFileReader` reuses `DirectFile` and the existing lossless MXFP4 to
native adapter. It handles unaligned GGUF starts, three roles in different
shards and a short aligned read at EOF when every required byte is present.
Adjacent raw gate/up/down ranges combine into one request. It keeps one open
file/read at a time, a fixed aligned slab and one assembled raw expert.
`IndexedExpertSource` owns the index and implements Strata's existing transient
`ExpertSource` contract. The entire top-6 is checked before payload access.
Caller-owned copies are serialized; asynchronous consumers finish or notify
before recycling a blob. CPU/GPU caching, admission formulas and math stay unchanged.

At original shape, reader buffers total **37,605,376 bytes**; with the reusable
native destination, source staging totals **57,511,952 bytes**. The budgeted
runner checks this allocation, index metadata and a conservative bound for
retained fixture mappings against the already-admitted source allowance before
constructing GPU model weights. It never copies the full set of expert weights into RAM.

### Measured validation on 2026-10-25

On the same RTX 2080 Ti, CUDA 12.8/SM75, with excess physical GPU capacity reserved
to leave **20 GiB**, the fixed original prompt `" The"` still produces
**`" The distance between"`**, IDs `[6520,1469]`. Two GPU slots per layer and a
512 MiB global CPU allowance match 7b. Counted GPU arenas remain
**17,093,242,404 bytes**. CPU cache statistics are unchanged: **20 hits,
700 misses and 674 evictions**, using **517,632,728 bytes** including metadata.

| Unified-source measurement | Result |
| --- | ---: |
| Registered original expert entries / files | 1,702 / 1,520 |
| Index metadata upper accounting | 1,439,780 bytes |
| Source staging + metadata + retained mapped-expert bound | 3,612,272,692 bytes |
| Existing admitted source allowance | 8,600,739,968 bytes |
| Converted expert misses | 700 |
| DirectFile requests / failures | 740 / 0 |
| Logical expert payload bytes | 13,160,448,000 |
| Requested and completed aligned bytes | 13,160,693,760 |
| Maximum process RSS (`time -v`) | 9,452,996 KiB |
| Load-inclusive elapsed time | 225.91 s |

The 740 requests comprise 680 contiguous raw-file requests and 60 GGUF role
requests for 20 experts. In 7b the latter payloads used mappings and were absent
from direct-read counters. These counters now cover every expert source miss;
they do not measure physical SSD-controller traffic. The one run establishes
functional regression and bounded accounting, not a throughput improvement.
The GPU allocation delta from two live samples is **18,380,554,240 bytes**;
this is not a continuous peak. The physical card remains 22 GiB with a capacity
reservation, rather than a physical 20 GiB board.

| Check | Result |
| --- | --- |
| Split-file index, offsets, guards, ID identity and failures | 32 checks pass natively and under ASan/UBSan |
| Original 5120/2304 GGUF and raw payload controls | 34 checks pass; native bytes exactly match the confirmed mapped/raw adapters |
| Existing host cache and memory admission | 34 and 23 checks pass, including ASan/UBSan |
| Routed GPU integration | 57 synthetic checks; 101 combined synthetic/original checks; bitwise output invariant through CPU/GPU eviction |
| GPU memcheck / racecheck | 57-check integration passes both; zero leaked bytes, errors or race hazards |
| Existing generation and four-layer prefix | 20 and 140 checks pass; all six focused CTests pass |
| CLI source option and budgets | Three process cases pass, including a path with spaces; `--plan` does not open weights; insufficient host/cache allowances reject |
| Python and strict C++ syntax | Six Python tests and seven translation units with `-Wall -Wextra -Werror` pass |
| Default Strata isolation | Core/kernels/engine archives match 7b with DeepSeek OFF and ON; main checkout clean |

The new non-root-shard guard test initially failed and exposed an omission in
this gate's first implementation. Scanning all shards and rejecting conflicting
guards fixed it; the initial failure is retained in `step7c-index-initial.log`.
The host sanitizer build also reports an existing format-truncation warning in
Strata's unchanged `gguf_split.hpp`; final affected-source strict syntax checks
pass. CPU/CUDA ran on Linux; Windows, HIP and SYCL were not tested.

### Entry, evidence and next gate

The 7b reproduction command now uses the unified source automatically. An
optional `--expert-gguf PATH` selects a complete expert GGUF, including split
files; it requires the pinned revision metadata and current MXFP4 layout.
Dense/Engram paths are still required. `--plan` inspects memory feasibility,
not checkpoint existence or coverage. No complete original expert checkpoint
was supplied or downloaded: complete coverage is tested on synthetic split files,
while original generation uses the existing guarded subset. Arbitrary prompts,
long context and full-model quality remain unvalidated.

`build-deepseek-cuda/step7c-validation.json` records the commands, source
provenance, measurements and limits. `step7c.patch` isolates this gate from
previous uncommitted work. The gate changes 14 source/build/test/doc files,
including three new files; 87 pre-existing DeepSeek runtime/header files are
unchanged. No model weights were downloaded or modified.

The proposed next gate, **7d**, is one explicit model manifest and loading entry
for dense weights, Engram and the expert source, with declared partial/full
coverage and revision checks. This removes the runner's hardcoded fixture-path
assembly while preserving the fixed-generation control. The user subsequently
confirmed 7c and authorized that implementation.

## Step 7d: model manifest, shared loader and independent generation entry

The user confirmed 7c and authorized this gate. Implementation and validation
are complete; the user subsequently confirmed 7d and authorized the revised 7e. `strata-deepseek41-generate` is a new
opt-in native executable using `LoadedModel`, `CudaTextPrefix`, `CudaTextIo`,
`CudaGreedyGeneration` and the confirmed source/cache/admission components.
It builds with `STRATA_BUILD_TESTS=OFF` and has no test-code dependency. Existing
component arithmetic, kernels and legacy reference runners stay unchanged.

### Manifest and loader contract

The manifest is a metadata-only GGUF, parsed with Strata's existing reader.
Schema 1 requires `general.architecture=deepseek41-manifest`, the pinned
revision and `1017-full40-context5` profile. Relative paths resolve against the
manifest directory. The parser rejects unknown fields, incorrect types, missing
paths, duplicate source IDs, out-of-range IDs/counts and embedded tensors. The
file is limited to 8 MiB before mapping. It describes external weights; it does
not copy, download or authenticate them.

Keys below use the prefix `strata.deepseek41.manifest.`:

| Field group | Contract |
| --- | --- |
| `version`, `revision`, `profile` | U32 schema 1 and pinned string identities |
| `reference_root`, `text_io` | Tokenizer/config directory and embedding/head component |
| `layer.L.{mhc,attention,shared,router}` | Explicit component paths for each original layer 0–39 |
| `layer.L.{router_weight,router_bias}` | Router tensor names, checked for original shape/type/alignment |
| `engram.L.{projection,token_map,table,complete}` | Original layers 1 and 14, with independent U32 coverage flags |
| `expert_gguf_count`, `expert_gguf.I.{layer,path}` | At most 40 layer entries, reusing the 7c split/range index |
| `expert_raw_count`, `expert_raw.I.{layer,expert,path}` | Explicit raw MXFP4 entries; at most 15,360 original IDs |
| `experts_complete` | U32 0 for guarded coverage or 1 requiring all 15,360 experts |

`LoadedModel` owns component mappings and the views passed to existing
constructors. It checks pinned revision declarations across component shards,
then reuses every existing dense/profile/shape reader. Original source-state
ownership is fixed by the 1017 geometry. Expert payload mappings used to build
the index may close before reads. Engram coverage must agree with its manifest
flag; the existing `EngramTable` validates table profiles, ranges and guarded
row IDs before GPU model allocation. Partial tables retain hole rejection.

The Python entry resolves the pinned tokenizer/config through the manifest;
its existing hash checks remain in force. Runtime tokenization imports no Torch.
The native CLI consumes token IDs and does not tokenize. Manifest metadata,
source staging and a conservative retained-expert-mapping bound are charged
against the existing source allowance. The fixed startup plan remains admission
plus live checks, rather than an OS memory cap. `--plan` validates the manifest
and memory feasibility without opening referenced model weights.

Revision metadata is agreement with the declared source, not cryptographic
payload authentication. The current component GGUF layouts and offline
provenance remain required. This loader is not a direct importer for arbitrary
Hugging Face safetensors or unrelated GGUF quantizations.

### Reproduce with the prepared original weights

Export a small manifest once; the exporter refuses to overwrite its output:

```sh
python3 tools/deepseek41/prepare_model_manifest.py \
  --output build-deepseek-cuda/step7d-model.gguf \
  --reference-root build-deepseek-cuda/reference-cache \
  --fixture-dir build-deepseek-cuda \
  --supplemental-dir build-deepseek-cuda/prefix-experts \
  --boundary-fixture build-deepseek-cuda/real-text-io.gguf \
  --engram-dir build-deepseek-cuda/step6ak-engram
```

The existing model-file names are confined to this preparation adapter. Native
loading follows the explicit paths in the manifest. The measured manifest is
**375,648 bytes** and references the same 54 component model mappings,
1,702 expert entries and original Engram row subsets used by the control.
No weight payloads were copied or downloaded.

Build and run the independent entry:

```sh
cmake --build build-deepseek-cuda --target strata-deepseek41-generate -j4
STRATA_IO_THREADS=2 STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB=20480 \
  python3 tools/deepseek41/generate.py \
  --model-manifest build-deepseek-cuda/step7d-model.gguf \
  --runner build-deepseek-cuda/src/deepseek41/strata-deepseek41-generate \
  --prompt ' The' --max-new-tokens 2 --cpu-expert-cache-mib 512
```

Add `--plan` for preflight. The CPU/GPU allowance defaults remain 64/20 GiB.
The Python entry refuses to combine `--model-manifest` with legacy model-path
options; its previous fixture-based invocation remains available as a control.

### Measured validation on 2026-10-25

The independent runner was launched from `/tmp`, with absolute paths to the
manifest and runner, on the same RTX 2080 Ti / CUDA 12.8 / SM75. Reserving the
physical card's excess capacity leaves **20 GiB** available. It returns
**`" The distance between"`**, IDs `[6520,1469]`, identical to 7c. Counted GPU
arenas are still **17,093,242,404 bytes**. CPU cache results remain 20 hits,
700 misses and 674 evictions, with 517,632,728 allocated bytes. The 740 direct
expert reads and 13,160,693,760 completed bytes also match 7c, with no I/O failures.

The source/manifest accounting is **3,612,549,907 bytes**, below the existing
8,600,739,968-byte source allowance. Maximum process RSS is **9,298,588 KiB**;
the observed GPU delta from two samples is **18,373,148,672 bytes**. Elapsed time is
**228.05 seconds including loading**. These measurements do not establish a
throughput gain, continuous GPU peak, full-system memory use or full-checkpoint
feasibility. The card is physically 22 GiB, with a capacity reservation.

| Check | Result |
| --- | --- |
| Manifest parser boundaries and geometry | 17 checks pass |
| Original loader, lifetime, wrong versions/layers/files and false coverage | 29 combined checks pass natively and under ASan/UBSan |
| Process preflight and CLI contracts | Eight new cases pass; legacy Python plan also passes |
| Existing memory/cache/index and GPU generation/prefix | All six focused CTests pass, including 20 generation and 140 prefix checks |
| Python and strict C++ syntax | Six Python tests and four affected translation units pass |
| Standalone build | Native generator builds with tests disabled; tests restored ON |
| Default Strata isolation | Core/kernels/engine archive hashes match 7c with DeepSeek OFF and ON; main checkout clean |

No new GPU sanitizer run was needed for new host loading/CLI code: CPU sanitizers
and the original GPU execution were rerun; GPU arithmetic did not change.
The 7c/6ak GPU sanitizer evidence remains historical. An initial indentation
warning in the new test was corrected. The sanitizer build retains the existing
`gguf_split.hpp` format-truncation warning. An initial review script ran while
the feature-isolation check had the option OFF; rerunning review after restoring
ON passed. These validation-script/build observations are retained separately
from model correctness results. Windows/HIP/SYCL were not tested.

`build-deepseek-cuda/step7d-validation.json` records commands and results;
`step7d.patch` isolates this gate. Ten source/build/test/doc files change,
including seven new files. All 93 pre-existing DeepSeek runtime/header files
remain unchanged.

This completes the independent loading/entry gate, not complete model coverage.
The manifest accurately declares guarded experts and both Engram tables; only
the fixed short sequence is accepted as measured evidence. The original proposal for
**7e** was complete model-package import and coverage acceptance. After reviewing
capacity, the user authorized a smaller quantization audit instead, documented
below. Full weights, arbitrary prompts, longer context, quality and measured
scheduling remain later work.

## Step 7e: bounded IQ1_M sample and upstream Strata audit

The user authorized the recommended quantized route, retaining the requirement
to stop after each review gate. The user subsequently confirmed 7e and authorized
7f. This gate downloads small real ranges, checks
SM75 quantized arithmetic and measures direct SSD reads. Complete model import
and migration to a different runtime are outside this gate.

### Pinned sources and download size

The audited [upstream Strata engine](https://github.com/Niko1221/Strata/blob/7dd5cf18e68c639a8773e096c5c79f971ea2a9f5/docs/DEEPSEEK41.md)
is commit `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`. It already has a
`strata-dsv41` engine and a GSQ intake/packing path. The downloaded source
subset has 22 files, verified against the commit's Git blob IDs and local
SHA256 records. Existing license and attribution files are retained with it.
Only its IQ and trellis CUDA translation units were compiled here; the complete
upstream engine has not yet been built or validated on this machine.

The [GSQ-RCO package](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/cb02422c900df489cea5833e8a92171224059b3c)
is pinned to `cb02422c900df489cea5833e8a92171224059b3c`. Its four IQ1_M
shards total **120,721,120,256 bytes**. The shared IQ4_NL Engram file is
**110,595,290,816 bytes**. Together they are **231,316,411,072 bytes
(231.32 GB)**. Upstream's normal setup also includes an approximately 8 GB
DSpark drafter, bringing that installation to about 239 GB. Neither complete
package nor drafter was downloaded in this gate. At validation, the local SATA
SSD had 196.65 GB available, less than the complete base package.

`prepare_quant_sample.py` reuses Strata's GGUF reader/writer. Each request is
at most 8 MiB and the requested payload budget is 96 MiB per invocation. Exact
uncompressed HTTP 206 ranges, total file sizes, shard IDs, tensor extents and
original 40-layer/384-expert/top-6 geometry are checked. A server returning a
whole file instead is rejected before reading its body. Output must be a new
directory; incomplete attempts retain their received-range records.

The successful run transferred **64,088,784 bytes**, including headers, and
wrote a **54,653,696-byte** compact sample. Including the two earlier failed
attempts, total completed model range payloads were **142,070,480 bytes**.
Source/API metadata are additional. Per-range hashes and the compact sample
hash are recorded; the published hashes of entire remote shards were not
independently verified, because those shards were not downloaded.

Sample SHA256:
`80df5ed95d53782ddb4b7088d5fdfd7b2b61e9afc54c56bf2baa04284dafc1e8`.
The compact artifact's 21 matrices and five Engram rows match their received
range hashes. The artifact declares the original 1017 revision used by this
project; sample parity does not establish full-model identity or quality.

### Reuse and actual format coverage

IQ1_M is a mixed-format package. Its 120 routed expert tensors contain 83
IQ1_S, 15 IQ1_M, 20 IQ2_XXS and two IQ2_S tensors. The older native IQ dispatch
in this worktree lacks IQ1_S. The probe therefore directly compiles unchanged
upstream Strata IQ kernels, while retaining the existing worktree's IQ4_NL
row decoder, GGUF reader, CPU ggml reference and `DirectFile` implementation.
No new GPU operator was written and no existing generation path was changed.

Seven sampled experts cover all gate/up-versus-down format pairs in the
40-layer backbone. Each sampled expert supplies all three full matrices:
gate/up `[5120,2304]` and down `[2304,5120]`, in GGUF dimension order.

| Original layer / expert | Gate and up | Down | Layers with this pair |
| --- | --- | --- | ---: |
| 0 / 84 | IQ1_S | IQ1_S | 17 |
| 7 / 6 | IQ1_S | IQ2_S | 2 |
| 9 / 7 | IQ2_XXS | IQ1_S | 7 |
| 11 / 4 | IQ1_S | IQ2_XXS | 2 |
| 12 / 3 | IQ1_S | IQ1_M | 7 |
| 25 / 1 | IQ2_XXS | IQ1_M | 2 |
| 28 / 6 | IQ1_M | IQ1_S | 3 |

Engram remains two original tables: layer 1 has 384,006,168 rows; layer 14 has
384,016,682 rows. IQ4_NL stores each 256-value row in 144 bytes. The sample
checks rows 0, 1 and the last row of layer 1, plus row 0 and the last row of
layer 14. It does not populate those tables or run the Engram hash lookup.

The upstream IQ matrix API consumes signed int8 activations with one FP32 scale
per 32 values. The current NVFP4/Q8_1 path has a different activation layout.
Connecting these paths requires an explicit adapter or reuse of the upstream
runtime; reinterpreting the old activation scratch is invalid. This gate
checks the upstream input contract directly, without claiming full MoE parity.

### Measured validation on 2026-10-25

Hardware: RTX 2080 Ti, SM75, CUDA 12.8; Fanxiang S103Pro 2 TB SATA SSD;
64 GB host RAM. The physical GPU reports 22 GiB. This small probe does not use
the earlier 20 GiB capacity reservation and does not measure full-model memory.

All 21 matrices pass three deterministic activation columns each, including
an exactly zero column, against independent CPU ggml dequantization and double
precision dot products on the same quantized weights. The predeclared limits
are relative L2 `5e-5` and maximum absolute error
`2e-4 * max(1, maximum absolute reference)`. Observed worst relative L2 is
**8.14e-8**; worst absolute error is **8.71e-7**. All **1,280 Engram output
values** match ggml FP32 output bit for bit. These are arithmetic checks,
not a comparison of IQ1_M model quality with the original checkpoint.

Resident single-column GPU projections take **31.71–35.78 microseconds** across
the 21 sampled matrices: five warmups, then the median of seven trials with
30 launches per trial, measured by CUDA events. Weights are already on the
GPU; timings exclude transfer, activation quantization, full expert execution,
routing and generation. There is no before/after speedup claim.

`DirectFile` performs five passes of 21 page-aligned reads from the compact
sample, one request in flight. Each pass reads 54,735,616 physical bytes for
54,650,880 logical expert bytes; returned bytes are checked against the sample.
The measured pass times are 114.447, 112.192, 112.049, 112.744 and 113.981 ms:
median **485.49 MB/s**. Linux O_DIRECT bypasses the OS page cache, but this is
a repeated 55 MB file, not random access over a complete 231 GB checkpoint.
It does not measure SSD controller cache behavior, Engram lookup latency,
end-to-end overlap or model tokens/s.

From the actual expert headers, an all-miss token would read **1,789,931,520
logical expert bytes** (40 layers, top-6, absent from both CPU and GPU caches).
Dividing by the sample's measured rate gives about **3.69 seconds of read
service time**. This is an arithmetic scenario, not a generation measurement:
cache hits reduce that traffic, and computation, Engram, storage access patterns
and overlap change total latency. It motivates checking cache behavior before
claiming interactive usability on this SATA machine.

| Check | Result |
| --- | --- |
| Bounded HTTP ranges, limits and integer header sizing | Seven Python checks pass |
| Final probe C++ syntax | `-Wall -Wextra -Werror` passes |
| Unchanged upstream CUDA sources | Build for SM75 succeeds |
| Actual sample arithmetic | 21 matrix checks and 1,280 exact Engram values pass |
| CUDA memcheck / racecheck | Zero errors / zero hazards, errors and warnings |
| Existing manifest, file-index and Engram host tests | Three CTests pass |
| Existing expert and Engram CUDA tests | Two CTests pass |
| Default Strata isolation | Core/kernels/engine archives match 7d with the probe ON and OFF |

Retained initial failures: the sample reader did not size integer metadata;
it then incorrectly assumed one combined Engram table. Both were corrected in
the new sampling adapter. An initial build lacked required upstream include
files. The first GPU run reached Engram and rejected the upstream row API's
unsupported IQ4_NL type; the final probe reuses Strata's existing IQ4_NL decoder.
The first sanitizer invocation used a command absent from PATH; rerunning with
the installed CUDA toolkit's absolute path passed. Upstream unused-code build
warnings remain. These failures are preserved separately from final successes.

No full model download, full-engine build, quantized generation, quality
benchmark, CPU quantized execution, arbitrary prompts, long contexts,
Windows/HIP/SYCL execution or throughput acceptance was performed.

### Reproduce and next gate

Provide the pinned upstream source root, including its ggml headers. CMake
checks hashes of the compiled upstream files and their local include
prerequisites. The recorded source root is
`build-deepseek-cuda/step7e-sources/upstream`. To obtain a new sample, run
`python3 tools/deepseek41/prepare_quant_sample.py --output-dir NEW_DIRECTORY`.
Do not overwrite or redownload the recorded sample just to rerun the probe.

```sh
cmake -S . -B build-deepseek-cuda \
  -DSTRATA_DEEPSEEK41_QUANT_PROBE=ON \
  -DSTRATA_DEEPSEEK41_UPSTREAM_DIR="$PWD/build-deepseek-cuda/step7e-sources/upstream"
cmake --build build-deepseek-cuda --target strata-deepseek41-quant-probe --parallel 3
STRATA_IO_THREADS=2 build-deepseek-cuda/src/deepseek41/strata-deepseek41-quant-probe \
  build-deepseek-cuda/step7e-sample-final/quant-sample.gguf
```

`STRATA_DEEPSEEK41_QUANT_PROBE` is OFF by default and restored OFF after
validation. `--check-only` omits projection timing for sanitizer runs. The
DeepSeek feature and existing tests remain ON in the working build.
`build-deepseek-cuda/step7e-validation.json` records the results and limits;
`step7e.patch` isolates this gate. Six source/build/test/doc files change;
all 98 pre-existing DeepSeek runtime/header files remain unchanged.

The proposed **7f** was a bounded SM75 build and intake compatibility gate for
the pinned upstream `strata-dsv41` engine. The user confirmed it; its outcome is
below. The original 7d control remains available as a comparison. A successful
IQ probe alone does not validate the complete engine or model.

## Step 7f: complete upstream engine build and intake readiness

The user confirmed 7e and authorized this gate. The complete pinned upstream
`strata-dsv41` now builds for SM75 and its version/help entry points execute.
The original worktree's runtime remains unchanged. This gate uses upstream
implementation directly: no operator, loader or scheduler was rewritten.
The new executable is an isolated build, not a replacement for the 7d runner.

### Source and build provenance

The source is Strata commit `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`, the
same commit inspected in 7e. The GitHub source archive downloaded here is
**29,973,463 bytes**, SHA256
`ad505f50d0740d2af4f1c7b96a631a811684ae1639e2e639f4d5495cae6460bd`.
All 4,892 regular archive files match their pinned Git blobs after restoring
21 archive line-ending conversions. Only the build/intake source subset is
extracted: 1,137 files, 19,375,259 bytes, including licenses and notices.
Every extracted source file remains unchanged after the build and tests.
No additional model weights, metadata shards or DSpark files were downloaded.

The existing llama.cpp source archive at
`3cf03257f219afbe7334045ff7c6a06ac68c627d` supplies ggml. Its 1,376 ggml files
match the existing local archive byte for byte; that archive's SHA256 is
`c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280`.
CMake's ggml commit display inherits the enclosing worktree commit because the
dependency has no `.git`; it is not the dependency's actual revision.

Build from this worktree, using the recorded source subset or a checkout of
the same upstream commit:

```sh
cmake -S build-deepseek-cuda/step7f-upstream -B build-deepseek-upstream-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_GGML_DIR="$PWD/build-deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d" \
  -DSTRATA_BUILD_ID=7dd5cf18e68c639a8773e096c5c79f971ea2a9f5
cmake --build build-deepseek-upstream-sm75 --target strata-dsv41 ds4_iq_test --parallel 3
build-deepseek-upstream-sm75/strata-dsv41 --version
```

The build includes the complete model, attention/dense/vision translation units,
GGUF loader, quantized CPU and GPU paths, and upstream's pre-SM80 vendor kernels.
`STRATA_PORTABLE=ON` uses an AVX2 CPU baseline. On this machine the CPU is an
Intel Xeon E5-2686 v4, 18 cores / 36 threads. CUDA is 12.8, GCC 13.3. The binary
is **52,511,960 bytes**, SHA256
`30982a43ad4dd82975d0fc33b49d5c80e42deb846bff621aab7ba10d3cdfe9f3`.
It reports `strata-dsv41 0.1.49.2` and the full pinned build ID. `cuobjdump`
finds ten ELF code objects, all SM75. Compilation does not establish that every
runtime path is correct on SM75.

### Reused tests and actual headers

```sh
build-deepseek-upstream-sm75/ds4_iq_test --type 16,19,22,29 --devices 0
```

The unmodified upstream test passes **24 checks** across IQ2_XXS, IQ1_S,
IQ2_S and IQ1_M: single-token gate/up and down, batched gate/up and down, and
two CPU comparisons per type. These checks use the upstream synthetic data and
its existing tolerances. They are separate from 7e's real-weight sample checks;
no whole-model accuracy or throughput claim follows. CUDA memcheck reports
zero errors and racecheck zero hazards, errors and warnings on this test.

Seventeen existing Python checks in `tools.test_setup_dsv41.Fit`,
`OfficialSets` and `Smallest` pass in the upstream source directory. They cover
planning, pinned files and the official packing recipe; they do not install or
download a model.

The real GGUF headers already saved in 7e also contain the first shard's
integer Engram metadata. Upstream's existing `Gguf`, `model_kv`,
`int_tensors_to_engram`, `geometry`, `write_engram` and `write_tokenizer`
functions parse these directly. No network request or fabricated weight data
is needed. The resulting geometry agrees with **36 shared original config
fields**. The two epsilon values match after exact FP32 serialization of the
JSON reference values. EOS is 1 in the quantized header; the selected reference
text-config has no direct EOS field, so it is not counted among the 36 checks.

The generated Engram metadata is **517,980 bytes**, SHA256
`08d55d6433f389b14aa115f4ea4eb31d6a2d63df1ae40821f03facf4c2d49142`:
original layers 1 and 14, 129,280 token-map values, eight multipliers, 48 primes
and 48 offsets. The tokenizer artifacts total 4,630,753 bytes. This validates
metadata conversion only, not chat-template behavior or model generation.

Upstream's Python size check detects all five saved headers as incomplete
files. Native `--pack-info` also rejects five truncated backing files with
exit code 2 while `CUDA_VISIBLE_DEVICES=-1`. That negative fixture contains
the actual 46,080 routed projection descriptors, two Engram descriptors and
one representative dense extent to include shard 4, which contains no routed
experts. Other dense tensors are absent, and the fixture is explicitly marked
as unusable for inference. A missing pack is separately rejected with exit 1.
No sparse files were created to simulate complete model weights.

### Capacity and package findings

The filesystem had **416.14 GB free** at the start of this gate and **415.85 GB**
at final checks. The 196.65 GB observed in 7e is historical; no files were
removed by this gate to create the extra space.

The pinned upstream planner, called with 64 decimal GB RAM, a 20 GiB GPU budget
and an 8,192-token context request, produces these **estimates**:

| Item | Planner result |
| --- | ---: |
| Dense GPU weights | 4.5 GB |
| GPU-resident routed experts | 2,121 of 15,360, about 15.8 GB |
| Experts assigned to the CPU side | 98.7 GB |
| RAM needed to cache that entire CPU side, including its allowance | 107.7 GB |
| Amount beyond the supplied RAM allowance | 43.7 GB |
| Classification | `tight`, uses SSD reads |
| Normal upstream download, including DSpark and metadata | 239.3 GB |
| Download plus upstream's 6 GB working allowance | 245.3 GB |

The planner is a coarse recommendation, not hard memory admission. Its GPU
dense allowance is derived from other cards, and its context field does not
calculate exact KV and prefill workspace growth. Actual VRAM allocation uses
live free space and `--reserve-mib`. The default RAM store uses currently
available RAM minus 12 decimal GB, capped at 48 GB, and checks its remaining
room during operation. OS page-cache behavior and temporary buffers also
matter. No full-model 64 GB / 20 GiB memory peak has been measured in this gate;
the physical card remains 22 GiB.

The normal installation consists of the 231,316,411,072-byte IQ1_M-plus-Engram
base, a 7,976,914,240-byte DSpark drafter and a 5,780,589-byte metadata shard:
**239,299,105,901 bytes**, before its working allowance. `pack_gguf.py` writes
pointer metadata, Engram hash tables and tokenizer files rather than copying
all weight payloads, so a second 231 GB weight copy is not part of this recipe.

The native engine supports `--dspark none`, which avoids loading the drafter;
`--dspark off` still loads it. The current IQ1_M first shard already provides
the Engram hash data used successfully above. A first backbone-only run can
therefore target the five base files, about **231.32 GB**, with drafting omitted.
That proposed recipe still requires full packing and generation validation.
Upstream's shipped setup includes DSpark by default; it was not run here.

One audited research helper, `tools/dsv41/ista_intake.py`, still has another
machine's paths and an older replacement/skip recipe. It was not executed.
Use the current `pack_gguf.py` and the plain official recipe: rebuilding the
Engram projections or skipping dense tensors is not the validated path.
The upstream planner includes speeds measured on other hardware and formats;
those numbers are not this machine's throughput, and its IQ1_M RAM class is
itself extrapolated. SATA performance and full-model quality remain open.

### Evidence, limits and next gate

The complete build, 24 native arithmetic checks, both GPU sanitizer runs,
17 Python checks, actual-header conversion and negative intake checks pass.
No upstream source patch was needed. The old core/kernels/engine archives
retain their 7e hashes and the main checkout is clean. Only this roadmap changes
in the versioned worktree; source/build/test evidence lives in the isolated
build directories. `build-deepseek-cuda/step7f-validation.json` indexes the
commands and reports; `step7f.patch` isolates this gate's documentation change.

Two audit-script issues are retained separately: the initial archive verifier
rejected Windows CRLF export conversion; its final extraction verifies the
restored bytes against the same Git blob hashes. The first negative pack lacked
expert descriptors, then the routed-only revision omitted dense-only shard 4.
The final fixture isolates all five file-size failures. No engine fix or
relaxed arithmetic threshold was used to make these checks pass.

Not run: full model loading or text generation, official-output parity,
quality, prefill/decode throughput, long-context behavior, DSpark, vision,
Windows/HIP/SYCL, or a measured whole-process memory ceiling. Existing 7d and
7e measurements remain historical controls, not a new run of this engine.

The proposed **7g** is resumable acquisition and full-file integrity checking
of the five IQ1_M/Engram base files, followed by the upstream pointer pack,
complete native pack checks and one short text-generation smoke with
`--dspark none`. Record startup allocation and CPU/GPU memory peaks before
claiming usability. Preserve the pinned 1017 identity and existing controls.
This entails about 231.32 GB of model downloads. The user confirmed this gate
and required a mirror with no proxy. Its completed acquisition, first-run failure
and validated workaround are recorded below. This is authorization for 7g only.

## Step 7g: complete mirror download and first quantized text generation

The user approved the five-file acquisition through a mirror without a proxy,
then authorized continuing the first-run investigation. All **231,316,411,072
bytes** are downloaded and have passed the upstream whole-file SHA256 checks.
The fixed quant revision remains `cb02422c900df489cea5833e8a92171224059b3c`.
Verification finished at **2026-10-26 01:17:49 Asia/Shanghai**. No DSpark, vision
or extra metadata shard was needed for this base-only run.

`tools/deepseek41/download_quantized.py` reuses the pinned upstream downloader's
range checks, eight connections, retry/resume state and SHA256 verification.
The entry point is `https://hf-mirror.com`; redirected storage CDN connections
also run directly, with proxy environment variables removed and urllib proxies
explicitly disabled. TLS checks and the pinned revision remain enabled. The
initial free disk was 412.71 GB. No existing files were removed for this gate.
The five exact hashes and the download log are indexed by
`build-deepseek-cuda/step7g-validation.json`.

The unchanged upstream packer produced **47,246 tensor descriptors in five
backing files**. Native `--pack-info` finds every file present and long enough,
no tensor without a device type, Engram layers 1 and 14, and no DSpark tensors.
The first shard supplies the integer Engram hash metadata. The plain pointer
recipe uses no tensor skips or dense overrides and makes no full weight copy.

### First startup failure and the bounded workaround

The original 7f engine stopped at `cudaHostAlloc 1262 MiB failed`. A diagnostic
interposer reproduced `cudaErrorInvalidValue (1)` for the **1,323,827,264-byte**
allocation of the 129,280 x 5,120 BF16 embedding table. More than 49 GB of host
RAM was available in the reproductions. A standalone allocation of the same
size succeeded; disabling transparent huge pages for the test process did not
resolve startup. A later default-path full run succeeded. The failure is
therefore **not deterministic**, and its kernel/driver cause is not established.

Source inspection shows that the embedding table is only read on the CPU;
rows are copied into existing pinned staging before GPU transfer. The small
patch in `tools/deepseek41/patches/pageable_embedding.patch` adds the opt-in
`DS4_PAGEABLE_EMBED=1`: keep this whole table in a normal host vector while
retaining the upstream row staging, dequantization and inference operations.
The default allocation path stays selected when the variable is absent.

The patch applies to a separate `build-deepseek-cuda/step7g-upstream` copy of
commit `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`. Only `src/dsv41/ds4_model.cu`
changes there; all **1,137 files** of the original 7f source remain unchanged.
The separate CUDA 12.8 / SM75 / portable AVX2 build succeeds and has ten SM75
ELF code objects. Its binary is **52,516,224 bytes**, SHA256
`7d1eebfd7395f3b29d46948b7eb7d0c56654db51c6af1bef334713beb0257946`, build ID
`7dd5cf18e68c639a8773e096c5c79f971ea2a9f5-consumer-pageable-embed`.
The original executable is retained as evidence and a control.

Both formal runs use this new executable without diagnostic interposition:
with the default allocation and with the opt-in. Both produce the identical IDs
`671,6102,294,8760,344,11111,16,1` and the answer:

> The capital of France is Paris.

A separate regression interposer forces the observed CUDA error at exactly the
large embedding allocation. The default path fails at the same location; the
opt-in loads and computes one input token successfully without requesting that
allocation. The interposer is test-only. Full generation in the formal run uses
no `LD_PRELOAD` substitution. These results validate a way around this allocation
dependency, not a general driver repair or a performance improvement.

### Actual short-run measurements

Configuration: **17 input tokens, at most 16 output tokens**, context 1,024,
greedy sampling, `--dspark none`, `--vision off`, 16-token windows,
`--prefill-chunk 0`, one GPU, 18 CPU threads, expert files mapped in place,
`--ssd-cache 32`, and no learned expert profile. A separate 2 GiB GPU allocation
restricts the physically 22 GiB card to a **20 GiB capacity**; the engine also
keeps `--reserve-mib 600`. The extra monitoring context consumes some of that
remaining capacity. The machine has about 64 GB system RAM and a SATA SSD.

| Measurement | Formal opt-in run |
| --- | ---: |
| Output, including EOS | 8 tokens |
| Prompt evaluation | 25.156 s (0.68 tokens/s) |
| Decode after the first prompt-derived token | 6.359 s, **1.10 tokens/s** |
| Process wall time, including load/exit | 66.07 s |
| Observed process high-water RSS | 15.93 GB |
| Observed GPU occupancy after subtracting the 2 GiB hold | 19.46 GiB |
| Observed process swap | 0 bytes |
| Minimum system available RAM | 34.37 GB |
| Decode disk reads | 368.8 MB/token |
| CPU-lane RAM-store hit fraction | 75.6% |
| RAM-store read wait | 783.4 ms/token |

The run held about 5.86 GB of dense weights and 13.31 GB of experts in VRAM
(1,740 of 15,360 experts). The configured RAM store can hold 32 GB but used only
1,920 of 3,676 slots in this short test. There were no reported store read
failures or evictions. The default-path successful control also measured 1.10
tokens/s and emitted the same IDs. File-cache state was not reset, so the wall
times do not establish a speedup. The SSD read/wait measurements justify
measuring caching and I/O in the next gate.

Memory is sampled once per second and can miss brief peaks. GPU occupancy
includes driver reservations, the monitoring context and other GPU users after
subtracting the test hold; it is not engine-exclusive VRAM. Host RAM is observed,
not hard capped. Eight output tokens are a smoke, not a sustained throughput or
quality benchmark. Longer inputs, five-prompt usability, official-output parity,
DSpark, vision and other backends remain untested by this gate.

### Reproduce and review

For a fresh source copy, apply the recorded patch and build with the same 7f
CMake flags, selecting `step7g-upstream`, `build-deepseek-consumer-sm75`, and the
candidate build ID above. The exact source diff, configure/build logs, source
hashes and architecture check are in `step7g-candidate-source.json` and
`step7g-candidate-build.json`. Re-run the short test from this worktree into a
new output directory:

```sh
python3 tools/deepseek41/smoke_quantized.py \
  --upstream-dir build-deepseek-cuda/step7f-upstream \
  --model-dir models/deepseek-v4.1-flash-1017-iq1_m \
  --exe build-deepseek-consumer-sm75/strata-dsv41 \
  --engine-sha256 7d1eebfd7395f3b29d46948b7eb7d0c56654db51c6af1bef334713beb0257946 \
  --pageable-embedding --gpu-cap-mib 20480 \
  --output-dir build-deepseek-cuda/step7g-smoke-reproduction
```

The wrapper verifies the original packer/tokenizer source and selected binary,
then explicitly enables the opt-in after clearing inherited runtime experiment
variables. Its default SHA remains the original 7f executable. The current
successful evidence is **`build-deepseek-cuda/step7g-candidate-smoke/result.json`**;
`step7g-full-smoke/result.json` intentionally preserves the original failure.
`step7g-validation.json` indexes all successful and failed runs. The 16 download
checks and nine monitoring checks from preparation remain recorded; the new
allocation regression and exact output comparison add targeted validation.

**7g is user-confirmed.** The user authorized the five-prompt **7h** baseline
with the next “continue”. Its measurements are recorded below.

## Step 7h: five real prompts on the complete quantized engine

The user confirmed 7g and authorized this bounded usability baseline on
2026-10-26. `tools/deepseek41/benchmark_quantized.py` drives Strata's existing
`--serve` stdin/stdout protocol, using the upstream tokenizer and the validated
7g binary and pack. It adds measurement code; the native inference code,
operators, defaults and model files are unchanged.

### Run and timing method

One engine stays loaded for five sequential requests. Each uses fresh KV state
(`STRATA_DS4_REUSE=0`), greedy decoding, the upstream smoke's non-thinking
prompt template and at most 64 output tokens. Expert caches remain resident,
and the OS page cache is not cleared. This records one run in the stated order,
not a controlled cold-cache benchmark or a comparison against 7g.

The profile remains context 1024, `--window 16 --prefill-chunk 0 --tp 1
--threads off`, 18 CPU threads, `--ssd-cache 32 --mmap-experts on`, no expert
profile, `--dspark none --vision off`, and `DS4_PAGEABLE_EMBED=1`. The physical
RTX 2080 Ti reports 22528 MiB; a separate allocation holds 2048 MiB to leave at
most **20 GiB**. The host is the same roughly 64 GB Xeon E5-2686 v4 machine
with the Fanxiang SATA SSD. Exact commands and hardware are recorded in
`step7h-usability/result.json` and `step7h-hardware.json` under
`build-deepseek-cuda/`.

The model becomes ready in **33.79 s**. First-token latency below begins at
each request submission, after readiness; launching this process and reaching
the first request's first token takes about **56.24 s** with the existing OS
cache state. Every output count includes EOS. The first output comes from
prefill, so decode speed is `(output tokens - 1) / native decode seconds`.
The upstream serve `STAT tok_s` includes that first output; the raw value is
retained separately and is not used as the corrected decode rate.

| Prompt | Input / output tokens | First token, s | Decode, tok/s | Request total, s | Request reads, GB | Decode reads, MB/step | VRAM expert hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chinese fact | 14 / 6 | 22.45 | 1.24 | 26.48 | 11.83 | 360.9 | 16.25% |
| Chinese explanation | 16 / 31 | 12.80 | 1.42 | 33.93 | 14.62 | 313.2 | 22.03% |
| Arithmetic | 26 / 38 | 10.29 | 1.87 | 30.13 | 13.01 | 248.1 | 28.28% |
| Ordering | 28 / 9 | 12.91 | 2.15 | 16.63 | 6.74 | 216.1 | 32.97% |
| Python function | 25 / 28 | 16.77 | 1.15 | 40.26 | 17.70 | 397.5 | 30.73% |

`Request reads` is the engine process's `/proc/<pid>/io` `read_bytes` delta,
including prefill and decode and excluding startup; GB/MB are decimal.
`Decode reads` and VRAM expert hit are the last native `STAT` for that request,
covering its decode steps. Reads total **63.91 GB** across the five requests.
The combined 107 decode steps take 72.1964 s, or **1.48 tokens/s**. Different
prompts and changing caches contribute to the variation; these numbers do not
establish an optimization gain.

### Answers and memory

All five predetermined checks pass. The actual answers and token IDs are kept
in `step7h-usability/result.json`; `answer-review.json` records the review and
functional checks.

- Chinese fact: “中国的首都是北京。”
- Explanation: coherent CPU-memory versus graphics-memory explanation, with
  37 Han characters against the requested maximum of 50. It is a simplified
  description using graphics rendering as the GPU example.
- Arithmetic: `18 × 3 = 54`, then `100 - 54 = 46`; correct change of 46 yuan.
- Ordering: “小明最高，小刚最矮。”
- Python: only a fenced function; valid syntax and seven fixed examples pass,
  including empty lists, negatives, odd-only lists, zeros and repeated values:

```python
def sum_even(nums):
    return sum(n for n in nums if n % 2 == 0)
```

All requests finish with EOS, with **6–38 output tokens each**; none reaches
the 64-token bound. These are simple sanity checks, not a quality score for
arbitrary tasks, a full 64-token endurance test or parity with original weights.

Observed engine RSS/HWM peaks at **30.42 GB (28.34 GiB)**. GPU use excluding
the 2 GiB capacity hold peaks at **19.46 GiB**; that figure includes the driver,
monitor context and other GPU users. Engine `VmSwap` stays **0**; minimum host
`MemAvailable` is **19.52 GB**. These are one-second samples plus request
boundaries on this host, not a new hard RAM limit. Startup logs show 1741 of
15360 routed experts in VRAM and a 32 GB RAM store ceiling. The process exits
normally, and its allocations are released after the run.

The serve protocol exposes VRAM hit and CPU/RAM-tier expert counts but omits
RAM-store hits, misses, evictions and read-wait time, even though native
counters already exist. Its `STAT ram=0` is a different RAM-tier counter and
**does not mean the RAM store has no hits**. The missing store hit ratio is
recorded as unavailable. The large read volume makes SSD/cache behavior the
next diagnostic target; this run alone does not separate disk wait from CPU
computation or prove which one limits throughput.

### Validation and reproduction

Six focused measurement tests pass: excluding the prefill-derived first token,
handling EOS without a decode step, rejecting inconsistent/cancelled/reused
requests, receiving split stdout lines across resident requests, reaping a
timeout and detecting early engine exit. Five actual requests complete without
errors or truncation, and seven tests of the generated pure function pass.
The binary SHA256 remains the 7g value
`7d1eebfd7395f3b29d46948b7eb7d0c56654db51c6af1bef334713beb0257946`.
All 1137 upstream source files are checked against their pinned manifest.
Existing full-weight hashes are trusted through their verified download record,
current sizes and completion markers under the shared lock; the 231 GB package
is not hashed again. No download or network service is started.

From this worktree, choose a new output directory:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --output-dir build-deepseek-cuda/step7h-reproduction \
  --gpu-cap-mib 20480
python3 -m unittest discover -s tests/deepseek41 -p test_quantized_benchmark.py -v
```

`step7h-validation.json` indexes the result, raw protocol/events, memory samples,
answer checks, unchanged-engine checks and gate-only patch review. No native
backend changes in this gate, so the existing validated CUDA binary is used
without a rebuild; this gate does not add HIP/SYCL validation.

**7h is user-confirmed.** The next “continue” authorized 7i's optional native
store diagnostics and bounded output checks, recorded below.

## Step 7i: opt-in native RAM-store diagnosis

The user confirmed 7h and authorized this gate on 2026-10-26. The new
`tools/deepseek41/patches/store_diagnostics.patch` applies after the 7g pageable
embedding patch to the pinned upstream source. It changes only
`src/dsv41/ds4_main.cpp`: `DS4_STORE_STATS=1` enables two `STORE` lines before
`DONE`, covering prefill and decode separately. The default protocol has no
new lines. Existing counters are copied and formatted once per phase, at the
request's end; no new I/O barrier or GPU synchronization is added.

`tools/deepseek41/benchmark_quantized.py` accepts an explicitly pinned candidate
binary and `--store-stats`, checks the new fields and retains the raw counts.
A requested diagnostic run fails if either phase is absent, counters are
invalid or phase token counts disagree with the stream. Without that flag,
unexpected `STORE` output also fails. The tokenizer, model pack, 20 GiB
capacity hold, streaming and memory monitor continue to use the prior path.

### What the counters mean

| Field | Meaning |
| --- | --- |
| `hits`, `misses` | Native RAM-store lookup counts, differenced over the phase. Hit ratio is `hits / (hits + misses)`; prefetch-used or already-loading accesses can be outside that denominator. |
| `submitted_bytes` | Expert payload bytes queued for demand or prefetch, before direct-I/O alignment; this is distinct from physical disk reads in `/proc/<pid>/io`. |
| `wait_ms` | Existing `ExpertStore::ensure` interval waiting for completion and finishing slots. It overlaps `cpu_lane_ms` and may overlap GPU work; these durations must not be added. |
| `evictions`, `read_fails`, `prefetched`, `prefetch_used`, `prefetch_wasted` | Existing phase-delta counters. |
| `slots`, `slots_max` | Allocated slot count and ceiling at the phase end, not the number of populated experts. These are gauges, not differences. |

Zero-denominator rates are unavailable rather than zero. Asynchronous activity
can cross a snapshot boundary; collecting the diagnostics does not pause it.
The normal run below reports no prefetch activity or failed store reads.

### Normal-profile result

The same five prompts run with fresh KV per request, at most 64 output tokens,
context 1024, the 7h windowed prefill profile, 18 CPU threads, 32 decimal GB of
RAM store, default LRU policy, DSpark none and the 7g embedding workaround.
Expert migration remains enabled. The host is the same Xeon E5-2686 v4,
roughly 64 GB RAM and Fanxiang SATA SSD; the RTX 2080 Ti is restricted to at
most 20 GiB using the same excess-capacity hold. OS caches are not cleared.

| Phase | Processed tokens | Store hit ratio | Phase wall, s | Store wait, s | Wait / wall | Submitted, GB | Evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 109 | 59.59% | 72.20 | 61.82 | 85.62% | 30.08 | 1526 |
| decode | 101 | 81.93% | 63.52 | 52.22 | 82.22% | 23.63 | 2024 |

| Prompt | First token, s | Decode, tok/s | Prefill store hit | Decode store hit | Decode wait / wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| zh_fact | 21.87 | 1.34 | 11.7% | 80.4% | 83.3% |
| zh_explanation | 11.92 | 1.62 | 60.7% | 83.6% | 81.4% |
| arithmetic | 10.25 | 1.89 | 77.0% | 84.9% | 79.5% |
| ordering | 12.07 | 2.51 | 71.8% | 88.5% | 72.9% |
| python | 16.10 | 1.25 | 62.4% | 74.7% | 86.2% |

The RAM store reaches its **3676-slot ceiling during the third prompt** and
records **3550 evictions** across the run. All store read-failure counters stay
zero. Engine-process physical reads total **57.54 GB** across the
requests, compared with the separately reported submitted payload bytes above.
Peak engine RSS/HWM is **30.43 GB**; observed GPU use excluding the hold is
**19.46 GiB**, including driver/monitor/other users. Engine `VmSwap`
stays zero, with at least **19.67 GB** host `MemAvailable` observed.

These measurements identify expert-read completion waiting as a major cost on
this profile. Even the 81.9% decode store hit ratio leaves enough misses to
spend most of decode waiting. This gate does not measure a faster cache policy
or a larger store. It supports testing the existing store budget before adding
new arithmetic kernels.

### Output controls and the historical comparison failure

The initial disabled run differed from 7h by one word in the explanation:
“图形渲染” versus “图形运算”. The planned historical-token ABBA comparison
stopped at that check; its failed report and logs remain in
`step7i-comparison.json` and `step7i-off-a/`. GPU-resident experts numbered 1740
rather than the historical 1741. That difference alone does not explain every
observed change: an unchanged-parent rerun with 1740 experts also changed the
arithmetic answer's expression while keeping the correct result of 46 yuan.
Historical output therefore cannot serve as an unconditional exact-token oracle.

The completed controls are:

- Under the normal profile, the candidate with diagnostics enabled matches the
  freshly rerun original 7g binary **token for token on all five prompts**.
  Startup residency was 1740 experts for the parent and 1741 for the enabled
  candidate, so these timings do not freeze expert placement. Their answers
  pass the same simple task checks. The Python function has the same IDs as
  7h's function, whose seven functional examples already passed.
- A separate first-three-prompt control uses the existing `--no-expert-cache`
  switch on the parent, candidate disabled and candidate enabled. All three
  runs have 1781 resident experts and identical prompt/output IDs. This isolates
  a stable expert path; it is not a selected production setting or an
  optimization result.
- Disabled runs emit no `STORE` lines; enabled runs emit exactly two per
  completed request. The phase token counts agree with prefill and actual
  decode steps. All six native runs (24 requests) complete normally within the
  same GPU bound, without engine swap or output truncation.

Mixed CPU/GPU placement and asynchronous expert changes are plausible sources
of numerical/run-to-run variation from reading the existing code. The specific
switching events have not been traced, and this gate does not claim to fix or
fully explain that existing behavior. It does not promise bitwise repeatability
for the dynamic path. The single fixed-path timing pair is also insufficient
to isolate small logging overhead; no speedup is claimed from these runs.

### Validation and reproduction

The SM75 CUDA build succeeds, with all ten CUDA ELF objects targeting SM75.
All 268 compile commands match the parent after normalizing source/build paths
and the main build-ID label. Ten measurement tests pass. A standalone C++ check
calls the actual native formatter with nonzero starting counters and with a
zero delta: counts are differenced, gauges stay absolute, and zero rates remain
unavailable. Its initial compile lacked CMake's `STRATA_VERSION` macro; the test
fixture now supplies it, and that initial compiler log is preserved.

The native main file is shared with HIP. The HIP SDK is unavailable here, so
that backend has not been built or run; this local patch is validated for the
requested CUDA/RTX 2080 Ti target. The SYCL path is not changed. Candidate
source/build provenance and exact commands are in `step7i-candidate-source.json`
and `step7i-candidate-build.json`. Parent source, binary, pack and earlier
original-weight controls are retained.

From this worktree, use a new output directory:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 \
  --output-dir build-deepseek-cuda/step7i-reproduction
```

`step7i-validation.json` indexes the build, tests, all six native runs, the
historical failed comparison, output controls, memory/I/O records and gate-only
patch review. `step7i-analysis.json` contains the actual counts and timings.

**7i is user-confirmed**, with the reproducibility and untested-backend limits
above. The next “continue” authorized 7j's 32/40 GB comparison below.

## Step 7j: compare the native RAM-store budget

The user confirmed 7i and authorized this comparison on 2026-10-26. Only the
RAM-store ceiling changes: 32 or 40 **decimal GB**, using the same validated 7i
binary, full IQ1_M/Engram base, model pack and tokenizer. This gate adds an
optional `--ssd-cache-gb 32|40` to `benchmark_quantized.py`; it forwards the
selected value to the existing native `--ssd-cache` option. Omitting the new
option preserves the prior command. No native source, operator, cache policy
or default changes.

### Method and acceptance

The run order is **32-a → 40-a → 40-b → 32-b**, with a fresh process for each
run and the same five 7h prompts inside each resident process. Every request
starts with fresh KV and allows at most 64 output tokens. Context stays 1024,
window 16, prefill chunk 0, CPU threads 18, greedy sampling, thinking off,
DSpark none, vision off and expert profile none. Dynamic expert migration and
native LRU remain enabled. `DS4_PAGEABLE_EMBED=1` and `DS4_STORE_STATS=1` apply
to all four runs. Expert caches warm across the five requests; OS caches are
not cleared. Startup time is separate from request latency and throughput.

Hardware is the Xeon E5-2686 v4, 64 GB-class host, RTX 2080 Ti and Fanxiang
S103 Pro SATA SSD already used above. The GPU reports 22 GiB; the same 2 GiB
capacity hold leaves at most **20 GiB** for the engine, driver and other CUDA
contexts. Linux 6.11.0-24, NVIDIA driver 570.144 and the existing CUDA 12.8/SM75
build are retained. This is a local bounded experiment, not the draft standard
benchmark. There are only two runs per setting, not a broad statistical sample.

Before running, acceptance required completed short answers, no truncation,
zero store read failures, no engine swap, observed GPU use within 20 GiB, and
at least **8 GiB host MemAvailable** before recommending 40 GB. The existing
native allocator also retains its 8 GiB growth floor and memory-pressure rules.
The actual store ceilings are **3,676** and **4,595** slots respectively; both
are reached. All four processes exit normally and release the GPU hold.

### Measurements

Decode rate is total decode steps divided by native decode seconds. The first
output token comes from prefill and is excluded from that numerator. Physical
reads below cover the five requests, excluding startup. GB is decimal; GPU
capacity and the acceptance floor use GiB. RSS is the process resident high
water observed in the run, not the requested store ceiling.

| Run | Decode steps | Decode tokens/s | Mean first token, s | Physical reads, GB | Store evictions | Peak RSS, GB | Minimum host available, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32-a | 96 | 1.519 | 14.46 | 59.71 | 3,382 | 30.40 | 19.71 |
| 40-a | 101 | 1.613 | 13.80 | 53.95 | 2,056 | 36.79 | 13.35 |
| 40-b | 102 | 1.639 | 13.66 | 56.33 | 1,993 | 36.81 | 13.34 |
| 32-b | 101 | 1.584 | 14.34 | 57.43 | 3,533 | 30.40 | 19.74 |

Pooling both runs per setting gives **1.551 → 1.626 tokens/s (+4.8%)**.
The two order-swapped pairs improve by 6.2% and 3.5%. Mean physical reads per
five-request run decrease from 58.57 to 55.14 GB (**5.9%**), and mean evictions
from 3,457.5 to 2,024.5 (**41.4%**). Mean first-token time changes from 14.40 to
13.73 s. These modest gains cost about **6.4 GB more peak process RAM**.

| Pooled native phase metric | 32 GB | 40 GB |
| --- | ---: | ---: |
| Prefill store hit ratio | 59.94% | 62.70% |
| Decode store hit ratio | 82.02% | 83.73% |
| Prefill read wait / phase wall time | 85.60% | 84.98% |
| Decode read wait / phase wall time | 82.52% | 81.67% |
| Decode read wait, ms/token | 531.96 | 502.30 |

The 7i counter definitions still apply: hit ratios can exclude already-loading
or prefetch-used accesses, submitted expert payload is not physical I/O, and
read wait overlaps CPU-lane/GPU work. Durations must not be added. Slots count
allocated capacity, not populated experts. All 40 phase reports have zero
failed reads and no reported prefetch activity. Read completion remains a
large part of the measured request time with either ceiling.

The improvement is not uniform. The first request has an empty RAM store and
does not benefit in this sample. These per-case values pool both runs of each
setting and retain the slower results:

| Prompt | Mean first token, 32 → 40 GB, s | Decode, 32 → 40 GB, tokens/s |
| --- | ---: | ---: |
| Chinese capital | 21.92 → 22.34 | 1.303 → 1.192 |
| Memory explanation | 12.03 → 12.32 | 1.548 → 1.470 |
| Arithmetic | 10.20 → 10.37 | 1.830 → 1.942 |
| Height ordering | 11.83 → 9.58 | 2.326 → 2.869 |
| Python function | 16.02 → 14.04 | 1.255 → 1.404 |

Startup takes 33.32–41.97 s and is not included in those figures. Initial
resident GPU expert counts are 1,740 / 1,741 / 1,741 / 1,741 in run order;
small variations in available VRAM remain visible in the logs. GPU use peaks
at about 19.46 GiB in every run. Engine `VmSwap` is zero throughout; this does
not imply that other system processes use no swap. RAM/VRAM observations use
one-second samples plus request boundaries, with process `VmHWM` retained.

### Answers, limits and next step

All **20/20 short answers** pass review and end with EOS before the output cap.
The capital, height ordering and Python function have identical output IDs in
all four runs. The explanation has two coherent variants of 30 and 37 Han
characters, both within the requested 50; these are simplified memory examples.
Arithmetic returns 46 yuan with a correct formula, but 40-b uses two arithmetic
lines instead of one. Each generated `sum_even` function passes seven fixed
examples after syntax and restricted-AST inspection, for 28 functional checks.

The complete answers, IDs and lengths are retained. Free generation therefore
does not yield identical workload traces: lengths, expert routing and cache
history can differ. For the three cases with identical output IDs in all four
runs, pooled decode improves by 9.8% and physical reads decrease by 9.7%.
That subset still follows the differing earlier requests, so it does not isolate
cache capacity from every runtime effect. The full-workload +4.8% result stays
the primary observation; neither figure establishes a universal speedup,
bitwise determinism, full-model quality or parity with original weights.

The explicit **40 GB ceiling is a reasonable candidate for the next bounded
validation on this host**, with at least 13.34 GB observed available RAM. The
existing default stays unchanged. Proposed **7k** expands to longer inputs and
128–256-token output limits while retaining context 1024, the 40 GB candidate
and the same 20 GiB GPU bound. Record actual completion lengths, answer quality,
latency, memory and cache behavior before further tuning. Longer-context scale,
endurance and broad model quality remain later work. No 7k experiment starts
before the user confirms this gate.

### Validation and reproduction

The ten existing measurement/protocol tests pass. All four native runs and all
40 store phase reports pass the configured checks. The 1,137-file original,
7g and 7i source trees match their preserved hashes, all three native binaries
and the three earlier core archives are unchanged, and the eight model-pack
files match. The 231.32 GB weights reuse the 7g full-file SHA256 records; each
run checks pinned sizes and completion markers under the shared download lock,
rather than rehashing the whole package. No new native build is needed for this
Python/docs-only gate; the prior CUDA validation is retained and no new
HIP/SYCL compatibility is claimed. The main Strata checkout remains clean.

Reproduce one run with a new output directory; change only `--ssd-cache-gb` for
the other setting. The four recorded runs use the order shown above:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 \
  --output-dir build-deepseek-cuda/step7j-reproduction-40
```

`step7j-validation.json` indexes all raw results, protocol/event/memory logs,
answer reviews, the hardware record and the gate-only patch. `step7j-analysis.json`
contains each run, pooled metrics, both order comparisons and all output-ID
groups. The per-run raw result status remains `measured_awaiting_text_review`;
its adjacent `answer-review.json` and the final validation record supply the
completed review without rewriting raw measurements.

**7j is user-confirmed.** The next “continue” authorized 7k's bounded longer
input/output checks below.

## Step 7k: bounded longer-input and longer-output checks

The user confirmed 7j and authorized this gate on 2026-10-26. The same 7i
SM75 native binary runs with an explicit **40 decimal GB RAM store** and the
same **20 GiB GPU capacity bound**. The Xeon E5-2686 v4, 64 GB-class host,
RTX 2080 Ti and Fanxiang SATA SSD are retained. Native arithmetic, LRU and
dynamic expert migration use the prior path.

`benchmark_quantized.py --suite long` adds three fixed tasks. Each declares
its own 128- or 256-token output cap. The stream guard, `DONE` count check and
prompt-plus-output context check use that cap. The default `--suite short`
retains the five prior prompts, template, token IDs and 64-token native requests.
No native source or binary changes in this gate.

### Workload and result

One resident process handles the three requests in the order below. Each
request starts with fresh KV; expert caches warm across requests and OS caches
are not cleared. Context remains 1024, window 16, prefill chunk 0, CPU threads
18, temperature 0, thinking off, DSpark none, vision off and expert profile
none. Both existing embedding and store-diagnostic opt-ins remain enabled.

Before the run, coverage required at least one input of 128 tokens or more and
one actual output longer than 128 tokens. Memory acceptance retained at least
8 GiB observed host MemAvailable, at most 20 GiB GPU use, zero engine swap and
zero native store read failures. Reaching an output cap would be reported as
length termination, not silently treated as a complete answer.

| Task | Input tokens | Output cap | Actual output, including EOS | Finish | First token, s | Decode tokens/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Fictional warehouse record extraction | 257 | 128 | 32 | EOS | 126.62 | 1.665 |
| Six-point Chinese memory explanation | 99 | 256 | 153 | EOS | 48.98 | 3.114 |
| Two Python functions with docstrings | 121 | 256 | 187 | EOS | 56.75 | 2.450 |

All three answers complete before their caps. The run processes 477 input
tokens and emits 372 outputs including three EOS tokens. Its 369 decode steps
take 143.36 native seconds, giving **2.574 tokens/s pooled decode**. Startup to
ready takes 37.02 s; the process runs for 416.67 s including startup and shutdown.
The three request wall times total 375.71 s. First-token latency is measured
from each request submission; startup is separate.

This workload differs from 7j, so the higher pooled rate is not an isolated
optimization gain over that five-question baseline. There is one measured run,
not repeated evidence of a general throughput level or bitwise determinism.

### Answers and generation behavior

- The extraction answer is a JSON object with exactly the six requested fields.
  It correctly identifies 红枫仓, 赵红 and C39. Received quantity is
  `16 + 25 + 4 = 45`, shipped is `11 + 6 = 17`, and remaining stock is 28.
- The Chinese answer contains six complete numbered explanations covering all
  requested topics. Their Han-character counts are 30, 31, 35, 29, 32 and 34,
  all within the requested 25–35. These are introductory explanations, not a
  general quality benchmark.
- Both generated Python functions have the requested one-line Chinese
  docstrings. After reading the entire answer and admitting only a small AST
  with pure builtins, seven fixed vectors per function pass: empty, mixed,
  negative, odd-only, zeros, duplicates and large integers. All **14 function
  checks** also preserve the input list. The summary keys and empty-case
  `None` values match the request.

Actual completion lengths, token IDs and full text are preserved. No request
hits its limit; this demonstrates completed 153- and 187-token responses, not
a completed 256-token response. The longest input is 257 tokens, so the test
does not establish behavior across a filled 1024-token context or beyond it.

Host-observed stream blocks provide a bounded view of sustained generation.
The first and last complete 32-step blocks change from **2.33 to 3.72 tokens/s**
for the explanation, and **1.82 to 3.01 tokens/s** for the code. Partial final
blocks are retained in the analysis rather than folded into those comparisons.
These are stream intervals, distinct from the native overall decode timer.
Expert placement, cache reuse and routing can all vary; this gate does not
isolate their individual effects or establish long-session stability.

### Memory and storage

| Observation | Measured result |
| --- | ---: |
| Peak process RSS / VmHWM | 39.05 GB / 36.37 GiB |
| Minimum host MemAvailable | 10.47 GB / 9.75 GiB |
| Peak GPU use excluding the capacity hold | 19.464 GiB |
| Engine VmSwap | 0 |
| Effective RAM-store slot ceiling | 4,595 |
| Physical process reads during the three requests | 158.07 GB |
| Prefill / decode store hit ratio | 70.87% / 89.10% |
| Prefill / decode read wait divided by phase wall time | 81.06% / 74.27% |
| Prefill / decode store evictions | 7,322 / 5,809 |
| Native store read failures | 0 |

The host still meets the preset 8 GiB availability floor, but headroom is lower
than the short-task 40 GB runs in 7j. RSS and system MemAvailable measure
different things; neither the store ceiling nor process RSS is a complete
host-memory budget. Samples are taken every second and at request boundaries,
with process HWM retained. Engine swap zero does not imply whole-system swap
zero. The separate 2 GiB GPU hold is released at exit.

The 7i counter definitions apply. Submitted expert payload is distinct from
physical reads; hit ratios can omit already-loading/prefetch-used accesses;
read wait overlaps CPU/GPU work and must not be added to those durations.
The large read traffic and read-completion intervals remain material in this
workload. The observed 49–127 s first-token waits are still slow for interactive
use even though the bounded tasks complete correctly.

### Validation, reproduction and next step

Twelve measurement/protocol tests pass, including the two new checks for
per-request 128/256 limits and EOS versus length termination. All prior short
protocol tests still pass. Static/tokenizer controls confirm unchanged short
prompts, template and IDs. All three tasks and six native phase reports pass
the configured bounds; no native crash, truncation or failed read is observed.
The three 1,137-file native source trees, all three native binaries, eight
model-pack files and three earlier core archives match their preserved hashes.
No new native build is required for this Python/docs change; prior CUDA
validation is retained without a new HIP/SYCL claim. The main checkout is clean.

From this worktree, use a new output directory:

```sh
python3 tools/deepseek41/benchmark_quantized.py --suite long \
  --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 \
  --output-dir build-deepseek-cuda/step7k-reproduction
```

`step7k-validation.json` indexes raw results, protocol/event/memory logs, answer
reviews, test logs, preservation checks and the gate-only patch. The measured
result remains immutable with its adjacent `answer-review.json` supplying the
completed text/code review. `step7k-analysis.json` retains all per-case metrics
and stream blocks. The weights reuse the 7g whole-file SHA256 verification;
current pinned sizes and completion markers are checked under the shared
lock, rather than repeating a 231.32 GB hash.

**7k is user-confirmed.** The next “continue” authorized the local text entry
in 7l below.

## Step 7l: bounded local single-turn text entry

The user confirmed 7k and authorized this gate on 2026-10-26.
`tools/deepseek41/run_quantized.py` accepts a question through `--prompt`, a
UTF-8 `--prompt-file`, or `--prompt-file -` for stdin. It streams answer text to
stdout and sends progress and completion notices to stderr. The short Chinese
[local usage guide](DEEPSEEK41_LOCAL_RUN.md) covers the tested profile and files.

From this worktree:

```sh
python3 tools/deepseek41/run_quantized.py \
  --prompt '中国的首都是哪里？请用一句话回答。'
```

The script resolves model/build assets from its own location, so an absolute
script path also works outside the worktree. The actual CUDA checks below run
from `/tmp`. Every invocation handles one question and exits; each invocation
therefore includes a fresh model load.

### Shared runtime and bounds

The entry calls the existing `benchmark_quantized` preflight and native runner.
The shared path accepts an optional single request, token callback and expected
pack hashes. It still checks the pinned source, binary, five-file verification
records and shared download lock; creates the same GPU-capacity hold; uses the
same native serve command and environment; and saves protocol, event and memory
logs. The native loader, operators, cache policy and all binaries are unchanged.

`prepare_case` shares prompt/context checks. For user input, the pinned upstream
tokenizer's ordinary-text span keeps literal control-token spellings inside the
question as text. Template controls remain outside that span. Incremental UTF-8
decoding consumes the upstream tokenizer's `token_bytes`, so characters split
across output tokens are reassembled. EOS stays in evidence and is omitted from
the displayed answer. The raw native output and EOS-free `completion_text` are
both retained in `result.json`.

The profile uses 40 **decimal GB** of RAM-store ceiling, 20 **GiB** GPU capacity,
context 1024, default 128 output tokens and a permitted output cap of 1–256.
It retains non-thinking greedy sampling, DSpark none, vision off, 18 CPU threads,
window 16, prefill chunk 0 and the existing embedding/store-diagnostic opt-ins.
Empty or over-context questions and invalid caps are rejected before model
startup. The binary and all eight pack/tokenizer file hashes are fixed to the
validated assets. No weight download or repeated 231.32 GB full-file hash occurs.

The entry checks at least 8 GiB host MemAvailable before startup and stops if
runtime samples fall below that floor, engine swap appears or GPU use exceeds
the capacity bound. RAM-store size is not a whole-process RSS hard cap. Sampling
is once per second and at request boundaries; unobserved transient peaks remain
possible. The native allocator retains its own existing memory protection.

An output-cap stop returns 0 with `finish_reason=length` and an explicit notice
that the answer may be incomplete. EOS completion returns 0 with `stop`.
Argument, input-reading and empty-question errors return 2; model preflight or
runtime errors return 1; Ctrl+C returns 130.
Timeout and interruption reuse the existing terminate/wait/kill fallback and
GPU-hold cleanup. SIGTERM is routed through the same interruption path; a real
SIGTERM run is not part of this gate's hardware evidence.

### Measured checks

| Real CLI path | Outcome | CLI exit | GPU use after exit |
| --- | --- | ---: | ---: |
| Chinese `--prompt` | `中国的首都是北京。`, EOS, six output tokens including EOS | 0 | 69 MiB |
| UTF-8 file, output cap 1 | `中国的`, `length`, incomplete-answer notice | 0 | 69 MiB |
| Stdin question, Ctrl+C after native READY/request start | `interrupted`, child terminated and reaped | 130 | 69 MiB |
| Startup timeout set to 1 s | Recorded startup timeout, child terminated and reaped | 1 | 69 MiB |

For the normal run, model startup-to-ready takes **43.89 s**, request-to-first
native output **21.84 s**, and the first stdout bytes arrive **67.19 s after CLI
launch**, including preflight and loading. Several stdout chunks arrive before
the final response; they match saved `completion_text` exactly after the normal
terminal newline. The command exits after 72.79 s. This short cold request's
decode rate is 1.201 tokens/s; it is not a same-workload optimization comparison.
The cap-1 run has zero decode steps because its only output comes from prefill.

Across these four controlled runs, peak observed process RSS is **13.38 GB**,
peak GPU use excluding the hold is about **19.46 GiB**, and engine swap is zero.
These short/cancelled runs do not reproduce 7k's broader memory load: the longer
three-task run above reached 39.05 GB RSS and 10.47 GB minimum host availability.
All four CLI and native PIDs are gone after their runs, and GPU use returns to
69 MiB. The native exits of -15 for Ctrl+C/timeout are expected controlled stops,
not unhandled model failures. The raw timeout report deliberately retains its
`failed` status; the validation record marks the expected stop check as passed.

**18 measurement/text tests pass.** They cover UTF-8/emoji token splits, EOS
suppression, the three input sources, cap notices, exact context admission,
empty/invalid input, and child reaping after callback errors, interruption and
simulated resource violations. Forced host/GPU/swap boundary violations use
fakes; no real memory-starvation or swap-inducing experiment is performed.
Real-asset preflight controls also confirm:

- All eight previous short/long prompt ID sequences remain unchanged.
- Literal user control strings keep ordinary IDs while template controls retain
  their expected counts.
- An intentionally different pack pin and an over-context question are rejected
  before CUDA allocation or native startup. Their failed raw preflight reports
  are retained as expected rejections.

The three 1,137-file native source trees, all three binaries, eight pack files
and three earlier core archives match their preserved hashes. No native build
is needed for this Python/docs gate; prior CUDA validation is retained without
a new HIP/SYCL claim. The main Strata checkout remains clean.

`step7l-validation.json` indexes the test log, preflight checks, four real CLI
runs, stdout chunk timings, raw protocol/memory records, preservation checks and
the gate-only patch. `step7l-cli-checks.json` retains every command and expected
exit. This is a bounded local entry, with the earlier quality/context and
cross-run reproducibility limits still applying.

**7l is user-confirmed.** The user accepted the recommended resident entry
in 7m below after reviewing the overall roadmap status.

## Step 7m: resident independent-question entry

The user authorized this gate on 2026-10-26. The existing local entry gains an
optional `--interactive` mode. From the worktree:

```sh
python3 tools/deepseek41/run_quantized.py --interactive
```

After the model is ready, each UTF-8 line submits one independent question.
`/exit` or EOF ends the session; Ctrl+C ends the entire session. Loaded weights
and native expert caches are reused, while every `GEN` uses fresh KV and reports
zero reused prompt tokens. Each prompt contains only its current question.
The [usage guide](DEEPSEEK41_LOCAL_RUN.md) includes the single-turn modes too.

### Implementation and bounds

The existing native runner accepts an optional request source and per-answer
finish callback. It starts one native process, waits for READY, then admits and
runs questions through the same checked protocol. Rejected over-context input
does not send GEN or reload the model. A length stop returns to the input prompt.
The output decoder and newline state reset at each answer boundary.

The Linux line reader uses readiness polling and incremental UTF-8 decoding.
While it waits for input, the runner continues memory samples and checks native
liveness. An exited engine ends the session with an error. Waiting for input is
excluded from the per-request timeout; startup and each submitted generation
retain their separate deadlines. No input-reader thread is left blocked at exit.

The profile remains 40 decimal GB RAM-store ceiling, 20 GiB GPU capacity,
context 1024, output cap 128 by default and 1–256 allowed, greedy non-thinking
sampling, DSpark none, vision off and the same pinned engine/tokenizer/weights.
The 8 GiB observed host-availability floor, engine-swap check and GPU checks
also remain active while idle. These are sampled checks, not an RSS hard cap.
Native arithmetic, LRU policy, operators and binaries do not change in this gate.

### Canonical real-CUDA checks

One process handles the following three valid questions, with a blank line and
one rejected 3000-character input between the first two. Its engine PID is
unchanged, READY appears once, and memory sampling continues over a deliberate
2.2-second input wait. `/exit` succeeds while the caller still keeps stdin open.

| Request | Input / output tokens, including output EOS when present | Finish | First token, s | Decode tokens/s |
| --- | ---: | --- | ---: | ---: |
| Chinese capital | 14 / 6 | EOS | 21.88 | 1.227 |
| Integers 1–300, one per line | 36 / 128 | Length | 21.43 | 3.466 |
| Book purchase/change arithmetic | 26 / 32 | EOS | 9.65 | 2.280 |

The capital is Beijing. The capped list is exactly the prefix **1 through 64**;
the entry reports that the answer may be incomplete. The next question correctly
returns `100 - 18 × 3 = 100 - 54 = 46` yuan. No previous question is inserted
into its input IDs. All three responses have zero reused prompt tokens and zero
store read failures; stdout matches the saved answer text and terminal newlines.

This session loads once in **43.93 s** and exits after **158.36 s**, including
loading, preflight, requests, the deliberate input wait and cleanup. First stdout
arrives 67.60 s after launch for the first cold request. Subsequent questions
reuse that loaded process; they still perform demand reads and prefill. These
are different tasks with different cache histories, so their rates are not a
controlled throughput gain or a general latency promise.

The session peaks at **36.37 GB RSS**, about **19.46 GiB GPU use** excluding the
capacity hold, and zero engine swap. Minimum observed host MemAvailable is
**13.26 GB / 12.35 GiB**. The RAM store reaches its 4,595-slot ceiling. Physical
reads during its three requests total about 47.93 GB, excluding startup.

Two additional fresh-process paths pass:

- An input ending at EOF without a final newline asks `1+1`; it returns `2`,
  then quits normally. READY appears once and the CLI exits 0.
- SIGINT while waiting for input ends the session with CLI exit 130 and native
  exit -15, reaps the child, and closes the session record. Idle resource
  samples continue before cancellation.

After each of these three canonical runs, GPU use returns to 69 MiB and all CLI
and native PIDs are gone. The later preservation snapshot observes idle GPU use
at 76 MiB; the capacity hold and model processes are absent. Sampled availability
stays above 8 GiB, GPU use stays within 20 GiB and engine swap remains zero.

### Retained failed checks and quality limits

The first hardware driver used the earlier six-point explanation as its intended
cap case, based on a previous 153-token completion. This run naturally emitted
EOS as output **128**, so it completed at the cap rather than stopping for
length. The driver's expected `stop / length / stop` assertion failed. Its three
answers, native exit 0, single READY/PID, input rejection and `/exit` are retained
in `step7m-initial-checks.json` and `step7m-resident/`; the driver and traceback
are also preserved. Only the check workload changed to the explicit long list.
No product or native code change was needed for that correction.

The retained explanation covers its six topics, but item 4 has **23 Han
characters** against the requested 25–35. Its item counts are 27, 27, 28, 23,
27 and 29. This is recorded as an instruction-following quality limit. Its
wording and length variation also reinforces 7i's existing reproducibility
limit. The final answer review checks six natural answers across the retained
and canonical runs for content; it does not claim that every strict format
constraint passes or establish general model quality.

An offline analysis helper also initially computed expected IDs with the
worktree tokenizer because its import order replaced the pinned tools path.
The failed helper/log are retained. It now follows the verified preflight/CLI
import order and asserts the actual pinned tokenizer module path. That correction
is confined to the offline helper; runtime source and native results are unchanged.
Details are in `step7m-analysis-import-check.json`.

### Validation and next step

**24 measurement/text/session tests pass**, including all previous 18. New
checks cover UTF-8 across input reads, blank input, `/exit`, final-line EOF,
per-answer decoder/newline reset, rejection/cap recovery, idle native exit,
and input waiting longer than the request timeout while sampling continues.
The idle-timeout and simulated-exit checks use fake engines; the actual CUDA
waits are short. There is no endurance or full-context claim.

The eight previous prompt-ID controls and two expected preflight rejections
still pass. Native source trees, all three binaries, model-pack files and core
archives match the preserved hashes. The 99 earlier runtime/header files and
main checkout remain unchanged. No new native build is required for this
Python/docs gate; prior CUDA validation is retained without a new HIP/SYCL claim.

`step7m-validation.json` indexes canonical runs, the retained initial run,
failed-check diagnoses, test logs, answer review, memory/event/stdout records,
preservation checks and the gate-only patch. `step7m-analysis.json` retains
per-question metrics and the unchanged native command/environment controls.

**7m is user-confirmed.** The next “continue” authorized the bounded native
LRU/LFU comparison in 7n below.

## Step 7n: native RAM-store LRU/LFU comparison

The user confirmed 7m and authorized this gate on 2026-10-26. The same 7i
SM75 binary, quantized weights and pack run with 40 decimal GB RAM-store ceiling
and a 20 GiB GPU capacity bound. Only the native `--ssd-policy` value changes.
The benchmark gains an optional `--ssd-policy lru|lfu` forwarding option and
records the selected policy. Omitting it preserves the prior native command;
the local single-turn/resident entry continues to use the pinned engine's LRU.
No native source, operator or binary changes.

Strata's native LRU evicts by oldest touch, and LFU by the lowest decayed hit
count, among unpinned slots not visible to GPU reads. The existing LFU settings
retain heat decay 0.97 every 240 accesses of a layer. This tests that pinned
implementation, not every possible frequency policy.

### Method and acceptance

The order is **LRU-a → LFU-a → LFU-b → LRU-b**. Each fresh native process handles
the same five 7h short prompts in the same order, with fresh KV per request and
up to 64 output tokens. Expert caches warm within each process; OS caches are
not cleared. Context 1024, window 16, prefill chunk 0, CPU threads 18, temperature
0, thinking off, DSpark none, vision off, expert profile none and dynamic expert
migration are held fixed. Hardware remains the Xeon E5-2686 v4, 64 GB-class host,
RTX 2080 Ti and Fanxiang SATA SSD, with the same separate GPU-capacity hold.

Native startup logs confirm policy **0 / 1 / 1 / 0**, eight read threads and
**4,595 slots** in each run. All four start with 1,741 GPU-resident experts.
Commands, environment, engine/pack hashes and input IDs match after normalizing
only the policy value. There are only two runs per policy; this is a bounded
local experiment, not the draft standard benchmark or a statistical superiority
claim. Acceptance also requires reviewed answers, no read errors/engine swap,
GPU at most 20 GiB and at least 8 GiB observed host MemAvailable. No default
switch is made from a small or inconsistent result.

### Measurements

Decode rate is total decode steps divided by native decode seconds. The first
output comes from prefill and is excluded from those steps. Physical reads
cover the five requests, excluding model startup. GB is decimal; GPU capacity
and the availability floor use GiB.

| Run | Decode steps | Decode tokens/s | Mean first token, s | Physical reads, GB | Store evictions | Peak RSS, GB | Minimum host available, GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRU-a | 107 | 1.649 | 13.48 | 57.33 | 2,108 | 36.84 | 12.56 |
| LFU-a | 107 | 1.617 | 13.54 | 57.30 | 2,346 | 36.72 | 12.97 |
| LFU-b | 107 | 1.591 | 13.50 | 58.83 | 2,356 | 36.73 | 13.27 |
| LRU-b | 101 | 1.647 | 13.48 | 54.69 | 2,087 | 36.85 | 12.85 |

| Pooled observation | LRU | LFU |
| --- | ---: | ---: |
| Decode tokens/s | 1.648 | 1.604 |
| Mean first-token latency, s | 13.478 | 13.516 |
| Mean physical reads per five-request run, GB | 56.01 | 58.07 |
| Mean store evictions per run | 2,097.5 | 2,351.0 |
| Prefill store hit ratio | 62.72% | 62.43% |
| Decode store hit ratio | 83.65% | 82.75% |
| Prefill read wait / phase wall time | 84.73% | 84.83% |
| Decode read wait / phase wall time | 81.24% | 82.19% |
| Decode read wait, ms/token | 492.85 | 512.48 |

LFU's pooled decode rate is **2.7% lower**, with the two order-swapped pairs
lower by 2.0% and 3.4%. Mean physical reads are **3.7% higher**, evictions
**12.1% higher**, and read-wait duration per decode token about **4.0% higher**.
The first-token means are effectively similar at this sample size. These
observations give no reason to switch the tested profile to LFU. They do not
prove that LRU is better for every workload or isolate the runtime cause of
every timing difference.

The result is not uniform across questions. Pooled per-case decode rates retain
the cases in which LFU is faster:

| Prompt | Mean first token, LRU → LFU, s | Decode, LRU → LFU, tokens/s |
| --- | ---: | ---: |
| Chinese capital | 21.92 → 21.92 | 1.310 → 1.261 |
| Memory explanation | 11.93 → 12.01 | 1.564 → 1.600 |
| Arithmetic | 9.91 → 9.86 | 1.953 → 2.052 |
| Height ordering | 9.57 → 9.66 | 2.573 → 2.686 |
| Python function | 14.07 → 14.13 | 1.379 → 1.174 |

All 40 phase reports have zero read failures and no reported prefetch activity.
All runs reach the 4,595-slot ceiling. GPU use peaks around 19.46 GiB, engine
VmSwap is zero, and minimum observed host availability is about 12.56 GB.
These are one-second/boundary observations with process HWM, not a whole-process
RAM hard cap. System processes can still use swap. Startup takes 33.09–43.15 s
and remains separate from request timings. All test processes exit and the
capacity hold is released; GPU use returns to 69 MiB in the preservation check.

The 7i counter definitions apply: hits can exclude in-flight/prefetch-used
accesses, submitted payload differs from physical I/O, and read wait overlaps
CPU-lane/GPU work. The durations must not be added. Slot count is allocated
capacity, not the number of populated experts.

### Answers, variability and conclusion

All **20 short-task answers** pass review and finish with EOS before their
64-token caps. The four generated `sum_even` functions pass seven fixed examples
each, for 28 functional checks after restricted-AST inspection. This is the
same small sanity workload, not general quality or original-weight parity.

Output IDs for the capital, ordering and code are identical across all four
runs. LFU-a changes one explanation token from graphics rendering to graphics
computation; both explanations remain coherent and within 50 Han characters.
Arithmetic uses two correct formula lines in three runs and one correct formula
line in LRU-b, changing output length from 38 to 32 tokens. Full text, IDs and
lengths are retained. Free generation and cache histories therefore differ.

On the three cases with identical outputs, LFU's pooled decode is 11.5% lower
and physical reads 5.3% higher. This subset still inherits the differing earlier
cache histories. It does not remove every confound, and the full-workload
**-2.7%** remains the headline observation. Both policies also have ordinary
run-to-run timing variation. There is no universal speedup or bitwise-determinism
claim from these four runs.

**Retain the existing LRU profile.** The current local entry remains available
with its tested memory limits and resident mode. LFU is not selected as the next
candidate on this evidence.

### Validation, reproduction and next step

The 24 existing measurement/text/session tests pass. Eight prior prompt-ID
controls and two expected preflight rejections also pass. The three 1,137-file
native source trees, three native binaries, eight pack files and three earlier
core archives match their preserved hashes. The 99 original runtime/header
files and main checkout remain unchanged. No native build is needed for this
Python/docs-only gate; prior CUDA validation is retained without a new HIP/SYCL
claim. Weight checks reuse the full-file verification record and current pinned
sizes/markers under the shared lock, without rehashing all 231.32 GB.

Reproduce one policy run with a new output directory; change only
`--ssd-policy` for the other policy:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 --ssd-policy lfu \
  --output-dir build-deepseek-cuda/step7n-reproduction-lfu
```

`step7n-validation.json` indexes raw protocol/events/memory, answers, tests,
hardware, preservation checks and the gate-only patch. `step7n-analysis.json`
retains each run, both order comparisons, pooled/per-case data and output-ID
groups. Per-run raw measurements remain immutable with adjacent answer reviews.

**7n is user-confirmed.** LRU remains the working configuration.

## Step 7o: native chunked-prefill audit and bounded comparison

**Result: keep `--prefill-chunk 0`.** Strata's existing 256-token configuration
runs on the SM75 GPU within the memory bounds and passes the twelve bounded task
checks, but full first-token latency increases and decode slows in both run
orders. No native operators, model weights or production defaults changed.

The opt-in benchmark `--prefill-chunk 0|256` selects the existing native switch,
enables `DS4_FLOW_STATS` in both arms and applies the existing live resource guard
to both. Omitting the new option preserves the prior native command, environment
and guard behavior; the production text entry remains byte-identical.

### Scope and implementation audit

Measured on 2026-10-26 with the same pinned diagnostic engine
`86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4`
and GSQ IQ1_M/Engram pack: E5-2686 v4 with 18 CPU workers, 64 GB-class RAM,
RTX 2080 Ti (physical 22 GiB, bounded to 20 GiB by the existing capacity hold),
SATA SSD, CUDA 12.8. The native RAM store stays at 40 decimal GB, LRU,
4595 slots and eight readers. Context stays 1024, DSpark none, vision off,
`--threads off`, `--tp 1`, temperature zero and profile none.

Source inspection followed dense FP16 reconstruction/cuBLAS FP32 accumulation,
mHC SGEMM and the existing low-bit light/SM70+ FP16 WMMA expert kernels. No native
BF16 or FP4 Tensor Core path was introduced. Positive `--prefill-chunk` enables
native prompt threading even though decode threading is off. The extra chunk
buffers and staging ring are allocated before resident experts: both candidate
runs start with **1653** experts in VRAM versus **1741** in both controls.

The same three 7k tasks have 257/99/121 input tokens and 128/256/256 output caps.
Each of four fresh processes handles them in that order with fresh KV per
request. The process order is windows-a, chunk-a, chunk-b, windows-b. No OS-cache
drop or forced path selection is used. Native logs confirm all six candidate
prompts use the flow. Both 257-token runs use layer-major reuse over balanced
129/128-token chunks and report a 3.0 GB layer cache. The default YOCO skip cannot
activate in this bounded context.

### Measured results

| Run | Startup, s | First-token seconds: extraction / explanation / code | Pooled decode, tok/s | Whole-request reads, GB | Output tokens including EOS |
| --- | ---: | --- | ---: | ---: | --- |
| windows-a | 40.72 | 124.86, 48.29, 55.47 | 2.613 | 157.39 | 32, 144, 192 |
| chunk-a | 43.49 | 140.90, 58.34, 83.12 | 2.152 | 200.91 | 32, 143, 187 |
| chunk-b | 43.20 | 135.57, 55.96, 79.19 | 2.230 | 200.77 | 32, 143, 187 |
| windows-b | 43.83 | 124.58, 48.36, 55.98 | 2.565 | 157.30 | 32, 149, 187 |

| Metric, two runs per configuration | Windows (`0`) | Native chunk policy (`256`) | Candidate change |
| --- | ---: | ---: | ---: |
| Mean first-token latency, s | 76.26 | 92.18 | +20.9% |
| Pooled decode, tok/s | 2.589 | 2.190 | -15.4% |
| Physical process reads per three-request run, GB | 157.35 | 200.84 | +27.6% |

Both order pairs agree on slower full prefill and decode. These are two trials
per configuration, not a statistical or universal claim. All four extraction
outputs have identical token IDs; explanation/code outputs can differ. Actual
lengths and per-case measurements remain in the raw files. Decode rate uses
`(generated tokens including EOS - 1) / native decode seconds`. Whole-request
latency and physical reads include both phases and can depend on output length.

### Why the stream timer is misleading here

In chunk-a, the 257-token native stream takes **96.199 s**, but complete prefill
takes **140.902 s**; the first control takes **124.857 s** for complete prefill.
The stream reuses 3384 expert reads from its layer cache, then the prefill STORE
snapshot records another **17.520 GB** of submitted payload. The two configurations
produce exactly the same extraction answer, but whole-request process reads are
75.51 GB for chunk-a versus 58.44 GB for windows-a.

Source inspection places the stream log before `cache_tick()` and
`cache_settle()`. The latter first spends up to its 6000 ms swap budget moving
hot experts into VRAM, then synchronously preloads hot CPU-lane experts into RAM.
That RAM preload is outside the swap time budget. Flow reads consult existing
RAM-store entries but do not automatically retain every streamed expert there.
This supports redundant reading/cache handoff as the next hypothesis to test.
The 44.703 s difference has **not** been exclusively attributed to preload:
there is no separate preload/swap timing or on/off isolation in this gate.

STORE hit/miss/wait counters omit flow I/O. In particular, a zero prefill STORE
wait is compatible with substantial native flow read waits. Queued payload bytes,
native stream bytes and `/proc/<pid>/io` physical reads have different meanings.
The raw diagnostics are retained rather than added together as disjoint costs.

### Validation and limits

All **12** answers pass the task/content checks and requested formatting; all end
at EOS before their caps. The six numbered Chinese points each satisfy 25–35 Han
characters. The four Python outputs pass **56** functional examples, including
empty/odd-only/negative/duplicate/zero/large-integer inputs and input preservation.
The GPU-memory explanation may mention gradients as a general use; this inference
engine does not need gradient storage.

All **24** existing benchmark/text/session tests pass, as do eight prior prompt-ID
controls, two pack/context rejection checks and five no-GPU option/resource checks.
The latter verify the unchanged default path, guard forwarding for 0/256 and
low-memory rejection before GPU allocation. Mock checks are separate from the
four actual native process runs.

Peak observed GPU use excluding the capacity hold is
**19.504 GiB**; peak process RSS is
**39.18 GB**; minimum host MemAvailable is
**10.40 GB**. Engine swap and STORE read failures
are zero. Memory observations use one-second samples plus request boundaries and
process HWM, not an instantaneous peak guarantee. The system's pre-existing swap
usage by other processes is not claimed to be zero.

The gate-only patch changes the benchmark and this roadmap. Existing native
sources, binaries, core archives, pack and production entry are preserved. No
new native build is needed; no new HIP/SYCL validation is claimed. This gate does
not establish long-context, YOCO, DSpark, endurance or original-weight parity.

Reproduce one arm in a new output directory; change only `--prefill-chunk` for
the other arm:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --suite long --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 --ssd-policy lru \
  --prefill-chunk 256 --output-dir build-deepseek-cuda/step7o-reproduction-chunk
```

`step7o-validation.json` indexes the audit, four raw runs, answer reviews, tests,
hardware, preservation checks and gate-only patch. `step7o-analysis.json` retains
pooled/per-case/both-order comparisons, output-ID groups and stream/full-prefill
differences. Raw measurements remain immutable with adjacent answer reviews.

**7o is user-confirmed.** The measured 256-token candidate failed the performance
criterion; the production profile continues to use windows.

## Step 7p: isolate native post-prompt RAM preload

**Result: disabling preload is a tradeoff, not a production-default change.**
Both run orders show lower first-token delay, whole-request time and physical reads with preload off, but slower first-eight-token and overall decode throughput.

On 2026-10-26, four fresh processes ran in on-a, off-a, off-b, on-b order on the
same SM75 engine, GSQ IQ1_M/Engram pack and hardware as 7o. Each handled the same
three 257/99/121-token inputs with output caps 128/256/256 and fresh KV. Both arms
use chunk 256, 40 decimal GB LRU (4595 slots/eight readers), 20 GiB effective GPU,
context 1024, DSpark none, vision off and 18 CPU workers. OS caches are not cleared.
Only `DS4_SETTLE_STORE=1/0` differs between arms. Both keep the default 6000 ms VRAM
settling budget and enable native verbose/flow diagnostics.

The new optional benchmark `--settle-store on|off` requires `--prefill-chunk 256`,
forwards that existing native environment switch and enables the native timers.
The old default/window/chunk command paths, production entry and operators remain
unchanged. Inherited DS4 variables are still stripped before explicit overrides.

### Results

| Run | First-token seconds: extraction / explanation / code | Pooled decode, tok/s | Three requests, s | Physical reads, GB | Output tokens including EOS |
| --- | --- | ---: | ---: | ---: | --- |
| on-a | 136.05, 55.87, 78.81 | 2.308 | 430.61 | 200.29 | 32, 153, 187 |
| off-a | 99.54, 53.04, 42.58 | 1.838 | 393.15 | 180.94 | 32, 143, 192 |
| off-b | 99.46, 52.48, 42.33 | 1.827 | 393.51 | 180.82 | 32, 148, 187 |
| on-b | 135.63, 55.88, 77.78 | 2.219 | 428.79 | 198.89 | 32, 138, 187 |

| Metric, two runs per setting | Preload on | Preload off | Off versus on |
| --- | ---: | ---: | ---: |
| Mean first-token latency, s | 90.00 | 64.90 | -27.9% |
| Pooled decode, tok/s | 2.264 | 1.833 | -19.0% |
| First eight decode steps, pooled stream tok/s | 1.143 | 0.618 | -45.9% |
| Mean time for three requests, s | 429.70 | 393.33 | -8.5% |
| Physical reads per three-request run, GB | 199.59 | 180.88 | -9.4% |

All twelve prompts actually use native flow. Initial GPU residency is 1653 experts
in every run. The six enabled preloads take 13.937–37.317 s each;
no post-prompt RAM preload executes in either disabled run. The raw native timer
and count are stored per request. STORE prefill submissions in disabled runs are
0 bytes, while flow I/O still occurs.

The extraction task is a useful fixed-output control: all four runs produce the
same 32 output token IDs. In pair a, turning preload off changes TTFT from
136.05 to 99.54 s, but decode from 1.309 to 0.776 tok/s; the first eight decode steps
stretch from 8.663 to 17.061 s. Whole-request time falls from 159.73 to 139.46 s and
physical reads from 75.51 to 65.56 GB. Thus some proactive reading is avoided, while
other reads move into generation. On the following explanation prompt, disabled
preload also leaves fewer RAM hits for flow: its first run reads about 23.3 GB of
expert payload from files versus 17.7 GB with preload. Cache effects persist across
requests even though KV is fresh.

### Interpretation

Native `cache_settle()` first settles GPU expert placement, then, when enabled,
synchronously calls `ExpertStore::preload()`. Its native log directly times that
RAM call. In on-a's extraction task it takes **35.235 s**; total cache settling is
**41.236 s**, versus about **6.001 s** with RAM preload off. This isolates the extra
RAM-preload work suspected in 7o without changing native code. The 6000 ms GPU
settling budget does not bound the subsequent RAM preload.

The switch therefore exchanges first-token delay for later demand reads; it does
not solve retaining streamed experts for decode. Keep the production windows profile. The preload-off chunk setting is a measured latency/throughput tradeoff; prioritize reuse of already streamed hot-expert bytes in the existing RAM cache to reduce duplicate reads.
The production windows configuration is not an arm in 7p; 7o's measurements are
context, not a paired claim that either 7p setting beats the production profile.

Free-generation explanation/code lengths and IDs vary. Decode uses actual
`(output tokens including EOS - 1) / native decode seconds`; early decode uses
host-observed intervals after the first output. Whole-request time and reads also
depend on output length. STORE payload, flow bytes and process physical reads are
different counters; no disjoint sum of those counters or overlapping waits is
claimed. Two runs per setting establish a bounded comparison, not statistical
superiority, general quality or long-context performance.

### Validation

All 12 answers pass the bounded content/function checks, and all finish at EOS
before their caps. The four Python answers pass **56** functional examples with
input preservation. **11/12** answers meet the requested format checks.
Explanation lengths are retained below; content success does not erase format
failures. All four prespecified runs are retained in the comparison.

| Run | Han characters in numbered points 1–6 | Requested per-item length |
| --- | --- | --- |
| on-a | 29, 32, 36, 31, 35, 38 | **Fail: outside 25–35 Han characters** |
| off-a | 30, 29, 30, 26, 30, 33 | Pass |
| off-b | 29, 28, 34, 29, 35, 33 | Pass |
| on-b | 29, 31, 31, 26, 26, 29 | Pass |

The **24** existing tests, eight prior prompt-ID controls, two pack/context
rejections and ten no-GPU option/environment/resource checks pass. The latter
cover unchanged defaults, explicit env isolation, native timer selection,
low-memory rejection and invalid option combinations. Mock checks are separate
from actual engine execution.

Peak observed GPU use excluding the capacity hold is
**19.511 GiB**; peak process RSS is
**39.16 GB**; minimum MemAvailable is
**10.13 GB**. Engine swap and STORE read failures are
zero. Sampling is once per second plus boundaries, with process HWM retained;
this is not an instantaneous peak or zero system-wide swap claim.

Only the benchmark and roadmap change. Existing native sources/binaries, core
archives, pack and production entry are preserved. No native rebuild or new
HIP/SYCL validation is claimed. Reproduction, in a new output directory:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --suite long --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-diagnostics-sm75/strata-dsv41 \
  --engine-sha256 86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4 \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 --ssd-policy lru \
  --prefill-chunk 256 --settle-store off \
  --output-dir build-deepseek-cuda/step7p-reproduction-off
```

Use `--settle-store on` for the control. `step7p-validation.json` indexes actual
runs, reviews including format failures, tests, hardware, source audit,
preservation and the independent patch. `step7p-analysis.json` retains per-case,
per-order and pooled timing/read/counter results.

**7p is user-confirmed.** The subsequent “继续” authorized the bounded 7q native handoff prototype below. The production window profile remains unchanged.

## Step 7q: retain completed flow reads in the native RAM store

**7q is user-confirmed; the local opt-in prototype is retained.** The new
`DS4_FLOW_STORE=1` path copies already streamed hot experts into free slots in
Strata's existing `ExpertStore`. It reuses the existing arena, quantized layout,
reader, demand/decode path, LRU policy and operators. It adds no attention or MoE
operator, new cache, weight format or production default. The production entry
continues to use the confirmed window profile and previous pinned binary.

### Scope and ownership

The reviewable `tools/deepseek41/patches/flow_store_handoff.patch` applies after
7i's patches, changing only `src/dsv41/ds4_store.hpp` and the host orchestration in
`src/dsv41/ds4_model.cu`. Historical source trees and binaries are preserved;
`build-deepseek-cuda/step7q-upstream` and
`build-deepseek-flow-store-sm75/strata-dsv41` are separate candidates.

Admission is considered after actual routing, only for picked experts with
native cumulative heat at least 4. Already-DMAd speculative items are skipped:
their staging bytes may already have been reused. For other eligible items,
`flow_pump` acquire-checks all three completed reads and the failure flag, queues
its existing DMA, then copies exactly the three matrix payloads before advancing
`n_dma`. The DMA only reads the same source. Staging cannot be reused until both
that counter and the existing DMA completion event permit it.

`retain_free` synchronously owns those bytes before publishing a ready location.
It never overwrites an existing/loading/pinned/visible slot and never calls the
evicting allocator or submits I/O. A full store skips admission. Arena capacity
and growth still use the existing budget and floor; admission also checks room
for one slot before faulting previously committed free pages. Mapped arenas
cannot grow through this API, avoiding CUDA host registration during kernels.
The native integration is limited to single GPU/single member, pageable store,
no copy-mode. Metadata reads/writes stay serialized on that worker; background
cache movers read model files. This API does not make upstream's plain location
table safe for arbitrary concurrent put/find callers.

The benchmark's optional `--flow-store on|off` requires an explicitly pinned
engine and `--prefill-chunk 256`; inherited DS4 overrides remain isolated.
Counters report retained payload bytes, duplicates, no-room skips, speculative
early skips, invalid spans and host copy/API time. Copied bytes are not new disk
reads, and copy time can overlap GPU work.

### Same-day bounded model comparison

On 2026-10-26, the unchanged 257-token `long_retrieval` fixture ran with a 128-token
output cap, one fresh native process per run: parent, off-a, on-a, on-b, off-b.
The parent is the existing 7i binary. Candidate off/on differ only in the new
switch; all keep RAM preload **on**, the native 6000 ms GPU settle budget, 40
decimal GB LRU/4595 slots, context 1024, 18 CPU workers, 20 GiB capacity target,
DSpark none, vision off and the same GSQ IQ1_M/Engram pack. OS caches were not
dropped. This gate tests the first request while RAM cache slots are available.
TTFT and request wall time start at GEN submission after READY; separate
model startup takes **40.94–43.45 s**,
excluded from those request timings.

| Run | First token s | Decode token/s | Request s | Physical reads GB | Post-prompt preload: experts / s | Retained experts | Initial VRAM experts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent | 135.689 | 1.2966 | 159.596 | 75.813 | 2327 / 35.403 | 0 | 1652 |
| off-a | 127.734 | 1.2947 | 151.677 | 71.745 | 2110 / 32.203 | 0 | 1653 |
| on-a | 97.317 | 1.4105 | 119.294 | 55.715 | 49 / 0.842 | 2916 | 1653 |
| on-b | 102.817 | 1.3028 | 126.611 | 57.408 | 51 / 0.857 | 2916 | 1653 |
| off-b | 132.356 | 1.3625 | 155.108 | 72.515 | 2290 / 34.805 | 0 | 1653 |

Candidate off/on means are **130.045 / 100.067 s** to
first token (-23.05%), **72.130 /
56.562 GB** physical reads
(-21.58%), and **153.392 /
122.952 s** per request
(-19.84%). Pooled decode is
**1.3277 / 1.3545
token/s** (+2.02%). The first eight decode
intervals average **8.664 /
8.292 s**. Per-order results and all raw output IDs are
in `step7q-analysis.json`. Both run orders reduce TTFT and physical reads. Decode
rate changes are **+8.94% /
-4.38%** in the two pairs, so a stable
decode-throughput gain is **not established**. Peak RSS increases from
**28.47 GB** with the feature off to
**32.51 GB** with it on, within the same store budget.

The two enabled runs retain **2916 / 2916** experts,
copy **21.715 / 21.715 GB**, and spend
**24.424 / 24.157 s** inside the copy/API
calls. Their early-speculation skips are **1697 / 1697**;
no-room skips are **0 / 0**. The preserved
preload still fills missing hot experts. No global hottest-set ranking or
full-cache replacement improvement is implemented.

### Validation and limits

All five native requests return the correct six-field JSON, stop at EOS after
32 output tokens including EOS, and pass the resource/read-failure checks.
All five output-ID sequences match the same-day parent:
**true**. This is evidence for this fixed task;
it does not establish general generation quality, full-logit equivalence,
long-context performance, repeated-turn benefit or unsupported backends.

The real native-header test passes ASan/UBSan: unequal/unaligned matrix payloads,
invalid inputs, independent byte ownership after staging mutation, zero-read
subsequent `ensure`, duplicate/in-flight preservation, pinned/borrowed/visible
slots, full budget, free-page floor, chunk growth and mapped-growth refusal.
A test-only linker wrapper pauses an actual `pread` to verify incomplete data is
not published or replaced. The initial test draft had incorrect argument counts
for existing `release`/`prefetch`; corrected before the passing sanitizer run.
The 24 existing Python tests, 15 no-GPU option/env/resource checks, eight prior
prompt-ID controls and two pack/context rejection checks pass.

The initial comparison controller stopped **after a successful parent run**
because it expected the previous gate's 1653 initial VRAM experts. Before that
unchanged binary even started, available CUDA memory was 7,405,568 bytes lower;
native placement uses current available memory and selected 1652 experts. The
cause of that ambient memory difference is not established. The next completed candidate-off
run had that memory available again and placed 1653 experts, triggering a second
strict controller check. Both native runs succeeded and their 32 output IDs
match. Both failed controllers and completed raw runs are preserved and reused.
Before any enabled measurement, the controller was corrected to record placement
and accept only the already observed 1652–1653 range; all resource limits remain
unchanged. The four candidate runs' actual resident counts are
**[1653]**. Capacity variation, if any, remains a
comparison limit; the ambient-memory source is not established. See
`step7q-controller-rca.json`, `step7q-comparison-initial.*` and
`step7q-comparison-capacity-retry.*`.

Across the five runs, peak sampled GPU use excluding the capacity hold is
**19.505 GiB**,
peak RSS is **32.51 GB**, and minimum
MemAvailable is **17.48
GB**. Engine swap and store read failures are zero. Measurements use one-second
samples plus boundaries and process HWM; this is not an instantaneous-peak or
system-wide zero-swap claim. Two runs per setting and one fixed request do not
justify promoting chunked prefill over the production window profile.

CUDA 12.8 SM75 Release/portable AVX2 built successfully; all ten cubins target
SM75. Candidate SHA256:
`2a8220ef222451ca6cdc58ba83d2fb90b0cc43d476faf5668703c8b2d87bf07b`.
HIP/SYCL compilers are unavailable locally, so those builds and executions are
**not run**; this remains a local prototype, not a cross-backend-ready change.

Reproduce the measured single-case wrapper in a new output directory:

```sh
python3 build-deepseek-cuda/run-step7q-case.py \
  --suite long --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-flow-store-sm75/strata-dsv41 \
  --engine-sha256 2a8220ef222451ca6cdc58ba83d2fb90b0cc43d476faf5668703c8b2d87bf07b \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 --ssd-policy lru \
  --prefill-chunk 256 --settle-store on --flow-store on \
  --output-dir build-deepseek-cuda/step7q-reproduction-on
```

`--flow-store off` is the candidate control. The normal benchmark entry can run
its full suite; that broader comparison is not part of the results above.
`step7q-validation.json` indexes source/build pins, ownership audit, tests,
controller failure, native runs, analysis, preservation checks and gate patch.

**7q is user-confirmed.** The subsequent “继续” authorized the bounded resident-request validation in 7r below.

## Step 7r: resident requests and cache-capacity behavior

**Result: benefits are concentrated before the RAM store fills; full-cache flow admission stops** This gate validates the existing 7q native binary
without changing operators, cache policy or production defaults.

The new benchmark `--suite resident` runs the unchanged three long fixtures,
then repeats the original retrieval question: input IDs 257/99/121/257, output
caps 128/256/256/128. The repeat has a distinct report ID but exactly the original
prompt IDs and expected answer. All four questions use one native process;
weights, routing profile and expert cache persist. KV/prefix reuse is disabled,
so this tests independent requests with a persistent expert cache, not retained
conversation context.

On 2026-10-26, the four fresh resident processes ran off-a, on-a, on-b, off-b.
Both arms use pinned 7q binary
`2a8220ef222451ca6cdc58ba83d2fb90b0cc43d476faf5668703c8b2d87bf07b`, the same
GSQ IQ1_M/Engram pack, 40 decimal GB LRU/4595 slots, 20 GiB GPU target, context
1024, chunk256, DSpark none, 18 CPU workers, RAM preload on and native GPU
settling. Only `DS4_FLOW_STORE=0/1` differs. Native layer-cache sizing stays
automatic. No OS cache drop or selected reruns are used. Each measured sequence
has one READY, four DONE records, zero reused KV tokens and a single engine PID.

### Measurements

TTFT and request wall time start at GEN after READY; model startup is separate.
Output lengths include EOS. Explanation/code lengths may vary, so their task
time and total reads must be interpreted together with content/format and
actual decode steps, rather than treating unequal outputs as identical work.

| Run | Sum of four request times s | Physical reads GB | Output lengths, requests 1–4 | Initial VRAM experts |
| --- | ---: | ---: | --- | ---: |
| off-a | 523.74 | 241.14 | 32, 142, 187, 32 | 1653 |
| on-a | 499.80 | 227.66 | 32, 156, 193, 32 | 1653 |
| on-b | 500.16 | 226.75 | 32, 159, 188, 32 | 1653 |
| off-b | 554.92 | 248.62 | 32, 153, 187, 32 | 1653 |

The next table gives **off / on** means; decode rates pool actual decode steps
and native decode time. Per-order results, startup times, all token-ID groups
and phase counters are in `step7r-analysis.json`.

| Task | First token s | Physical reads GB | Decode token/s |
| --- | ---: | ---: | ---: |
| long_retrieval | 131.05 / 104.31 | 72.79 / 57.61 | 1.303 / 1.390 |
| long_explanation | 62.69 / 43.42 | 53.75 / 49.25 | 2.459 / 2.556 |
| long_python | 78.03 / 81.34 | 68.83 / 71.09 | 2.385 / 2.346 |
| long_retrieval_repeat | 92.01 / 92.19 | 49.51 / 49.25 | 2.184 / 2.156 |

The first request's mean TTFT falls from 131.047 to 104.311 s (-20.40%), with 20.85% fewer physical reads. The four-request sequence averages 539.332 / 499.982 s (off/on), a 7.30% reduction, and 244.881 / 227.205 GB physical reads (-7.22%). Both run orders reduce whole-sequence wall time and reads, but the effect varies: wall -4.57% / -9.87%, reads -5.59% / -8.80%. This is a bounded task-level observation, with different explanation/code outputs.

Later requests 2-4 average 6.78% lower TTFT and 1.45% fewer physical reads; whole-request time is +0.29% / -5.90% by order, so that later-task time gain is not consistent in both orders. More importantly, after a prior request has reached capacity and evicted experts (requests 3-4 here), mean TTFT is **85.023 / 86.768 s** (+2.05%), physical reads **59.170 / 60.170 GB** (+1.69%), and pooled decode **2.354 / 2.317 token/s** (-1.58%). Both orders show those small unfavorable changes in this phase. The identical repeated retrieval alone is **92.011 / 92.193 s** to first token, effectively unchanged in this small sample. No sustained full-cache speedup is established.

### Capacity and admission

`STORE slots` is allocated capacity, not an occupancy count. The comparison
records when a previous request reaches 4595 allocated slots **and evicts
experts**, then reports later flow admission separately. No-room can include
floor refusal as well as absent free slots; it is not a standalone occupancy
metric. Reads, bytes copied into the store and overlapped copy/wait timers are
also separate quantities.

| Enabled run | Request | Retained | No-room skips | Early-DMA skips | Preload reads | Slots after decode | Request evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| on-a | 1 | 2916 | 0 | 1697 | 51 | 4032 | 0 |
| on-a | 2 | 584 | 140 | 58 | 227 | 4595 | 2785 |
| on-a | 3 | 0 | 1216 | 24 | 2446 | 4595 | 6186 |
| on-a | 4 | 0 | 3481 | 35 | 2523 | 4595 | 3205 |
| on-b | 1 | 2916 | 0 | 1697 | 51 | 4032 | 0 |
| on-b | 2 | 593 | 133 | 62 | 211 | 4595 | 2785 |
| on-b | 3 | 0 | 1197 | 22 | 2439 | 4595 | 6110 |
| on-b | 4 | 0 | 3510 | 30 | 2544 | 4595 | 3245 |

Both enabled runs retain 2916 experts on request 1, then 584 / 593 on request 2. The second request reaches the 4595-slot ceiling and reports evictions. Requests 3 and 4 retain **zero** new experts in both runs, with no-room counts 1216/3481 and 1197/3510. Existing cache hits still help, but the current free-slot-only API cannot replace old residents during the flow. The observed admission plateau matches that implementation boundary; an eviction/admission policy was not tested. The repeated retrieval uses the existing automatically sized layer cache (smaller than on the first request) under the same RAM floor, not an added memory pool.

### Answer and resource review

All complete answers and Python ASTs were read before the restricted functional
checks. Retrieval and repeated retrieval require the exact six-field JSON.
Python checks cover seven input vectors for each function, including input
preservation and Chinese docstrings. Explanation content and the requested
25–35 Han characters per item are reported separately; a length failure is
retained even when the content is correct.

| Run | Han counts for explanation items 1–6 | Length check | All bounded content checks |
| --- | --- | --- | --- |
| off-a | 30, 31, 31, 26, 30, 29 | True | True |
| on-a | 29, 33, 35, 31, 36, 37 | False | True |
| on-b | 30, 31, 35, 31, 34, 35 | True | True |
| off-b | 30, 31, 34, 31, 34, 37 | False | True |

All 16 answers pass the bounded content/function review and stop at EOS before their caps. All eight retrieval output-ID sequences match. The four Python answers pass **56** functional examples with input preservation. **14/16** answers satisfy all requested formats: on-a explanation items 5/6 contain 36/37 Han characters, and off-b item 6 contains 37, exceeding the requested 25-35. Those two format failures are retained; no answer was selectively rerun.

Peak sampled GPU use excluding the capacity hold is
**19.504 GiB**;
peak RSS is **39.88 GB**;
minimum MemAvailable is **9.11 GB**.
Engine swap and STORE read failures are zero. Samples are once per second plus
boundaries, with process HWM retained; no instantaneous-peak or system-wide
zero-swap claim is made. Initial VRAM expert counts are
**[1653]** and are observed rather than copied from the
previous session.

The 24 existing Python tests, 17 no-GPU option/env/resource/sequence checks,
eight prior input-ID controls and two pack/context rejection checks pass.
Only the benchmark and roadmap change. The four native source trees/binaries,
core archives, pack, production entry and 99 earlier runtime/header files are
preserved. The 7q evidence hashes are verified using the pre-7r snapshot for the
two actively edited files. No native rebuild or new HIP/SYCL validation is
claimed. The native implementation remains the previously verified CUDA/SM75
prototype; other-backend limitations from 7q still apply.

Reproduce one complete resident sequence in a new output directory:

```sh
python3 tools/deepseek41/benchmark_quantized.py \
  --suite resident --smoke-result build-deepseek-cuda/step7g-candidate-smoke/result.json \
  --exe build-deepseek-flow-store-sm75/strata-dsv41 \
  --engine-sha256 2a8220ef222451ca6cdc58ba83d2fb90b0cc43d476faf5668703c8b2d87bf07b \
  --store-stats --gpu-cap-mib 20480 --ssd-cache-gb 40 --ssd-policy lru \
  --prefill-chunk 256 --settle-store on --flow-store on \
  --output-dir build-deepseek-cuda/step7r-reproduction-on
```

`--flow-store off` is the control. `step7r-validation.json` indexes actual native
runs, answer reviews including failures, analysis, checks and the gate patch.
Two resident runs per setting establish a bounded observation, not general
quality or sustained-session performance. This gate does not compare the new
candidate against the production window profile.

**7r complete; user confirmed the next comparison.** Keep the opt-in prototype and current production defaults. Recommended next gate: compare this resident workload directly against the existing production window-prefill path on the same day, before deciding whether to invest in safe full-cache replacement or a path-selection policy. The authorized same-day comparison is recorded in 7s below. No policy change or default promotion was made.

## Step 7s: current production windows versus chunked flow-store

**Result: chunked handoff improves the first request, while production windows remain faster over the four-request sequence** This is a same-day comparison of complete native
paths. The 7r on/off experiment isolates handoff; this gate answers whether the
candidate is preferable to the current local production profile.

On 2026-10-26, fresh resident processes ran window-a, flow-a, flow-b, window-b.
Each receives the same four independent requests as 7r: input IDs
257/99/121/257, output caps 128/256/256/128, with the final request repeating the
first exactly. One READY and four DONE records per engine, fresh KV per request,
weights/profile/expert cache retained. There is no OS-cache drop, selective
rerun, conversation history, or additional model download.

- Window: exact production 7i binary `86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4`,
  `--prefill-chunk 0`, with native handoff absent.
- Flow: existing 7q binary `2a8220ef222451ca6cdc58ba83d2fb90b0cc43d476faf5668703c8b2d87bf07b`,
  `--prefill-chunk 256`, `DS4_FLOW_STORE=1`.
- Both: unchanged IQ1_M/Engram pack, 40 decimal GB LRU/4595-slot maximum,
  20 GiB GPU target, context1024, temperature0, 18 CPU workers, DSpark none,
  STORE/flow diagnostics and native verbose logs. RAM preload/GPU settle remain
  at their native defaults; the post-prompt settle stage belongs to chunk mode.

The only harness change is opt-in `--native-verbose`, enabling equal native
logging without forcing a chunk-only preload switch. Existing invocations keep
their command and environment. Native source, binaries, operators and the
production entry are unchanged. Workspace, initial GPU expert capacity and
post-prompt settling differ by design between paths; their effect is included
in the task-level comparison.

### Timing and physical reads

Times start at GEN after READY; startup is shown separately. Reads are process
physical read bytes during requests, excluding startup. Outputs include EOS.
Explanation/code are free generation, so unequal lengths and histories limit
exact-work interpretations. Decode pools actual decode steps (output count
minus one) over native decode time. Early-eight timing is retained in the JSON.

| Run | Startup s | Four-request time s | Physical reads GB | Output tokens 1–4 | Initial GPU experts |
| --- | ---: | ---: | ---: | --- | ---: |
| window-a | 43.03 | 472.94 | 200.36 | 32, 158, 187, 32 | 1740 |
| flow-a | 43.92 | 494.34 | 227.32 | 32, 146, 187, 32 | 1653 |
| flow-b | 44.75 | 492.26 | 225.73 | 32, 143, 187, 32 | 1653 |
| window-b | 43.40 | 469.59 | 198.86 | 32, 145, 192, 32 | 1741 |

The next table gives **window / flow** means.

| Request | First token s | Physical reads GB | Decode tokens/s |
| --- | ---: | ---: | ---: |
| long_retrieval | 124.16 / 99.30 | 58.40 / 56.22 | 1.710 / 1.402 |
| long_explanation | 48.29 / 44.02 | 42.29 / 48.31 | 3.138 / 2.472 |
| long_python | 55.92 / 80.54 | 56.92 / 71.58 | 2.533 / 2.309 |
| long_retrieval_repeat | 89.13 / 93.92 | 42.00 / 50.41 | 2.335 / 2.094 |

The first request averages 124.157 / 99.303 s to first token (window/flow), a 20.02% improvement for flow. Its whole-request time improves 14.67%, but decode falls from 1.710 to 1.402 token/s and early-eight decode takes 5.47 / 8.02 s. Across all four requests, windows average **471.266 s and 199.608 GB**; flow averages **493.304 s and 226.523 GB**. Flow is 4.68% slower and reads 13.48% more. Both orders agree: sequence wall +4.53% / +4.83%, physical reads +13.46% / +13.51%. Overall pooled decode is 2.608 / 2.231 token/s (-14.47%). These are bounded task observations, not a universal throughput result.

Later requests 2–4 have 13.01% higher TTFT and 20.60% more reads on flow. Fixed requests 3–4 average 72.523 / 87.232 s TTFT (+20.28%) and 49.458 / 60.995 GB reads (+23.33%). The repeated retrieval has identical 32-token output in every run, yet it takes 89.126 / 93.919 s to first token and reads 41.995 / 50.406 GB (window/flow). All eight retrieval outputs are identical by token ID; the Python outputs also match exactly in window-a/flow-a/flow-b. Window-b has a longer docstring. Explanation output lengths differ, so whole-sequence time is accompanied by per-task and token-normalized results.

### Cache state and resources

Allocated slots are not occupancy. Prior capacity pressure means a previous
request reported both the 4595-slot allocation ceiling and evictions. Requests
3/4 are compared by fixed sequence index in both arms, not by selecting
different request populations according to cache state. Handoff no-room can
also reflect the host memory floor. No-room, submitted STORE bytes and physical
SSD reads are distinct measurements.

| Run | Request | Prior capacity/eviction | Slots after decode | Request evictions | Handoff retained |
| --- | ---: | --- | ---: | ---: | ---: |
| window-a | 1 | False | 4595 | 2710 | n/a |
| window-a | 2 | True | 4595 | 4331 | n/a |
| window-a | 3 | True | 4595 | 6146 | n/a |
| window-a | 4 | True | 4595 | 5258 | n/a |
| flow-a | 1 | False | 4032 | 0 | 2916 |
| flow-a | 2 | False | 4595 | 2670 | 594 |
| flow-a | 3 | True | 4595 | 6139 | 0 |
| flow-a | 4 | True | 4595 | 3324 | 0 |
| flow-b | 1 | False | 4032 | 0 | 2916 |
| flow-b | 2 | False | 4595 | 2650 | 589 |
| flow-b | 3 | True | 4595 | 6079 | 0 |
| flow-b | 4 | True | 4595 | 3277 | 0 |
| window-b | 1 | False | 4595 | 2696 | n/a |
| window-b | 2 | True | 4595 | 4300 | n/a |
| window-b | 3 | True | 4595 | 6142 | n/a |
| window-b | 4 | True | 4595 | 5238 | n/a |

Both window runs reach the allocation ceiling and report evictions during request 1. Both flow runs do so during request 2; they retain 2916 experts on request 1, then 594 / 589 on request 2, and zero on requests 3/4. Mean post-prompt RAM preload is 38.622 s on the Python request and 40.345 s on repeated retrieval. The free-slot handoff boundary remains, and substantial later reads remain. Initial GPU experts are 1740/1741 for windows versus 1653/1653 for flow: different workspace is intentional, with one-expert variation inside the window arm. Automatic repeated-query layer-cache sizing is 1.4 / 0.8 GB in the two flow runs, versus 3.0 GB on the first request, under the original host floor. No fixed initial placement or larger layer-cache pool was forced.

The existing native selector chose flow for all eight chunk-enabled requests. On requests 3/4 its chosen-path predictions were 15.77/30.86 s and 15.20/30.79 s, while measured full prefill was about 80–81 / 94 s. This discrepancy and the large preload cost justify a controlled selector experiment, but do not establish a single root cause. The read-only `step7s-path-audit.json` confirms that `mx_mat` already resolves ready ExpertStore slots before RAM-residency sampling; missing store awareness is not an established bug. `pick_path`, `forward_chunks` and `feed_windows` already exist in the same model. A mixed path will still have the chunk engine's workspace/placement, so its benefit must be measured rather than estimated by combining separate-run results. The expert-cache settle timer includes RAM preload; the timers are not summed twice.

| Run | Peak RSS GB | Peak GPU GiB | Minimum host available GB |
| --- | ---: | ---: | ---: |
| window-a | 39.67 | 19.458 | 8.84 |
| flow-a | 39.88 | 19.504 | 8.85 |
| flow-b | 39.85 | 19.504 | 8.95 |
| window-b | 39.70 | 19.464 | 8.91 |

All runs obey the live 8 GiB host-available floor and 20 GiB GPU target; engine
swap and STORE read failures are zero. Memory is sampled once a second plus
request boundaries, and process HWM is retained. GPU excludes the capacity hold
but includes driver/context/other users. These are not instantaneous-peak or
system-wide zero-swap claims. Hardware is the local RTX2080Ti, limited to
20 GiB usable GPU memory, 64 GB-class RAM, Xeon E5-2686v4 and SATA SSD.

### Answers and checks

All complete Chinese/code outputs were read before restricted Python execution.
The retrievals require the exact six-field JSON; Python uses seven vectors per
function, checking values, input preservation and Chinese docstrings. The six
explanation topics and 25–35 Han characters per item are reviewed separately.
Python fences are stripped for execution as before, but this gate also flags
them separately against the explicit code-only output request. This stricter
format accounting does not change timings or imply a regression versus 7r.

| Run | Han counts 1–6 | Length pass | Raw Python only | Bounded content pass |
| --- | --- | --- | --- | --- |
| window-a | 30, 32, 35, 30, 34, 37 | False | False | True |
| flow-a | 30, 30, 33, 29, 32, 33 | True | False | True |
| flow-b | 29, 31, 33, 29, 30, 31 | True | False | True |
| window-b | 30, 31, 31, 27, 29, 29 | True | False | True |

All 16 bounded content/function checks pass and all requests reach EOS before the output cap. All eight retrieval answers match the required six-field JSON; 56 restricted Python functional examples pass with unchanged inputs. One explanation fails the requested length: window-a item 6 has 37 Han characters. All four code answers use Markdown fences, so strict plain-code formatting fails even though extracted code and docstrings pass. Strict request-format accounting is therefore **11/16**, or **15/16** under the prior fence-normalized rule. This accounting difference is explicitly separated from performance and from general quantization-quality claims; no answer failure was discarded or selectively rerun.

Passed: 24 existing Python tests, 20 no-GPU option/environment/resource/sequence
checks, eight prior input-ID controls and two expected pack/context rejections.
The gate patch is checked against the 204-file snapshot. All 99 prior runtime/
header files, four native source trees/binaries, three core archives, eight pack
files and production entry are preserved; main checkout is clean. Previous 7r
hashes are checked using the pre-7s snapshot for the edited benchmark/roadmap.
No native rebuild or new HIP/SYCL validation is claimed. An initial offline
analysis-regex quote error was corrected; failing and passing compile logs are
retained. It did not affect native measurements or require a model rerun.

`build-deepseek-cuda/run-step7s-comparison.py` records exact commands and hashes;
`step7s-comparison.json`, `step7s-analysis.json`, the four `step7s-*/result.json`
files and adjacent answer reviews retain both orders, failures and phase data.
`step7s-validation.json` indexes the evidence and gate patch. Two processes per
path and four bounded requests do not establish general quality, long-context
or sustained-session performance.

**7s complete; user confirmed the mixed-path experiment.** Retain the production window profile and the opt-in flow-store prototype. Recommended next gate: reuse Strata's existing native path selector/call paths for a bounded, opt-in same-process mixed-path experiment (first request via flow, later requests via windows), preserving the chunk workspace and measuring real costs before calibrating a production selection policy. This is a hypothesis to test, not a promised speedup or a default change. Full-cache replacement is deferred until this simpler reuse path is assessed; no new operator or scheduler is proposed.
The authorized mixed-path experiment is recorded in 7t below.

## Step 7t: native first-flow/later-window experiment

**Result: the native mixed path preserves the first-request gain and reduces this four-request sequence by 4.52%** This gate implements and measures the controlled
mixed-path experiment recommended in 7s. It reuses Strata's native operators,
ExpertStore, `pick_path`, `forward_chunks` and `feed_windows`.

`DS4_MIXED_PREFILL=1` selects flow for the first successful eligible prefill in a
model's lifetime and windows for later eligible prefills. The small schedule
state is in a native header; the model calls it at the existing path-selection
and completion boundaries. Failed/cancelled feeds do not advance the state.
Each feed clears any uncommitted choice before a single-token or all-logits
bypass can complete; bypasses do not consume the first flow. KV reset keeps the
phase, while a new model starts over. This is a controlled experiment, not an
adaptive cost model. For this resident suite each independent request has
exactly one eligible feed, so the observed schedule is flow/windows/windows/windows.
Split prompts, prefix reuse, interrupted native requests and all-logits numerical
execution are not new hardware-validation claims; host state tests cover their
schedule boundaries, while the pre-existing native caller owns execution.

The default is off: the original selector remains in control. Benchmark
`--mixed-prefill on|off` requires a pinned binary, resident suite, chunk256 and
flow-store on; it enables native timing logs and isolates inherited environment
variables. The 20 GiB GPU limit, 8 GiB host floor, 40 decimal GB LRU/4595-slot
maximum, context1024, 18 CPU workers, DSpark none, fresh KV and model pins remain.
The mixed engine keeps its chunk workspace throughout the sequence; it does not
recover the extra GPU expert capacity of a window-only load. Post-prompt RAM
preload/GPU settling run on the first flow request; later windows fill the store
through the original CPU path. No cache replacement algorithm was added.

### Build and default-off control

New binary: `build-deepseek-mixed-prefill-sm75/strata-dsv41`,
SHA256 `8cdf46e2c28e365a11bffec7b31ac6820a246599c0aa75f812874777bad7be42`. CUDA12.8/SM75 Release with portable AVX2 builds
successfully; all ten embedded cubins target SM75. The parent 7q binary remains
unchanged. HIP/SYCL compilers are unavailable, so those builds/execution remain
unverified; this is a local CUDA prototype.

`tools/deepseek41/patches/mixed_prefill.patch` applies and reverses exactly on
pinned 7q sources. It changes native path selection and adds the small state
header, without changing kernels. The real header passes the host ASan/UBSan
check for disabled delegation, bypass, failed/cancelled retry, success transition,
repeated windows and model isolation. The 24 existing Python tests, 27 mock
option/env/resource checks, eight prior input-ID controls and two expected
pack/context rejections also pass.

Two real default-off probes run parent7q then candidate7t. Both Python launchers
inherit `DS4_MIXED_PREFILL=1`; the benchmark must strip it because neither probe
explicitly selects the feature. Native logs must show the original selector,
no mixed-schedule records, one flow and the existing handoff. Both retrievals
match by token ID and produce the required JSON. These probes validate bounded
compatibility and resource behavior, not a speedup.

| Probe | TTFT s | Physical reads GB | Initial GPU experts | Handoff retained |
| --- | ---: | ---: | ---: | ---: |
| parent | 120.00 | 55.97 | 1640 | 2930 |
| candidate | 98.39 | 55.03 | 1640 | 2930 |

The parent probe reported 0.40 GB/s stream reads versus 0.49 GB/s in the new
default-off probe, despite the same 9426 streamed-expert count. This timing
variation is retained and is not attributed to the disabled new feature.

### Same-day resident comparison

Four fresh processes run window-a, mixed-a, mixed-b, window-b on the local
RTX2080Ti (20 GiB GPU target), 64 GB-class RAM, Xeon E5-2686v4 and SATA SSD.
Window uses the exact production 7i binary; mixed uses the new 7t binary with
handoff and schedule enabled. Both retain the same native diagnostics.
Inputs/caps are 257/99/121/257 and 128/256/256/128, identical to 7s. Every sequence
has one READY, four DONE records, one engine PID, fresh KV and resident
weights/profile/cache. Each mixed sequence logs one flow plus three windows,
with only one handoff/preload record. No OS-cache drop or selected rerun is used.

Times begin at GEN after READY; startup is separate. Physical reads are process
request reads, excluding startup. Decode pools actual steps (output count minus
one) over native decode time. Outputs include EOS. Free explanation/code lengths
can differ; these are task-level measurements with output and quality checks.

| Run | Startup s | Four-request time s | Physical reads GB | Output tokens 1–4 | Initial GPU experts |
| --- | ---: | ---: | ---: | --- | ---: |
| window-a | 33.52 | 475.22 | 198.85 | 32, 138, 187, 32 | 1728 |
| mixed-a | 43.28 | 451.01 | 194.04 | 32, 140, 192, 32 | 1640 |
| mixed-b | 43.39 | 457.39 | 195.90 | 32, 165, 187, 32 | 1640 |
| window-b | 45.73 | 476.15 | 198.96 | 32, 141, 193, 32 | 1726 |

The following gives **window / mixed** means; early-eight timings, phase counters,
cache pressure, memory and both orders remain in `step7t-analysis.json`.

| Request | First token s | Physical reads GB | Decode tokens/s |
| --- | ---: | ---: | ---: |
| long_retrieval | 125.07 / 93.82 | 58.44 / 54.49 | 1.685 / 1.405 |
| long_explanation | 48.45 / 49.54 | 40.55 / 42.78 | 3.017 / 3.031 |
| long_python | 56.05 / 57.23 | 57.36 / 55.51 | 2.444 / 2.489 |
| long_retrieval_repeat | 91.01 / 92.10 | 42.56 / 42.19 | 2.305 / 2.259 |

First-token latency for the identical retrieval averages **125.067 / 93.821 s** (window/mixed), a 24.98% improvement. First-request wall time is 143.462 / 115.890 s (-19.22%). First-request decode is slower, 1.685 / 1.405 token/s; early-eight decode takes 5.49 / 8.08 s. Across four requests, mean request time is **475.683 / 454.201 s** (-4.52%), with **198.904 / 194.968 GB** physical reads (-1.98%). Both orders improve whole-sequence time (-5.09% / -3.94%) and reads (-2.42% / -1.54%). Including measured startup, mean startup-plus-request time is **515.309 / 497.540 s**; startup varies, especially window-a. These are task-level observations on two sequences per path, not an exact-work microbenchmark or a general speedup claim.

The gain is concentrated in the first request. Later requests average 65.170 / 66.290 s TTFT (+1.72%), essentially the same mean physical reads (+0.01%), and pooled decode 2.622 / 2.661 token/s (+1.46%). Their mean task time rises 1.83%, with longer mixed explanation outputs. The identical repeated retrieval takes 91.009 / 92.101 s to first token, so sustained repeated-query acceleration is not established. Overall pooled decode is 2.511 / 2.489 token/s (-0.88%). All eight resident retrieval outputs match by token ID; explanation and most Python outputs differ and remain in the evidence.

Each mixed process records exactly one flow followed by three windows, one native handoff and one post-prompt preload. Both retain 2930 experts on the first request; subsequent handoff/preload fields are absent because those stages do not run, not because an admission attempt failed. The original CPU window path handles later cache fills and eviction. Allocation and prior-capacity/eviction flags remain recorded separately; allocated slots alone are not occupancy. Mixed retains 1640 initial GPU experts in both runs, versus 1728/1726 for production windows. Its chunk workspace remains present. Ambient GPU capacity and host memory also vary across runs, and counts differ from 7s; the conclusions use fresh same-day controls, not historical timings. No full-cache replacement or extra RAM pool was implemented.

| Run | Peak RSS GB | Peak GPU GiB | Minimum host available GB |
| --- | ---: | ---: | ---: |
| window-a | 39.62 | 19.461 | 10.96 |
| mixed-a | 39.96 | 19.508 | 10.47 |
| mixed-b | 40.01 | 19.508 | 10.94 |
| window-b | 39.68 | 19.522 | 13.02 |

All six native processes obey the resource limits; engine swap and STORE read
failures are zero. Samples are once a second plus boundaries, with process HWM
retained. GPU excludes the capacity hold but includes driver/context/other users;
these are not instantaneous peaks or system-wide swap claims. Initial placement
is observed, not asserted equal to a previous session or to the other path.

### Answer review and evidence

Complete outputs are read before restricted Python execution. Eight resident
retrievals require the exact JSON; seven vectors per function check Python
results and unchanged inputs. Explanation content, Han length and strict plain-
Python formatting are separate. Fences are stripped for execution but counted
as code-only format failures, consistently with 7s.

| Run | Han counts 1–6 | Length pass | Raw Python only | Bounded content pass |
| --- | --- | --- | --- | --- |
| window-a | 29, 31, 31, 26, 26, 29 | True | False | True |
| mixed-a | 27, 29, 33, 29, 31, 33 | True | False | True |
| mixed-b | 30, 31, 36, 33, 36, 40 | False | False | True |
| window-b | 27, 27, 31, 29, 34, 35 | True | False | True |

Both compatibility retrievals and all eight resident retrievals produce the required JSON. All 16 resident content/function checks pass and every request reaches EOS before its cap. The 56 restricted Python examples pass, including input preservation. Strict request-format success is **11/16**: mixed-b explanation items 3/5/6 contain 36/36/40 Han characters (limit35), and all four Python answers have Markdown fences. Under the prior fence-normalized rule this is 15/16. These failures are retained; longer outputs and quality variation are not discarded or selectively rerun. This is not a general quantization-quality evaluation.

The production entry, four prior native source trees/binaries, core archives,
model/pack and 99 earlier runtime/header files are preserved. The new fifth
native tree has 1138 pinned files. Prior 7s evidence is checked using the pre-7t
snapshot for the edited benchmark/roadmap. Main checkout is clean. No download,
commit, push or PR is part of this gate.

Exact commands/hashes are in `step7t-candidate-build.json`,
`run-step7t-validation.py`, `run-step7t-comparison.py` and their logs. The native
compatibility report, four raw result directories, adjacent answer reviews,
analysis, gate patch and preservation checks are indexed by
`step7t-validation.json`. Two resident sequences per setting do not establish
long-context, conversation-reuse or general quantization quality/performance.

**7t complete; user confirmed.** The native prototype and its measurements are preserved. Step7u below tests the recommended first-prompt length and order variations.

## Step 7u: first-prompt length and request order

**Result: forcing the first prefill through flow helps this long-first sequence, but regresses the short-first sequence.**

This gate adds benchmark-only `--resident-order short-first|long-first`.
Both orders use the same existing fixtures twice: a 14-token Chinese capital
question (cap64), and the 257-token warehouse retrieval (cap128). Orders are
short/long/short/long and long/short/long/short. Prompt IDs match the earlier
fixtures exactly. Each process has one READY, four DONE records and fresh KV
per request, while weights, routing profile and the expert store stay resident.

Unchanged production7i windows are compared with the exact 7t mixed engine:
first eligible prefill flow, then three windows, with the existing handoff.
No native source, operator, cache policy, production entry or default changes.
Both binaries are SHA256-pinned and reuse their prior CUDA12.8/SM75 builds;
no new native build is needed. HIP/SYCL execution remains untested.

Measurements run on the RTX2080Ti with a 20 GiB target, 64 GB-class host RAM,
Xeon E5-2686v4 (18 CPU workers), and SATA SSD. Limits remain context1024,
40 decimal GB LRU/4595 slots, 8 GiB host available floor, zero engine swap,
temperature0 and DSpark none. Identical native verbose/STORE diagnostics are
enabled. Chunk workspace remains allocated in mixed mode; actual GPU expert
placement is recorded rather than asserted equal. No OS cache drop is used.

Run order was fixed before measurement: short-window-a, short-mixed-a,
long-window-a, long-mixed-a, long-mixed-b, long-window-b, short-mixed-b,
short-window-b. Thus each prompt order has two window/mixed comparisons in
opposite execution orders. No native run is selectively discarded or rerun.

### Same-day results

Times below start at GEN after READY; startup is separate. Physical reads are
per-process request reads, excluding startup. Decode uses output count minus
one over native decode time. Short/long are labels for 14/257 tokens, not a
long-context benchmark. Both task content and input length change between the
first-prompt examples; two points cannot identify a general length threshold.

| Run | Startup s | First TTFT s | Four-request time s | Reads GB | Output tokens 1–4 | Initial GPU experts |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| short-window-a | 43.14 | 21.82 | 268.41 | 106.98 | 6, 32, 6, 32 | 1729 |
| short-mixed-a | 41.51 | 26.20 | 274.53 | 110.02 | 6, 32, 6, 32 | 1641 |
| long-window-a | 43.81 | 124.70 | 268.47 | 106.35 | 32, 6, 32, 6 | 1729 |
| long-mixed-a | 41.06 | 99.66 | 233.88 | 97.71 | 32, 6, 32, 6 | 1641 |
| long-mixed-b | 41.78 | 97.36 | 234.23 | 97.91 | 32, 6, 32, 6 | 1641 |
| long-window-b | 36.89 | 125.33 | 266.66 | 106.41 | 32, 6, 32, 6 | 1729 |
| short-mixed-b | 41.87 | 23.95 | 268.60 | 108.01 | 6, 32, 6, 32 | 1641 |
| short-window-b | 41.75 | 21.87 | 264.37 | 105.44 | 6, 32, 6, 32 | 1729 |

The following reports **window / mixed** means, two runs per path/order.

| First prompt | First TTFT s | Four-request time s | Reads GB | Later mean TTFT s | Pooled decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-first | 21.84 / 25.07 | 266.39 / 271.57 | 106.21 / 109.02 | 67.69 / 67.83 | 1.736 / 1.674 |
| long-first | 125.01 / 98.51 | 267.57 / 234.06 | 106.38 / 97.81 | 34.30 / 31.66 | 1.816 / 1.774 |

Short-first mean TTFT rises from 21.842 to 25.073 s (+14.79%). Its first complete request rises from 26.009 to 32.845 s (+26.28%). Four-request time rises from 266.388 to 271.565 s (+1.94%); reads rise from 106.210 to 109.017 GB (+2.64%). Long-first mean TTFT falls from 125.014 to 98.514 s (-21.20%). Its first complete request falls from 143.137 to 120.135 s (-16.07%). Four-request time falls from 267.565 to 234.057 s (-12.52%); reads fall from 106.382 to 97.810 GB (-8.06%). Both pairs agree on each sequence's direction. Including startup gives 308.836 / 313.254 s for short-first and 307.916 / 275.476 s for long-first. Startup varies and remains separate in the evidence. First-request pooled decode is slower in mixed mode for both tasks: 1.200 / 0.643 tokens/s for short, and 1.710 / 1.433 for long. This is a prefill/cache observation, not a general decode-throughput improvement.

| First prompt | Pair | First TTFT change | Four-request time change | Physical read change |
| --- | --- | ---: | ---: | ---: |
| short-first | a | +20.07% | +2.28% | +2.85% |
| short-first | b | +9.52% | +1.60% | +2.44% |
| long-first | a | -20.08% | -12.88% | -8.12% |
| long-first | b | -22.31% | -12.16% | -7.99% |

Per-request TTFT follows the fixed request indices; cache-pressure-selected
populations are not mixed together. Values are **window / mixed** seconds.

| Order | Request 1 | Request 2 | Request 3 (repeat) | Request 4 (repeat) |
| --- | ---: | ---: | ---: | ---: |
| short-first | 21.84 / 25.07 | 108.64 / 113.66 | 9.58 / 8.54 | 84.85 / 81.28 |
| long-first | 125.01 / 98.51 | 9.59 / 10.12 | 84.55 / 76.10 | 8.76 / 8.76 |

All four mixed processes log exactly one flow followed by three windows. The short14 flow streams 1349 experts (10.0 GB file payload), retaining 226 in the RAM store; both logs report zero additional post-prompt preloads. The long257 flow reports 9437 expert uses, 6055 from files (45.1 GB) and 3382 from the layer cache; both retain 2942 RAM-store experts and preload another 56/60. These counters describe different routing/cache histories and overlap with other work; they are not additive timing components or an isolated causal proof. Initial GPU placement is 1729 experts for every window process versus 1641 for every mixed process, whose chunk workspace persists. Later short-first mean TTFT is nearly unchanged (+0.21%); later long-first mean TTFT falls 7.70%, including repeated retrieval at 84.547 / 76.100 s. Those are fixed request-index comparisons. STORE slots remain allocated capacity rather than occupancy; the full phase counters and prior-eviction flags are retained in the analysis. Two distinct prompts and two repeats per arm support a scoped finding about these sequences, not a universal short/long cutoff or conversation-context reuse claim.

### Quality, resources and evidence

All 32 requests reach EOS before their caps and pass their requested content/format checks: 16 correct one-sentence Chinese answers, and 16 strict JSON objects with unique keys and exactly the six expected values. All short outputs are the identical six token IDs (including EOS); all retrieval outputs are the identical 32 IDs. Thus each sequence generates exactly 76 output tokens and 72 decode steps. No answer failure or truncation is discarded. This deliberately narrow two-fixture check does not replace the broader quality findings in 7t or evaluate general quantization quality.

| Run | Peak RSS GB | Peak GPU GiB | Minimum host available GB |
| --- | ---: | ---: | ---: |
| short-window-a | 38.34 | 19.463 | 14.25 |
| short-mixed-a | 38.93 | 19.510 | 14.35 |
| long-window-a | 38.38 | 19.463 | 14.38 |
| long-mixed-a | 38.31 | 19.510 | 14.47 |
| long-mixed-b | 38.45 | 19.510 | 14.46 |
| long-window-b | 38.35 | 19.463 | 14.38 |
| short-mixed-b | 38.97 | 19.510 | 13.90 |
| short-window-b | 38.41 | 19.463 | 14.48 |

All eight native runs satisfy the bounds, with zero engine swap and STORE read
failures. Samples are once a second plus boundaries, with process HWM kept;
GPU excludes the capacity hold but includes driver/context/other users.
These are not instantaneous peak or system-wide swap claims.

The 24 existing Python tests and 36 no-GPU option/environment/resource/sequence
checks pass. New checks cover both orders in both modes, exact prior input IDs
and caps, environment isolation, fresh KV, and rejection of unsupported input
combinations before GPU/output-directory creation. The first control script
failed on a historical report schema: 7i stored cap64 at method level, while the
new assertion expected a per-case key. The old report was inspected, that
assertion was fixed, and its failed script/log/results were retained in
`step7u-failed-controls-v1`. No native measurement had started at that failure.

The five prior native source trees/binaries, model/pack, core archives,
production entry, 99 earlier runtime/header files and all 83 prior7t evidence
hashes are preserved (using the pre-7u snapshot for edited benchmark/roadmap).
No model download, commit, push or PR occurs. Main checkout remains clean.

Exact commands and native logs are in `run-step7u-comparison.py`,
`step7u-comparison.json` and the eight result directories. Raw results remain
unchanged; adjacent answer reviews pin their hashes. Analysis, complete output
groups, retained tooling failure, gate patch and preservation checks are
indexed in `step7u-validation.json`.

**7u complete; user confirmed.** Its measurements and experimental mixed mode remain preserved. Step7v below validates the original chooser on the recommended first-prompt cases.

## Step 7v: original native chooser on first requests

**Result: the original chooser keeps the long-first gain but still sends this short first prompt through the slower flow profile.**

This gate adds benchmark-only `--first-request-only`, requiring the existing
pinned resident-order setup. Each fresh process receives exactly one eligible
feed: the unchanged short14/cap64 or retrieval257/cap128 fixture. One READY
and one DONE are required. There is no previous request/KV history; native
startup calibration still runs and OS cache is not cleared.

The reference is the unchanged production7i window binary. Auto uses the exact
7t binary with chunk256, flow-store on and mixed-prefill **off**, delegating to
Strata's original `pick_path`. No forced mixed-path record is permitted.
Native calibration, both predicted costs and the actual selected path are
required. Existing operators, cache policy, production entry and defaults are
preserved. No native rebuild is needed; prior CUDA12.8/SM75 builds are pinned
by SHA256. HIP/SYCL execution remains untested.

Hardware remains RTX2080Ti (20 GiB target), 64 GB-class RAM, Xeon E5-2686v4
with 18 CPU workers, and SATA SSD. Context 1024, 40 decimal GB LRU/4595 slots,
8 GiB host available floor, zero engine swap, temp0 and DSpark none remain.
Both paths have the same verbose/STORE diagnostics. Auto keeps chunk workspace;
its GPU expert placement can differ from the window-only load.

Before measurement the order was fixed: short-window-a,short-auto-a,
long-window-a,long-auto-a,long-auto-b,long-window-b,short-auto-b,short-window-b.
Thus each fixture has a window/auto pair in both execution orders. No native
run is discarded or selectively repeated. This is eight first requests, not
a measurement of subsequent learning or first-auto/later-window integration.

### Same-day measurements

All request times start after READY; startup is separate. Physical reads are
per-process request reads. Decode excludes the first sampled output.

| Run | Path | Startup s | TTFT s | Request s | Reads GB | Output tokens | Initial GPU experts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-window-a | windows | 34.96 | 21.85 | 25.71 | 11.48 | 6 | 1729 |
| short-auto-a | flow | 32.53 | 23.35 | 30.32 | 13.76 | 6 | 1641 |
| long-window-a | windows | 32.64 | 125.53 | 143.85 | 57.95 | 32 | 1729 |
| long-auto-a | flow | 41.80 | 98.73 | 120.44 | 54.43 | 32 | 1641 |
| long-auto-b | flow | 35.50 | 97.44 | 119.45 | 53.90 | 32 | 1641 |
| long-window-b | windows | 36.91 | 124.92 | 142.47 | 56.28 | 32 | 1729 |
| short-auto-b | flow | 39.13 | 24.70 | 32.44 | 14.97 | 6 | 1641 |
| short-window-b | windows | 32.76 | 22.04 | 25.91 | 11.45 | 6 | 1729 |

Means below are **production window / native auto**, two runs per setting.

| Fixture | TTFT s | Request s | Physical reads GB | Decode tokens/s | Startup + request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 21.95 / 24.02 | 25.81 / 31.38 | 11.47 / 14.37 | 1.292 / 0.680 | 59.68 / 67.20 |
| long | 125.22 / 98.08 | 143.16 / 119.95 | 57.11 / 54.17 | 1.728 / 1.418 | 177.94 / 158.60 |

Short14 mean TTFT is 21.946 / 24.023 s (window/auto, +9.47%); complete request time is 25.814 / 31.376 s (+21.55%), and physical reads are 11.465 / 14.366 GB (+25.30%). Long257 mean TTFT is 125.223 / 98.081 s (-21.68%); request time is 143.161 / 119.946 s (-16.22%), and physical reads are 57.112 / 54.165 GB (-5.16%). Both execution-order pairs agree: short TTFT regresses 6.86%/12.05%; long improves 21.35%/22.00%. Auto decode is slower for both, at 0.680 versus 1.292 tokens/s for short and 1.418 versus 1.728 for long. Thus the long-request benefit comes from prefill, not a general decode-throughput gain. Startup varies from 32.53 to 41.80 s; startup-plus-request means are 59.677 / 67.204 s (short) and 177.938 / 158.597 s (long). First-request results do not establish subsequent resident-session behavior.

| Fixture | Pair | TTFT change | Request time change | Read change |
| --- | --- | ---: | ---: | ---: |
| short | a | +6.86% | +17.90% | +19.83% |
| short | b | +12.05% | +25.17% | +30.80% |
| long | a | -21.35% | -16.27% | -6.07% |
| long | b | -22.00% | -16.16% | -4.22% |

### Decisions and prediction gaps

| Auto run | Selected | Selected prediction s | Observed native prefill s | Window prediction s | Independent window prefill s | Estimated RAM resident |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short-auto-a | flow | 40.23 | 23.35 | 31.31 | 21.85 | 7% |
| short-auto-b | flow | 39.77 | 24.70 | 25.50 | 22.04 | 9% |
| long-auto-a | flow | 205.85 | 98.72 | 178.37 | 125.53 | 6% |
| long-auto-b | flow | 198.88 | 97.43 | 174.97 | 124.92 | 8% |

All four auto runs use the original chooser, emit a calibration and one decision, and execute flow. The native rule at kPickRes=0.9 requires a predicted-faster window path to cost at most half of flow when estimated CPU-expert RAM residency is below 90%. Here it reports 6-9% residency. All four predictions rank windows faster but fail that twofold margin, explaining the recorded choice. For the short prompt, the selected prediction is 40.23/39.77 s versus actual native prefill 23.348/24.695 s; window predictions are 31.31/25.50 s versus independent window 21.849/22.041 s. For the long prompt, flow predictions are 205.85/198.88 s versus actual 98.725/97.429 s; window predictions are 178.37/174.97 s versus independent window 125.528/124.915 s. These logs and the source establish why the choice is made. The full prediction error is not reduced to a single proven cause. In particular, using only the lower recorded prediction would choose windows for both long cases too, while the independently measured production-window profile is slower there. That is an inference from the logged rule and separate references, not a tested policy change. A blanket removal of the margin is therefore not supported by these measurements.

The estimator reports 20.2/100.8 GB of CPU-lane expert demand for short/long. Actual flow logs show 10.0/45.1 GB file payload; long also reuses 3382 expert uses from its layer cache. These are different quantities, not directly interchangeable physical-read totals. Both short runs retain 226 RAM experts and preload 0 more; both long runs retain 2942 and preload 59/65 more. Initial placement is 1729 GPU experts in each window reference and 1641 in each auto process. Their workspace/placement differences remain part of the profile comparison; a same-workspace forced-window probe is still needed to isolate a short override's benefit. Only two fixed prompts are measured, so no universal token threshold is established.

Predictions estimate expert transfer/compute costs; observed prefill includes
additional path/checkpoint/settle work. These are recorded prediction gaps,
not a fitted replacement model. The window estimate is compared to a separate
production process with different workspace/placement, not an identical-state
counterfactual. Raw calibration rates, estimated expert GB, flow/handoff/preload
and STORE counters remain in the report. STORE submitted bytes are payload;
physical reads and allocated slots are distinct measurements.

### Quality, resources and preservation

All eight requests reach EOS before their caps: four correct one-sentence Chinese answers and four strict JSON objects with unique keys and all six expected values. Every short response has the same six token IDs and every retrieval has the same 32, including EOS. Content and requested format checks all pass; raw results are preserved. This is a narrow two-fixture check and does not replace the broader quality findings from earlier gates.

| Run | Peak RSS GB | Peak GPU GiB | Minimum host available GB |
| --- | ---: | ---: | ---: |
| short-window-a | 13.36 | 19.463 | 39.51 |
| short-auto-a | 7.27 | 19.510 | 45.87 |
| long-window-a | 37.12 | 19.463 | 16.09 |
| long-auto-a | 32.83 | 19.510 | 20.38 |
| long-auto-b | 33.06 | 19.510 | 20.24 |
| long-window-b | 37.11 | 19.463 | 16.03 |
| short-auto-b | 7.54 | 19.510 | 45.24 |
| short-window-b | 13.36 | 19.463 | 39.33 |

All eight runs satisfy the resource limits. Engine swap and STORE read failures
are zero. Memory uses one-second/boundary samples and process HWM; GPU excludes
capacity hold but includes driver/context/other users. This is not a system-wide
zero-swap or instantaneous-peak claim.

The 24 existing Python tests and 45 no-GPU controls pass. Controls cover the
original four-request behavior, one-request metadata/caps and exact input IDs,
non-mutation of fixtures, resource guards and rejection before GPU startup.
Inherited DS4_PATH_PICK, DS4_CHUNK_MIN and forced-mixed switches are removed;
auto explicitly records DS4_MIXED_PREFILL=0. No native implementation test or
build is claimed for this benchmark-only change.

Five native source trees/binaries, core archives, model/pack, production entry
and 99 earlier runtime/header files remain unchanged. All 127 prior 7u evidence
hashes are verified, using the pre-7v snapshot for edited benchmark/roadmap.
Main checkout is clean; no download, commit, push or PR is part of this gate.

`step7v-comparison.json` records exact commands and native decisions. Raw
results remain unchanged and adjacent answer reviews pin their hashes.
`step7v-selector-source-review.json` identifies the existing chooser rule.
Analysis, patch, preservation checks and all supporting artifacts are indexed
in `step7v-validation.json`.

**7v complete; user confirmed.** The first-short-feed experiment recommended here is validated in 7w below; prior measurements remain preserved.

## Step 7w: default-off first-short-prefill window guard

**State and boundary checks pass, but the fixed first-feed <=16 window guard has no stable performance benefit; retain default off.**

This experiment adds `DS4_FIRST_WINDOW=1` to the pinned native engine and
benchmark-only `--first-window on|off`. The existing `kWin=16` bounds a
one-window candidate, not a fitted or universally optimal cutoff. The first
successful eligible feed uses existing windows if it fits; a larger first feed
uses the original chooser and also consumes the opportunity on success.
All subsequent eligible feeds return to the original chooser. This is not the
first-auto/later-window resident integration.

Only the existing model and prefill-state header change. The reviewable patch
is `tools/deepseek41/patches/first_window_prefill.patch`; it builds a separate
`build-deepseek-first-window-sm75/strata-dsv41`. Native windows/flow execution,
chunk allocation/loading/calibration, automatic cost formula and KV reset are
byte-identical to the 7t parent. No new operator, cache policy or memory budget
is introduced. The production entry and defaults remain unchanged.

State selection is pending until the existing success callback. Failed or
cancelled execution does not consume it; `begin_feed` clears stale pending
state. Calls that bypass `pick_path` do not consume it. In the tested native
configuration this includes single-token calls and `Model::feed` all-logits
calls above the window width. These failure/bypass boundaries are host-tested
and source-reviewed, not newly exercised hardware cancellation/all-logits or
prefix-reuse workloads. KV reset preserves the model-lifetime state; a new
model starts fresh. Forced windows leave `pm.last=-1`, so that forced decision
does not train the original cost factors. The existing mixed schedule retains
priority if both switches are forced externally; the benchmark rejects such
combinations and requires explicit `mixed-prefill off`.

### Fixed validation setup

CUDA 12.8 Release/SM75 with portable AVX2 builds successfully; all ten cubins are
SM75. HIP/SYCL compilers are unavailable, so those builds/executions remain
unverified. Candidate SHA256:
`a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96`.

All native runs retain the RTX 2080 Ti 20 GiB target, 8 GiB host-available floor,
40 decimal GB LRU/4595 slots, context 1024, chunk 256, flow-store on, 18 CPU
workers, temp 0, DSpark none and verbose/STORE diagnostics. The existing
14-token Chinese question and 257-token retrieval are preserved. Prefixing
the Chinese question with `请问，` / `请问一下，` gives exact 16/17-token
fixtures under the pinned tokenizer. Caps stay 64 for Chinese QA and 128 for
retrieval. These are related prompts with different token/routing content;
the experiment does not isolate token count from content.

Twenty-two processes produce 24 independent requests, with fresh KV. Four
parent/candidate probes omit the new flag and verify default-off compatibility
for 14/257. Their inherited `DS4_FIRST_WINDOW=1` is stripped by the harness.
The performance matrix uses only the same candidate binary, with identical
chunk workspace and commands apart from the on/off environment value.
Each fixture is paired in both execution orders:14-off-a,14-on-a,16-off-a,
16-on-a,17-off-a,17-on-a,257-off-a,257-on-a,257-on-b,257-off-b,17-on-b,
17-off-b,16-on-b,16-off-b,14-on-b,14-off-b. Two final on-mode processes
exercise 14→14 and 257→14 on the same model. No OS cache drop or selective rerun
is used. Compatibility and lifetime probes are correctness controls, excluded
from performance statistics.

### Same-binary measurements

Times start after READY; startup is separate. Reads are physical per-process
request reads in decimal GB. Decode excludes the first sampled output.

| Run | Decision / path | TTFT s | Request s | Reads GB | Output tokens | Initial GPU experts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 14-off-a | automatic / flow | 24.40 | 32.27 | 15.01 | 6 | 1641 |
| 14-on-a | first_window / windows | 22.15 | 26.06 | 11.63 | 6 | 1641 |
| 16-off-a | automatic / flow | 21.57 | 27.79 | 12.20 | 6 | 1641 |
| 16-on-a | first_window / windows | 23.91 | 27.64 | 12.33 | 6 | 1641 |
| 17-off-a | automatic / flow | 15.32 | 21.47 | 9.05 | 6 | 1641 |
| 17-on-a | automatic / flow | 12.56 | 18.54 | 7.64 | 6 | 1641 |
| 257-off-a | automatic / flow | 87.03 | 109.11 | 48.52 | 32 | 1641 |
| 257-on-a | automatic / flow | 92.62 | 114.25 | 52.05 | 32 | 1641 |
| 257-on-b | automatic / flow | 95.37 | 116.93 | 52.37 | 32 | 1641 |
| 257-off-b | automatic / flow | 89.30 | 112.24 | 50.12 | 32 | 1641 |
| 17-on-b | automatic / flow | 22.79 | 30.15 | 13.68 | 6 | 1641 |
| 17-off-b | automatic / flow | 18.51 | 24.51 | 10.53 | 6 | 1641 |
| 16-on-b | first_window / windows | 24.01 | 27.74 | 12.32 | 6 | 1641 |
| 16-off-b | automatic / flow | 12.68 | 18.86 | 7.78 | 6 | 1641 |
| 14-on-b | first_window / windows | 22.14 | 26.08 | 11.63 | 6 | 1641 |
| 14-off-b | automatic / flow | 11.00 | 17.26 | 7.08 | 6 | 1641 |

Means are **off / on**, two runs per setting. Neither startup variation nor
short-request decode speed establishes sustained throughput.

| First prompt tokens | TTFT s | Request s | Reads GB | Decode tokens/s | Startup + request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 17.70 / 22.15 | 24.77 / 26.07 | 11.05 / 11.63 | 0.708 / 1.273 | 57.27 / 58.33 |
| 16 | 17.13 / 23.96 | 23.32 / 27.69 | 9.99 / 12.33 | 0.807 / 1.341 | 55.63 / 59.76 |
| 17 | 16.91 / 17.68 | 22.99 / 24.35 | 9.79 / 10.66 | 0.823 / 0.749 | 55.12 / 59.17 |
| 257 | 88.17 / 94.00 | 110.68 / 115.59 | 49.32 / 52.21 | 1.377 / 1.436 | 145.80 / 149.06 |

| Tokens | Pair | TTFT change | Request change | Read change |
| --- | --- | ---: | ---: | ---: |
| 14 | a | -9.23% | -19.22% | -22.50% |
| 14 | b | +101.28% | +51.06% | +64.20% |
| 16 | a | +10.83% | -0.54% | +1.10% |
| 16 | b | +89.33% | +47.07% | +58.35% |
| 17 | a | -18.02% | -13.63% | -15.60% |
| 17 | b | +23.17% | +23.02% | +29.94% |
| 257 | a | +6.42% | +4.71% | +7.27% |
| 257 | b | +6.80% | +4.17% | +4.48% |

With the same chunk workspace and 1641 initial GPU experts in every run, the 14-token mean TTFT changes from 17.702 to 22.146 s (+25.11%), request time 24.766 to 26.073 s (+5.28%), and physical reads 11.046 to 11.631 GB (+5.30%). Its two TTFT pairs disagree: -9.23% in forward order, +101.28% in reverse order. The 16-token mean TTFT changes from 17.126 to 23.958 s (+39.89%), request time 23.323 to 27.687 s (+18.71%), and reads 9.991 to 12.329 GB (+23.40%). Its TTFT regresses in both pairs (+10.83%/+89.33%). Faster short decode under windows does not recover the mean request cost; each short answer has only five measured decode steps, so this is not a sustained-throughput result.

Both switches delegate17/257 to the original chooser, which selects flow in all eight matrix controls. Even these unchanged-path controls vary:17-token TTFT changes -18.02%/+23.17% across the two pairs, and 257 changes +6.42%/+6.80%. Their reads also vary. These are observations of sequential process/IO-state variability, not evidence of an optimization caused by this guard. In particular,14-token automatic TTFT ranges 11.00–24.40 s while forced windows stay near 22.15 s;16-token automatic TTFT ranges 12.68–21.57 s while forced windows stay near 23.96 s. A fresh model does not establish identical cache or storage state. Cache-state influence is an inference supported by the read/timing variation, not an isolated proof of its exact source. Two runs per arm and related fixed prompts do not establish a universal threshold, statistical speedup, or general inferiority of native windows. They are sufficient to reject promoting this fixed guard on the current evidence.

### Compatibility, one-use state and quality

Default-off parent/candidate 14 outputs have identical token IDs. Default-off parent/candidate 257 outputs have identical token IDs. Their timings are not used to claim an optimization.

- `lifetime-short-repeat`: 14 tokens: first_window / windows → 14 tokens: automatic / flow.
- `lifetime-long-then-short`: 257 tokens: automatic / flow → 14 tokens: automatic / flow.

The short-repeat process forces only its first 14-token feed; its second 14-token feed uses the automatic picker. The long-then-short process uses the automatic picker for both 257 and 14: a successful large first feed has consumed the opportunity. Both automatic later decisions happen to select flow. These probes establish the intended same-model one-use behavior with fresh KV, not conversation-prefix or split-feed behavior.

All 24 requests reach EOS before their caps:17 correct one-sentence Chinese answers and 7 strict JSON objects with unique keys and all six exact values. Every Chinese response uses six tokens including EOS and every retrieval uses 32; output token IDs match within each matrix fixture and in both parent/candidate default-off comparisons. Raw answers are retained. These narrow checks do not supersede the broader quality limitations recorded in earlier gates.

### Resources, checks and preservation

Across all 22 processes, maximum observed GPU use is 19.510 GiB, maximum process HWM is 35.71 GB, and minimum host available memory is 17.38 GB. Engine swap and STORE read failures are zero; every run passes the fixed bounds. GPU observations exclude the capacity hold but include context/driver/other users. Memory uses one-second/boundary samples plus process HWM, not an instantaneous-peak or system-wide zero-swap claim.

Both real-header state suites pass under ASan/UBSan, including unchanged mixed
scheduling behavior and the first-window success/failure/retry/large-first
boundary. The 24 existing Python tests and 66 no-GPU controls pass. Controls
retain the original four-request fixtures, check exact 14/16/17/257 IDs and
caps, one-/two-request metadata, inherited switch isolation, resource guards
and invalid-combination rejection before engine startup.

The five previous native source trees/binaries remain unchanged; the sixth
candidate verifies all 1138 pinned source files with exactly two changes.
The three core archives, model/pack, production entry and 99 previous
runtime/header files are preserved. All 112 prior 7v evidence hashes are
verified, using the pre 7w snapshot for edited benchmark/roadmap files.
The main checkout is clean. No download, commit, push or PR is part of this gate.

Raw results are unchanged and adjacent answer reviews pin their SHA256 hashes.
`step7w-native-validation.json` records all 22 commands, calibration/decision
logs, flow/handoff/preload events and resource checks. `step7w-analysis.json`
separates the 16-run matrix from compatibility/lifetime controls. The gate-only
patch, source review, build/tests, complete answers and preservation checks
are indexed by `step7w-validation.json`.

A copied answer-review metadata label initially said 7v; it was corrected to 7w
and the initial reports are retained under `step7w-initial-review-metadata`.
No raw result or native measurement changed, and no GPU run was repeated.

**7w complete; user confirmed.** The default-off guard and all original measurements are preserved. The recommended cache-state audit follows in 7x.

## Step 7x: offline prefill cost and cache-state audit

**The saved traces locate the short-prefill variation in streamed read waits
and post-prompt cache settling. They do not justify a fixed token threshold
or an immediate correction to the chooser.**

This gate adds `tools/deepseek41/audit_prefill_costs.py`, an offline reader for
saved single-GPU benchmark results and their adjacent native logs. It audits
all 30 archived 7v/7w processes and 32 requests, retaining input hashes and
native request boundaries. It launches no engine and changes no model,
operator, native selector, benchmark entry or production default.

The primary comparison is the eight 7w matrix requests that use native flow
at 14/16/17 tokens. The four long matrix controls, 7v references, compatibility
probes and resident lifetime probes remain separate. These are analyses of
previous CUDA/SM75 measurements on the RTX 2080 Ti with the same 20 GiB GPU
bound, 40 decimal GB RAM store, 18 CPU workers and SATA SSD. There are no new
performance measurements or backend-build claims in this gate.

### What explains the recorded variation

| Run | Estimated RAM % | Direct / buffered payload GB | Native prefill s | Stream s | Read wait within stream s | Inclusive settle s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14-off-a | 9 | 8.8 / 1.2 | 24.402 | 18.380 | 15.612 | 6.001 |
| 16-off-a | 12 | 7.4 / 3.5 | 21.572 | 15.580 | 12.595 | 5.973 |
| 17-off-a | 14 | 5.5 / 5.7 | 15.317 | 11.921 | 8.817 | 3.375 |
| 17-on-a | 15 | 5.0 / 6.2 | 12.557 | 10.820 | 7.710 | 1.720 |
| 17-on-b | 13 | 8.0 / 3.2 | 22.792 | 16.774 | 13.516 | 6.001 |
| 17-off-b | 15 | 6.1 / 5.0 | 18.504 | 13.174 | 9.926 | 5.305 |
| 16-off-b | 17 | 4.7 / 6.2 | 12.677 | 10.318 | 7.386 | 2.334 |
| 14-off-b | 17 | 3.9 / 6.1 | 10.999 | 8.665 | 6.019 | 2.306 |

Stream plus inclusive settle accounts for **99.746–99.926%** of native
prefill; the remaining 16.8–27.9 ms covers other work and timing boundaries.
Read waits occupy 69.46–84.94% of stream wall time. They are already inside
stream time, as is the handoff work. Adding these nested counters again would
overstate elapsed time.

| Same-fixture difference | Prefill difference s | Stream difference s | Of that, read-wait difference s | Settle difference s |
| --- | ---: | ---: | ---: | ---: |
| 14-off-a minus 14-off-b | 13.403 | 9.715 | 9.593 | 3.695 |
| 16-off-a minus 16-off-b | 8.895 | 5.262 | 5.209 | 3.639 |
| 17-off-a minus 17-on-a | 2.760 | 1.101 | 1.107 | 1.655 |
| 17-on-b minus 17-off-b | 4.288 | 3.600 | 3.590 | 0.696 |

For 14 tokens, 9.715 s of the 13.403 s prefill difference is in stream
execution, including 9.593 s of read-wait difference. Another 3.695 s is in
settling; the remaining difference is -7.3 ms. For 16 tokens, the corresponding
stream/settle differences are 5.262/3.639 s. These are additive observations
between separate process states, not a controlled causal estimate of a
particular cache mechanism. In these samples the large change is not in the
non-wait portion of stream execution.

The actual file-expert counts stay identical within each short fixture:
1349, 1470 and 1502 for 14, 16 and 17 tokens. Their file payloads remain
10.0, 10.9 and 11.2 GB. The IO path mix and wait time change. This supports
investigating cache/IO conditions rather than attributing the variation to a
different amount of expert computation. In 7v the two short auto references
both spent about 6 s settling and sent 8.3/8.9 GB through direct reads; in
7w's faster 14-off-b run those figures are 2.306 s and 3.9 GB. A new model
process alone does not establish the same storage/cache state.

### Prediction and measurement have different meanings

| Prompt tokens | Predicted expert demand GB | Observed file payload GB | Descriptive ratio |
| --- | ---: | ---: | ---: |
| 14 | 20.2 | 10.0 | 2.02 |
| 16 | 22.7 | 10.9 | 2.08 |
| 17 | 23.9 | 11.2 | 2.13 |

These first-feed fixtures use one chunk, no speculative experts and no
RAM-store experts in the stream. The initial demand estimate is about twice
the observed payload. Source review shows `m=1` and initial cost factors of 1;
the profile's CPU-lane probability mass is spread across non-VRAM experts
and passed through a saturating exponential. The spread is learned after a
single-chunk flow. This documents a bias on these related prompts, not a
universal multiplier to apply to every model request.

The chooser's 9–17% residency estimates and the IO queue's 12–61% buffered
payload shares are **different populations and times**. The chooser samples
non-VRAM experts before routing, with stride `max(1, nc/32)` (approximately
32 samples, not a strict cap). It samples two pages per matrix and requires
all three matrices resident, unless the expert is already in the RAM store.
The IO queue later checks the actual per-matrix job's full page range.
Buffered reads also cover unavailable or failed direct reads. The resulting
percentage gap is not an observed `mincore` false-negative rate or an exact
cache-hit rate.

The benchmark's process `read_bytes` spans GEN through DONE: prefill,
settling and decode. No prefill-only physical IO snapshot exists in these
records. The audit leaves `prefill_physical_read_bytes` and
`exact_prefill_cache_hit_pct` null instead of subtracting STORE payload from
whole-request reads or treating buffered payload as verified hits. GB figures
in the native stream log are rounded to one decimal.

Long requests also use two chunks, 3382 layer-cache reuses and 307 unpicked
speculative experts. Resident later requests have changed GPU/RAM placement
and learned state. Their demand/payload ratios are not pooled with the first
short-feed comparison.

### A bounded next experiment

All eight primary short-flow requests report **zero RAM preloads**, yet
inclusive cache settling takes **1.720–6.001 s**, or 13.70–28.67% of prefill.
The four long controls report 6.981–7.213 s of inclusive settling, including
0.979–1.212 s of RAM preload. Subtracting that nested preload leaves about
6 s of swap-loop/other settling work.

`cache_settle()` gives the existing swap loop a default `DS4_SETTLE_MS=6000`
budget and may exit early when no moves remain. RAM preload follows that
loop, inside the same logged total. It is not an unconditional six-second
sleep or a hard cap on the combined stages. Crucially, a non-positive budget
returns before both swaps and RAM preload. Setting it to zero would not
isolate the swap budget from the preload path.

The next proposed gate compares **positive 1000 ms versus explicit 6000 ms**
using the existing native switch. Use fresh 14-token and 257-token requests,
two runs per setting/fixture in both orders: eight processes total. Keep
flow-store on, RAM preload enabled, mixed-prefill off, first-window off,
chunk256 and the existing memory bounds. Review full request time, decode,
TTFT, actual swaps/preloads/reads, answers and resource limits together.
A smaller budget can alter expert placement and preload membership, so a
first-token reduction alone would not establish a net gain. No such trial
has started, and no selector or default changes are authorized by this audit.

### Verification and preservation

The eight focused offline tests pass. They cover repeated-request boundaries,
nested timing, multi-chunk reuse, absent optional data, malformed/ambiguous
traces, non-finite measurements, input hashes and refusal to overwrite
reports. Every archived trace is parsed; incomplete or conflicting diagnostics
are rejected. Unknown optional timings remain unknown, not zero.

The six native source trees/binaries, production entry, benchmark, model/pack
and 99 previous runtime/header files remain unchanged. All 272 prior 7w and
112 prior 7v evidence hashes are checked with the appropriate pre-edit
snapshots for the roadmap/benchmark. There is no new GPU run, compilation,
download, commit, push or PR. The main checkout remains clean.

An initial offline summary helper consumed a generator before computing its
maximum. Its script and error log are retained in
`step7x-initial-analysis-error`; materializing the range values fixed it.
Raw evidence and the audit parser were unaffected.

`step7x-trace-audit.json` preserves every parsed request and its input hashes;
`step7x-analysis.json` records the role-separated calculations.
`step7x-source-review.json` pins source hashes and exact lines for six findings.
The scoped patch, tests, preservation checks and evidence index are recorded
in `step7x-validation.json`.

**7x complete; user confirmed.** Its offline attribution remains preserved; the proposed positive-budget comparison is measured in 7y below.

## Step 7y: positive native cache-settle budget comparison

**The 1000 ms budget improves fresh short-request latency, with a decode tradeoff; long-request gains are inconsistent. Keep it experimental.**

This gate adds benchmark-only `--settle-ms 1000|6000`, selecting the existing
native `DS4_SETTLE_MS`. It requires a pinned first-request resident probe with
short-first/long-first order, chunk 256, flow-store on, RAM preload explicitly
on, mixed-prefill off and first-window off. The argument is omitted by default;
inherited native budget/selector overrides are stripped. Unsupported budgets,
including zero, and incompatible combinations are rejected before startup.
The production entry, native operators, selector and defaults remain unchanged.

Both arms use the exact previously built CUDA12.8/SM75 binary,
`build-deepseek-first-window-sm75/strata-dsv41`, SHA256
`a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96`.
No native rebuild is needed; HIP/SYCL execution is not newly validated.
The RTX 2080 Ti keeps the 20 GiB target; the Xeon E5-2686v4 uses 18 CPU workers,
40 decimal GB LRU/4595 slots, an 8 GiB host-available floor, context 1024,
temperature 0 and DSpark none on the same 64 GB-class RAM/SATA system.

Only the explicit positive budget differs between the two native environments.
RAM preload stays enabled after the swap loop. The original automatic chooser
is allowed to select either native path, and its actual choice is recorded.
The budget applies to flow's swap-loop condition, not to an unconditional sleep
or the complete settle duration. RAM preload follows and is included in the
settle log; zero would bypass both stages and is excluded from this comparison.
A smaller positive budget can still alter expert placement and preload membership.

Eight processes each receive one fresh-KV request: unchanged 14-token Chinese
QA/cap64 or 257-token retrieval/cap128. The order was fixed before measurement:
14-6000-a,14-1000-a,257-6000-a,257-1000-a,257-1000-b,257-6000-b,14-1000-b,
14-6000-b. Each fixture is compared in both execution orders. No OS cache drop
or selective rerun is used. Previous gates provide pinned commands/fixtures;
performance comparisons below use only these same-day measurements.

### Whole-request measurements

TTFT and request time start after READY; startup is separate. Decode excludes
the first sampled output. Physical reads cover the complete request.

| Run | Path | Startup s | TTFT s | Request s | Reads GB | Decode tokens/s | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14-6000-a | flow | 38.11 | 21.60 | 29.13 | 13.32 | 0.664 | 6 |
| 14-1000-a | flow | 31.97 | 12.19 | 19.35 | 8.34 | 0.698 | 6 |
| 257-6000-a | flow | 32.04 | 93.51 | 115.59 | 53.13 | 1.404 | 32 |
| 257-1000-a | flow | 38.75 | 96.08 | 117.62 | 53.53 | 1.440 | 32 |
| 257-1000-b | flow | 35.18 | 90.83 | 111.31 | 49.91 | 1.514 | 32 |
| 257-6000-b | flow | 32.22 | 96.81 | 119.66 | 52.82 | 1.356 | 32 |
| 14-1000-b | flow | 35.56 | 17.65 | 26.62 | 12.12 | 0.557 | 6 |
| 14-6000-b | flow | 33.10 | 21.65 | 28.83 | 13.05 | 0.697 | 6 |

Means are **6000 ms / 1000 ms**, two requests per setting/fixture. Decode rates
are pooled by actual decode steps and elapsed native decode time.

| Input tokens | TTFT s | Request s | Reads GB | Decode tokens/s | Startup + request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 21.62 / 14.92 | 28.98 / 22.98 | 13.19 / 10.23 | 0.680 / 0.620 | 64.59 / 56.75 |
| 257 | 95.16 / 93.46 | 117.63 / 114.46 | 52.97 / 51.72 | 1.380 / 1.476 | 149.75 / 151.43 |

| Input tokens | Pair | TTFT change | Request change | Read change |
| --- | --- | ---: | ---: | ---: |
| 14 | a | -43.56% | -33.58% | -37.43% |
| 14 | b | -18.48% | -7.65% | -7.15% |
| 257 | a | +2.76% | +1.75% | +0.76% |
| 257 | b | -6.17% | -6.98% | -5.51% |

For the 14-token fixture, 1000 ms lowers mean TTFT from 21.62 to 14.92 s
(-31.00%) and mean request time from 28.98 to 22.98 s (-20.68%). Both
execution orders improve request time (-33.58% and -7.65%). However, pooled
decode falls from 0.680 to 0.620 tokens/s (-8.81%); pair b falls from 0.697
to 0.557 tokens/s (-20.00%). Each short answer has only five decode steps.

For the 257-token fixture, mean TTFT changes from 95.16 to 93.46 s (-1.79%)
and request time from 117.63 to 114.46 s (-2.69%). Pair a regresses by 1.75%
in request time while pair b improves by 6.98%. Pooled decode changes from
1.380 to 1.476 tokens/s (+6.98%). Startup plus request also remains mixed:
short means change from 64.59 to 56.75 s, while long means change from
149.75 to 151.43 s (+1.12%). Startup is not part of the post-READY TTFT.

These are observed results from two requests per setting/fixture, not a
stable general speedup. Mean improvements alone do not justify changing
the production default or introducing a token-count threshold.

### Native stage attribution

| Run | Stream s | Read wait within stream s | Inclusive settle s | Swaps during settle | RAM preload experts / ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14-6000-a | 15.576 | 12.820 | 6.001 | 394 | 0 / 0 |
| 14-1000-a | 11.170 | 8.418 | 1.000 | 401 | 0 / 0 |
| 257-6000-a | 86.467 | 57.962 | 6.889 | 331 | 54 / 888 |
| 257-1000-a | 94.031 | 65.437 | 1.922 | 47 | 52 / 921 |
| 257-1000-b | 88.824 | 59.997 | 1.865 | 145 | 51 / 863 |
| 257-6000-b | 89.630 | 60.598 | 7.046 | 602 | 58 / 1045 |
| 14-1000-b | 16.629 | 13.841 | 1.001 | 75 | 0 / 0 |
| 14-6000-b | 15.628 | 12.798 | 6.001 | 441 | 0 / 0 |

All eight automatic decisions select flow, with 1641 initially GPU-resident
experts. The positive budget reduces the observed inclusive settle stage:
short means are 6.001 / 1.0005 s and long means are 6.9675 / 1.8935 s
(6000 / 1000 ms). Short RAM preload remains zero; long RAM preload remains
active, averaging 56 / 51.5 experts and 966.5 / 892 ms. This confirms that
the positive budget retains the preload path; it does not preserve identical
expert placement or preload membership.

Other phase timings vary. In short pair a, stream is 4.406 s faster for
1000 ms, in addition to 5.001 s less settling. In short pair b, stream is
1.001 s slower and native decode is 1.794 s slower, partially consuming
the 5.000 s settle reduction. In long pair a, stream is 7.564 s slower,
outweighing 4.967 s less settling. Long pair b has 0.806 s less stream
and 5.181 s less inclusive settling. The full TTFT/request differences
therefore cannot all be attributed to the budget itself.

Stream already includes read waits and handoff work. The cache-settle log
includes RAM preload; these nested times must not be added twice. Native
payload GB and buffered IO classification are different from whole-request
process reads. These traces do not contain exact prefill-only physical IO or
cache-hit rates. Sequential runs retain OS cache and can differ in calibration
and IO state despite equal commands, workspace and resource limits.

### Quality, resources and verification

All eight answers pass content, requested-format and truncation review:
four Chinese factual answers and four strict JSON retrieval answers. Each
fixture has identical output token IDs across both settings and both orders.
Short answers contain six tokens including EOS; long answers contain 32.
All stop at EOS before their respective 64/128-token caps. This is a check
of these two fixtures, not a broader model-quality assessment.

All eight processes satisfy the resource bounds: maximum observed GPU use 19.510 GiB, maximum process HWM 33.03 GB, minimum host available 19.57 GB. Engine swap and STORE read failures are zero. Memory uses one-second/boundary samples plus process HWM. GPU excludes the capacity hold but includes context/driver/other users; engine swap is not system-wide swap.

The 24 existing Python regressions and 89 no-GPU controls pass. The controls
retain all 66 earlier option/sequence checks, add explicit/omitted-budget
command and environment checks for both fixtures, reject 16 incompatible
combinations, and preserve the low-host-memory rejection. The sole candidate
binary is pinned throughout; fixtures, output caps, resource bounds and
inherited-switch isolation are checked before and after each native run.

All six native source trees/binaries, three core archives, model/pack,
production entry and 99 previous runtime/header files remain unchanged.
Prior evidence from 7x/7w/7v is verified using the corresponding pre-edit
snapshots where benchmark/roadmap files changed. The main checkout remains
clean. No download, commit, push or PR is part of this gate.

Raw results remain unchanged; adjacent answer reviews pin their hashes.
`step7y-comparison.json` records all commands and native traces;
`step7y-analysis.json` retains both orders, output-ID groups, stage attribution
and resource observations. The gate-only patch, controls, reviews and
preservation checks are indexed in `step7y-validation.json`.

**7y complete; user confirmed.** Its fresh-request measurements remain preserved; the proposed resident comparison is measured in 7z below.

## Step 7z: resident positive cache-settle budget comparison

**The 1000 ms budget lowers four-request time in both orders, with repeat-task decode and final-short regressions. Retain it as an experimental candidate.**

The benchmark's existing `--settle-ms 1000|6000` now also accepts the
existing four-request short-first/long-first resident sequences. The pinned
engine, chunk 256, flow-store on, RAM preload on, mixed-prefill off and
first-window off remain required. First-request-only probes keep their 7y
behavior, and omitted-budget invocations retain their prior commands and
defaults. No native operator, production entry or default is changed.

Both arms reuse the exact prior CUDA 12.8/SM75 engine,
`build-deepseek-first-window-sm75/strata-dsv41`, SHA256
`a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96`.
No rebuild is needed; HIP/SYCL execution is not newly validated. Hardware is
the same RTX 2080 Ti, Xeon E5-2686v4/18 workers, 64 GB-class RAM and SATA SSD.
Bounds remain 20 GiB GPU, 40 decimal GB LRU/4595 RAM-store slots, 8 GiB host
available, context 1024, temperature 0 and DSpark none.

Each process loads once and answers four independent questions. Short-first
is short14, long257, repeat short14, repeat long257; long-first reverses the
task pair and then repeats it. Caps remain 64/128 tokens respectively. Every
request starts with fresh KV; expert caches, routing profile and OS state
persist. The native chooser is automatic on every request. A window-selected
request does not exercise flow's settle budget, and absent flow-stage timings
are retained as unknown. Positive budget bounds the swap-loop condition;
RAM preload follows and is included in settle time.

The eight-process/32-request plan was fixed before measurement:
short-6000-a, short-1000-a, long-6000-a, long-1000-a, long-1000-b,
long-6000-b, short-1000-b, short-6000-b. Each request order therefore has
both arm execution orders. No cache drop, selective rerun or new model
download is used. Prior runs pin fixtures and commands; performance below
compares only the new same-day arms.

### Session measurements

Active time sums the four GEN-to-DONE request times after READY. Session
span runs from first GEN to last DONE and includes harness gaps. Startup is
reported separately. Physical reads sum the four complete-request process
read counters, excluding model startup. Decode pools actual decode steps
and native decode time, excluding each request's first sampled output.

| Run | Paths in request order | Startup s | Active four-request s | Session span s | Reads GB | Pooled decode tokens/s | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-6000-a | flow / flow / flow / flow | 32.20 | 202.35 | 202.36 | 92.00 | 2.256 | 76 |
| short-1000-a | flow / flow / flow / flow | 41.80 | 188.75 | 188.76 | 86.39 | 2.154 | 76 |
| long-6000-a | flow / flow / flow / flow | 41.39 | 214.17 | 214.18 | 96.76 | 2.369 | 76 |
| long-1000-a | flow / flow / flow / flow | 43.46 | 198.34 | 198.35 | 89.21 | 2.336 | 76 |
| long-1000-b | flow / flow / flow / flow | 38.79 | 191.96 | 191.98 | 84.86 | 2.501 | 76 |
| long-6000-b | flow / flow / flow / flow | 37.38 | 211.98 | 211.99 | 95.82 | 2.382 | 76 |
| short-1000-b | flow / flow / flow / flow | 43.31 | 180.58 | 180.59 | 82.79 | 2.113 | 76 |
| short-6000-b | flow / flow / flow / flow | 35.28 | 219.11 | 219.12 | 97.46 | 2.137 | 76 |

Means are **6000 ms / 1000 ms**, two sessions per setting/order.

| Request order | Active four-request s | Session span s | Startup + span s | Reads GB | Pooled decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-first | 210.73 / 184.66 | 210.74 / 184.67 | 244.48 / 227.23 | 94.73 / 84.59 | 2.195 / 2.133 |
| long-first | 213.08 / 195.15 | 213.09 / 195.16 | 252.47 / 236.29 | 96.29 / 87.03 | 2.375 / 2.416 |

| Request order | Pair | Active time change | Read change | Decode change |
| --- | --- | ---: | ---: | ---: |
| short-first | a | -6.72% | -6.10% | -4.52% |
| short-first | b | -17.58% | -15.05% | -1.12% |
| long-first | a | -7.39% | -7.80% | -1.38% |
| long-first | b | -9.45% | -11.44% | +5.02% |

With 1000 ms, mean four-request active time changes from 210.73 to 184.66 s
(-12.37%) for short-first and from 213.08 to 195.15 s (-8.41%) for
long-first. All four paired comparisons improve: short-first a/b are
-6.72%/-17.58%, and long-first a/b are -7.39%/-9.45%. The complete-request
physical-read means decrease from 94.73 to 84.59 GB (-10.70%) and from
96.29 to 87.03 GB (-9.61%) respectively.

Startup remains variable and separate. Mean startup plus session span changes
from 244.48 to 227.23 s (-7.06%) for short-first and from 252.47 to
236.29 s (-6.41%) for long-first. Pooled decode across each complete session
changes from 2.195 to 2.133 tokens/s (-2.81%) and from 2.375 to 2.416
(+1.71%). Per-pair decode changes are mixed and cannot establish a general
throughput improvement.

Each setting/order has only two sessions. These observed gains cover this
fixed four-request task multiset, with resident expert/OS caches and fresh
KV, not sustained throughput or general workloads. Calibration and IO state
vary across sequential runs, so the entire latency/read reduction cannot
be causally assigned to the budget alone.

### First, later and repeated requests

Each cell is **6000 ms / 1000 ms**, retaining request position and task.

| Order | Position / task | TTFT s | Request s | Reads GB | Pooled decode tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| short-first | 1: zh_fact | 20.35 / 16.78 | 27.23 / 25.28 | 12.25 / 11.67 | 0.726 / 0.588 |
| short-first | 2: long_retrieval | 88.15 / 80.65 | 109.06 / 100.35 | 49.29 / 45.98 | 1.483 / 1.574 |
| short-first | 3: zh_fact_repeat | 12.91 / 6.85 | 13.64 / 7.54 | 5.87 / 3.22 | 6.872 / 7.201 |
| short-first | 4: long_retrieval_repeat | 56.53 / 46.64 | 60.80 / 51.49 | 27.33 / 23.72 | 7.254 / 6.391 |
| long-first | 1: long_retrieval | 97.30 / 91.00 | 118.91 / 111.65 | 53.33 / 50.57 | 1.434 / 1.501 |
| long-first | 2: zh_fact | 15.34 / 9.53 | 18.93 / 12.99 | 8.48 / 5.64 | 1.393 / 1.446 |
| long-first | 3: long_retrieval_repeat | 58.26 / 51.60 | 62.82 / 56.68 | 29.03 / 24.71 | 6.809 / 6.112 |
| long-first | 4: zh_fact_repeat | 11.86 / 13.22 | 12.41 / 13.83 | 5.45 / 6.12 | 9.024 / 8.073 |

Short-first benefits in aggregate despite a slower initial-short decode:
0.726 to 0.588 tokens/s (-18.99%). Its repeat-short request improves from
13.64 to 7.54 s (-44.68%), but repeat-long decode falls from 7.254 to
6.391 tokens/s (-11.90%). Repeat-long full request time still decreases
from 60.80 to 51.49 s because of the lower prefill/settle cost.

Long-first also has a concrete later-request regression. The final repeated
short request increases from 12.41 to 13.83 s (+11.45%), regressing in both
pairs. Its decode falls from 9.024 to 8.073 tokens/s (-10.54%). Repeated-long
decode falls from 6.809 to 6.112 tokens/s (-10.24%), although its full
request improves from 62.82 to 56.68 s (-9.78%). Pooling just the later
requests gives 4.714 versus 4.481 tokens/s, about 4.95% lower for 1000 ms.
These position-specific costs must accompany the improved total session
time; neither repeated-task nor later-request decode is uniformly better.

### Native phases and cache pressure

| Run / request | Path | Stream s | Inclusive settle s | Settle swaps | RAM preload experts / ms | STORE slots prefill / decode | STORE evictions prefill / decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-6000-a / 1 | flow | 12.072 | 4.547 | 582 | 0 / 0.0 | 256 / 640 | 0 / 0 |
| short-6000-a / 2 | flow | 77.490 | 7.363 | 543 | 82 / 1362.0 | 3264 / 4288 | 0 / 0 |
| short-6000-a / 3 | flow | 6.215 | 7.133 | 224 | 63 / 1128.0 | 4416 / 4416 | 0 / 0 |
| short-6000-a / 4 | flow | 41.824 | 13.540 | 181 | 552 / 8767.0 | 4595 / 4595 | 552 / 92 |
| short-1000-a / 1 | flow | 16.226 | 1.001 | 149 | 0 / 0.0 | 256 / 768 | 0 / 0 |
| short-1000-a / 2 | flow | 83.051 | 1.021 | 63 | 1 / 20.0 | 3392 / 4288 | 0 / 0 |
| short-1000-a / 3 | flow | 5.461 | 1.344 | 62 | 20 / 342.0 | 4352 / 4352 | 0 / 0 |
| short-1000-a / 4 | flow | 39.168 | 7.806 | 46 | 422 / 6805.0 | 4595 / 4595 | 422 / 94 |
| long-6000-a / 1 | flow | 90.836 | 6.903 | 547 | 56 / 902.0 | 3008 / 4096 | 0 / 0 |
| long-6000-a / 2 | flow | 8.513 | 6.851 | 252 | 50 / 849.0 | 4224 / 4416 | 0 / 0 |
| long-6000-a / 3 | flow | 41.963 | 16.128 | 231 | 625 / 10127.0 | 4595 / 4595 | 625 / 99 |
| long-6000-a / 4 | flow | 4.809 | 7.511 | 111 | 285 / 4509.0 | 4595 / 4595 | 285 / 7 |
| long-1000-a / 1 | flow | 89.430 | 1.961 | 42 | 52 / 960.0 | 3008 / 4032 | 0 / 0 |
| long-1000-a / 2 | flow | 8.365 | 1.119 | 44 | 7 / 118.0 | 4096 / 4288 | 0 / 0 |
| long-1000-a / 3 | flow | 42.992 | 9.217 | 44 | 506 / 8215.0 | 4595 / 4595 | 506 / 85 |
| long-1000-a / 4 | flow | 4.959 | 9.216 | 43 | 503 / 8215.0 | 4595 / 4595 | 503 / 3 |
| long-1000-b / 1 | flow | 88.472 | 1.821 | 65 | 51 / 821.0 | 3008 / 4032 | 0 / 0 |
| long-1000-b / 2 | flow | 8.163 | 1.391 | 222 | 25 / 390.0 | 4160 / 4288 | 0 / 0 |
| long-1000-b / 3 | flow | 42.718 | 8.146 | 61 | 442 / 7145.0 | 4595 / 4595 | 442 / 95 |
| long-1000-b / 4 | flow | 5.009 | 7.216 | 35 | 378 / 6215.0 | 4595 / 4595 | 378 / 4 |
| long-6000-b / 1 | flow | 89.584 | 6.968 | 591 | 58 / 967.0 | 3008 / 4096 | 0 / 0 |
| long-6000-b / 2 | flow | 8.464 | 6.824 | 241 | 48 / 822.0 | 4224 / 4416 | 0 / 0 |
| long-6000-b / 3 | flow | 41.807 | 16.472 | 232 | 645 / 10470.0 | 4595 / 4595 | 645 / 99 |
| long-6000-b / 4 | flow | 4.859 | 6.511 | 87 | 257 / 4138.0 | 4595 / 4595 | 257 / 6 |
| short-1000-b / 1 | flow | 15.276 | 1.001 | 95 | 0 / 0.0 | 256 / 768 | 0 / 0 |
| short-1000-b / 2 | flow | 75.932 | 1.033 | 46 | 2 / 33.0 | 3392 / 4352 | 0 / 0 |
| short-1000-b / 3 | flow | 5.509 | 1.351 | 66 | 21 / 350.0 | 4416 / 4416 | 0 / 0 |
| short-1000-b / 4 | flow | 37.811 | 8.361 | 40 | 458 / 7359.0 | 4595 / 4595 | 458 / 82 |
| short-6000-b / 1 | flow | 18.031 | 6.000 | 354 | 0 / 0.0 | 256 / 704 | 0 / 0 |
| short-6000-b / 2 | flow | 84.358 | 6.830 | 427 | 49 / 828.0 | 3328 / 4352 | 0 / 0 |
| short-6000-b / 3 | flow | 5.614 | 6.826 | 219 | 46 / 826.0 | 4416 / 4480 | 0 / 0 |
| short-6000-b / 4 | flow | 41.718 | 15.814 | 231 | 575 / 9812.0 | 4595 / 4595 | 575 / 91 |

All 32 native choices select flow. Reducing the swap-loop budget does not
bound the full cache-settle stage: the 1000 ms arm records 1.001 to 9.217 s
of inclusive settling, while RAM preload across all requests ranges from
0 to 10.470 s. Preload stays enabled and its membership changes.

For long-first's final repeated short request, mean stream time increases
from 4.834 to 4.984 s and inclusive settling from 7.011 to 8.216 s. Their
combined 1.355 s increase accounts for the observed TTFT regression. Mean
RAM preload grows from 4.324 to 7.215 s despite the shorter swap loop.
This is measured phase attribution; the responsible placement/selection
changes have not been isolated into a separate causal experiment.

The RAM store reaches its 4595-slot ceiling. There are 6405 evictions summed
across the 64 phase snapshots from 32 requests, with no read failure. Short-first
6000/1000 ms arms record 1310/1056 evictions and 3007/3020 decode misses;
long-first records 2023/2016 evictions and 2689/2521 decode misses. Smaller
counters do not by themselves imply faster decode.

All 32 flow-prefill STORE hit and miss counters are zero despite nonzero
stream/file/process reads. Their hit ratio is undefined; zero denominators
must not be interpreted as no reads or a perfect cache. These counters
cannot describe the full streaming IO path on their own.

Full rows and phase counters remain in `step7z-analysis.json`. Stream already
contains read waits and handoff; inclusive settle includes RAM preload.
These nested times cannot be added twice. STORE hit/(hit+miss) excludes
prefetch/in-flight accesses; submitted bytes are queued payload. Neither
is an exact prefill-only physical read or cache-hit measurement. Repeated
requests also change routing profile, placement and OS cache, so their
speedup cannot be attributed to RAM store alone.

### Quality, resources and verification

All 32 answers pass content, requested-format and truncation review:
16 Chinese capital answers and 16 strict unique-key JSON retrieval answers.
Both tasks have identical output token IDs across settings, orders and
repeat positions. Short answers have six tokens including EOS and long
answers have 32; every four-request session has 76 output tokens. All stop
at EOS before the 64/128-token caps. This validates these two fixtures only,
not broader model quality or longer-context behavior.

Maximum observed GPU use is 19.510 GiB; maximum process HWM is 37.30 GB; minimum host available is 15.32 GB. All process bounds pass. Engine swap and STORE read failures are zero. Memory uses one-second/boundary samples plus process HWM; GPU excludes capacity hold but includes context/other users. Engine swap is not system-wide swap.

The 24 existing Python regressions and 99 no-GPU controls pass. Controls
retain 88 earlier checks; the obsolete no-first-request rejection is replaced
by 11 resident checks covering exact four-request fixtures, command/env
compatibility, forbidden orders and low host memory. All new native requests
are matched to immutable fixture/cap references, and commands/environment,
actual decisions, per-phase STORE counters and capacity bounds are checked.

All six native source trees/binaries, three core archives, eight pack files,
production entry and 99 existing runtime/header files are preserved. Prior
7y/7x/7w/7v evidence is checked using pre-edit snapshots where needed. Weight
verification reuses prior full hashes plus per-run sizes/.done markers; no
full 231 GB rehash is claimed. The main checkout remains clean.

Raw results are preserved with adjacent hash-pinned answer reviews. The
fixed-order comparison, per-position analysis, output-ID groups, all tool
failures, gate-only patch and preservation results are indexed in
`step7z-validation.json`. No commit, push or PR is part of this gate.

**7z complete; user confirmed.** Its 1000/6000 ms resident measurements remain preserved; the proposed production comparison is measured in 7aa below.

## Step 7aa: production windows versus the 1000 ms flow candidate

**The candidate lowers the tested resident-session time by 23–28%, with short-answer regressions. It merits an opt-in entry, not default replacement.**

This gate uses the existing benchmark without changing its source. It
compares the current daily-entry production windows profile against the
experimental chunk 256/flow-store/1000 ms profile. Both load once per process
and run the same independent short14/cap64 and long257/cap128 tasks plus
repeats, with fresh KV and resident expert/routing/OS caches. Short-first is
short, long, repeat short, repeat long; long-first reverses each pair.

Production uses `build-deepseek-diagnostics-sm75/strata-dsv41`, SHA256
`86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4`.
The candidate uses `build-deepseek-first-window-sm75/strata-dsv41`, SHA256
`a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96`.
Both exact CUDA 12.8/SM75 binaries are reused without rebuilding. Native
source, benchmark, production entry and defaults remain unchanged.
HIP/SYCL execution is not newly validated.

The RTX 2080 Ti retains the 20 GiB target, with 18 Xeon E5-2686v4 workers,
64 GB-class RAM/SATA SSD, 40 decimal GB LRU/4595 slots, an 8 GiB host floor,
context 1024, temperature 0 and DSpark none. Mocked preflight captures the
actual `run_quantized.py` native command and defaults. The windows control
adds explicit equivalent LRU and common verbose/flow diagnostics. Both
profiles retain STORE diagnostics. Inherited selector/budget switches are
stripped; only the candidate explicitly enables flow-store, RAM preload,
1000 ms swap budget, mixed-prefill off and first-window off.

This compares complete configurations. Native build, windows0/chunk256
workspace, initial GPU expert capacity, routing and cache-settle behavior
intentionally differ. It does not isolate the budget or kernel speed.
Candidate decisions remain automatic. Positive budget bounds only the swap
loop condition; RAM preload follows and is included in settle time. Missing
flow-only timings on windows requests are unknown, not zero.

The fixed plan is short-window-a, short-flow-a, long-window-a, long-flow-a,
long-flow-b, long-window-b, short-flow-b, short-window-b: eight native
processes and 32 requests, both request orders and both profile execution
orders. No OS cache drop, selective rerun or new download is used. Earlier
runs pin fixtures and commands; timing comparisons use only these new arms.

### Session measurements

Active time sums four GEN-to-DONE request times after READY. Session span
runs from first GEN to last DONE and includes harness gaps. Startup is
separate. Process physical reads cover complete requests, excluding startup.
Decode pools actual decode steps/native decode time and excludes the first
sampled output from each request. Output lengths accompany each comparison.

| Run | GPU experts initially | Paths in request order | Startup s | Four-request active s | Reads GB | Pooled decode tokens/s | Output tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| short-window-a | 1729 | windows / windows / windows / windows | 42.71 | 264.56 | 105.11 | 1.757 | 76 |
| short-flow-a | 1641 | flow / flow / flow / flow | 43.60 | 212.26 | 85.73 | 2.154 | 76 |
| long-window-a | 1729 | windows / windows / windows / windows | 38.08 | 262.67 | 101.64 | 1.952 | 76 |
| long-flow-a | 1641 | flow / flow / flow / flow | 38.80 | 198.11 | 88.76 | 2.417 | 76 |
| long-flow-b | 1641 | flow / flow / flow / flow | 38.02 | 189.30 | 84.69 | 2.474 | 76 |
| long-window-b | 1729 | windows / windows / windows / windows | 38.45 | 272.23 | 105.60 | 1.845 | 76 |
| short-flow-b | 1641 | flow / flow / flow / flow | 38.77 | 188.81 | 85.29 | 2.157 | 76 |
| short-window-b | 1729 | windows / windows / windows / windows | 36.53 | 259.64 | 101.22 | 1.849 | 76 |

Means are **production windows / 1000 ms flow candidate**, two sessions per
setting/order. These labels name the configurations; actual paths are above.

| Request order | Four-request active s | Session span s | Startup + span s | Reads GB | Pooled decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-first | 262.10 / 200.53 | 262.11 / 200.54 | 301.73 / 241.73 | 103.16 / 85.51 | 1.802 / 2.155 |
| long-first | 267.45 / 193.70 | 267.46 / 193.71 | 305.73 / 232.13 | 103.62 / 86.72 | 1.897 / 2.445 |

| Request order | Pair | Active time change | Physical-read change | Decode change |
| --- | --- | ---: | ---: | ---: |
| short-first | a | -19.77% | -18.44% | +22.57% |
| short-first | b | -27.28% | -15.73% | +16.63% |
| long-first | a | -24.58% | -12.68% | +23.82% |
| long-first | b | -30.46% | -19.80% | +34.07% |

Mean four-request active time changes from 262.10 to 200.53 s (-23.49%)
for short-first and from 267.45 to 193.70 s (-27.57%) for long-first.
Every predeclared pair improves: short-first a/b are -19.77%/-27.28%,
and long-first a/b are -24.58%/-30.46%. Including startup plus session
span gives 301.73 to 241.73 s (-19.89%) and 305.73 to 232.13 s (-24.07%).

Complete-request physical-read means change from 103.16 to 85.51 GB
(-17.11%) and from 103.62 to 86.72 GB (-16.31%). Pooled native decode
changes from 1.802 to 2.155 tokens/s (+19.60%) and from 1.897 to 2.445
(+28.88%). These whole-session rates include the fixed mix of first and
repeated tasks; they are not sustained or universal throughput figures.

Each profile/order has two sessions. All four paired latency improvements
and all output lengths are retained, including the slower candidate first
short answers. Short-first candidate session times vary from 188.81 to
212.26 s. Sequential OS cache, calibration and routing/placement can vary.
The comparison establishes an observed benefit for this fixed task mix,
not a general speedup or an isolated effect of the 1000 ms budget.

### First, later and repeated requests

Each cell is **production windows / flow candidate**.

| Order | Position / task | TTFT s | Request s | Reads GB | Pooled decode tokens/s | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| short-first | 1: zh_fact | 21.92 / 18.75 | 25.91 / 27.49 | 11.63 / 12.80 | 1.253 / 0.572 | [6, 6] / [6, 6] |
| short-first | 2: long_retrieval | 107.13 / 81.92 | 124.93 / 100.99 | 48.53 / 45.75 | 1.741 / 1.626 | [32, 32] / [32, 32] |
| short-first | 3: zh_fact_repeat | 9.52 / 6.90 | 11.45 / 7.64 | 4.82 / 3.00 | 2.597 / 6.742 | [6, 6] / [6, 6] |
| short-first | 4: long_retrieval_repeat | 83.57 / 59.56 | 99.81 / 64.42 | 38.18 / 23.97 | 1.910 / 6.380 | [32, 32] / [32, 32] |
| long-first | 1: long_retrieval | 127.36 / 92.73 | 144.85 / 113.17 | 56.37 / 51.01 | 1.773 / 1.516 | [32, 32] / [32, 32] |
| long-first | 2: zh_fact | 9.51 / 9.97 | 11.56 / 13.37 | 4.76 / 5.83 | 2.438 / 1.470 | [6, 6] / [6, 6] |
| long-first | 3: long_retrieval_repeat | 84.02 / 50.02 | 100.52 / 54.98 | 38.07 / 24.52 | 1.879 / 6.260 | [32, 32] / [32, 32] |
| long-first | 4: zh_fact_repeat | 8.60 / 11.55 | 10.51 / 12.19 | 4.42 / 5.36 | 2.615 / 7.799 | [6, 6] / [6, 6] |

There are consistent short-answer tradeoffs. In short-first, the candidate's
first TTFT improves from 21.92 to 18.75 s (-14.47%), but the complete short
answer increases from 25.91 to 27.49 s (+6.07%). Its decode falls from
1.253 to 0.572 tokens/s (-54.33%). These very short answers have only five
decode steps, so this rate must not be generalized to longer generations.

In long-first, the second short answer increases from 11.56 to 13.37 s
(+15.62%), and the final repeated short answer from 10.51 to 12.19 s
(+15.93%). The latter has faster decode, 2.615 to 7.799 tokens/s, but its
TTFT grows from 8.60 to 11.55 s (+34.24%), outweighing the decode saving.
These short-request regressions occur in both paired execution orders.

The longer tasks provide the main full-session benefit. Initial long-answer
time decreases from 124.93 to 100.99 s in short-first (-19.17%) and from
144.85 to 113.17 s in long-first (-21.87%), despite initially slower decode.
Repeated-long time changes from 99.81 to 64.42 s (-35.46%) and from 100.52
to 54.98 s (-45.31%). Its pooled decode improves from 1.910 to 6.380 and
from 1.879 to 6.260 tokens/s, approximately 3.34x/3.33x. Short-first's
repeated short answer also improves, from 11.45 to 7.64 s (-33.26%).

### Native phases and cache pressure

All 16 production requests use windows; all 16 candidate automatic choices
select flow. Initial GPU expert counts are consistently 1729 versus 1641
under the same 20 GiB target. Peak GPU use is approximately 19.463 versus
19.510 GiB. The builds and workspace/placement differ, so results describe
complete profiles rather than a change in budget or operator speed alone.

Both RAM stores reach their 4595-slot limit. Summed across the 64 phase
snapshots from 32 requests, STORE records 37082 evictions: 34175 on windows
and 2907 on the candidate. Decode misses are 7208/5614. Windows-prefill
hits/misses are 143450/45347, while flow-prefill reports 0/0 despite
nonzero stream/file/process reads. Flow's ratio is undefined. The paths
exercise different IO/counter populations; eviction or hit counts alone
cannot quantify the speedup or an exact physical-cache hit rate.

Candidate inclusive settling ranges from 1.001 to 9.302 s, with RAM preload
from 0 to 8.302 s. The 1000 ms value is a positive swap-loop budget and
does not bound total settling. Window requests have no flow-stage timing;
those absent measurements remain null, not zero.

All 32 raw rows, phase counters, path choices, output-ID groups and within-run
repeat pairs remain in `step7aa-analysis.json`. Stream contains read waits
and handoff; inclusive settle contains RAM preload. Do not add nested times
twice. STORE submitted bytes are queued payload; hits/misses do not cover
all flow-stream IO. Undefined hit ratios remain null. These counters cannot
be treated as identical prefill-cache populations across the two paths.
Repeated requests also change routing/placement and OS cache.

### Quality, resources and verification

All 32 answers pass content, requested-format and truncation review:
16 Chinese factual answers and 16 strict unique-key JSON retrieval answers.
Each task has identical output token IDs across profiles, orders and repeat
positions. Short answers contain six tokens including EOS, long answers 32;
every four-request session has 76 output tokens. All stop at EOS before
the 64/128-token caps. These two fixtures do not establish broader model
quality, longer-context behavior or official GPU numerical parity.

Maximum observed GPU use is 19.510 GiB; maximum process HWM is 38.42 GB; minimum host available is 14.34 GB. All process bounds pass. Engine swap and STORE read failures are zero. Samples are one-second/request-boundary plus process HWM; GPU excludes capacity hold but includes context/other users. Engine swap is not system-wide swap.

Seven new no-GPU controls pass: actual production-entry command capture,
four exact profile/order checks and two low-host-memory rejections. Prior
7z regression/control evidence is preserved, not rerun: production and
benchmark source are byte-identical. Each native process has one READY,
four DONE, fixed prompt IDs/caps, no KV reuse and no read/resource failure.
Complete answers are reviewed without discarding any timing or length.

All six native source trees/binaries, three core archives, eight pack files,
benchmark, production entry and 99 runtime/header files are preserved. Prior
7z/7y/7x/7w/7v evidence is verified using pre-edit snapshots where needed.
Weight checks reuse prior full hashes plus current sizes/.done markers;
no new full 231 GB rehash is claimed. The main checkout remains clean.

Raw results are hash-pinned with adjacent answer reviews. Controls, commands,
analysis, gate-only documentation patch, failures and preservation checks are
indexed in `step7aa-validation.json`. No commit, push or PR is part of this gate.

**7aa complete; user confirmed.** Its complete-profile measurements remain preserved; the proposed opt-in entry is implemented and checked in 7ab below.

## Step 7ab: opt-in native flow entry

**The measured flow candidate is now explicitly selectable for single text
questions and resident independent questions. Windows remains the default.**

`run_quantized.py` accepts `--profile windows|flow-experimental`. Omitting
it or choosing windows keeps the prior native7i command, environment and
pins. The experimental choice selects the exact native7w engine previously
measured in 7aa: chunk 256, flow-store on, positive 1000 ms swap-loop budget,
RAM preload on, mixed-prefill off and first-window off. Its native chooser
remains automatic; no native operator or backend source is changed.

The public entry passes a named text-profile contract to the shared runner.
That path requires one text source, the pinned candidate SHA256 and pack
verification, STORE diagnostics, 40 decimal GB LRU, 20 GiB GPU and the complete
measured settings. Unknown/incomplete or incompatible profile arguments fail before
output creation/GPU allocation. Asset-verification failures are recorded
before GPU startup. Existing benchmark combinations retain their
prior admission behavior; the opt-in does not relax their text-mode checks.

The tokenizer/plain-user spans, streaming, input reader, protocol, memory
sampling, timeouts and cleanup are reused. Shared `TextStream`, `LineInput`,
`prepare_case`, `done_metrics`, `parse_store_line` and `run_engine` source
segments remain byte-identical. The context stays 1024, output cap 1–256,
with an 8 GiB host-available floor, zero engine-swap requirement, existing
model/pack/tokenizer pins and inherited-environment isolation.

Both exact prior CUDA 12.8/SM75 binaries are reused without rebuilding:
production7i SHA256
`86c9c86d4d30bf0136f15226f17ec43785a58461992f61b819d4e02e9a037ce4`,
candidate7w SHA256
`a28b9f809c5db153bb16f8ff326429fbe0146dcf3fdaa5cffbc5d1acbecb0f96`.
No new HIP/SYCL execution or native cancellation measurement is claimed.
Normal CLI exit and shared fake-child interrupt/error cleanup are checked.

### Usage and choice

From the worktree root:

```sh
python3 tools/deepseek41/run_quantized.py --profile flow-experimental --interactive
```

For a single question:

```sh
python3 tools/deepseek41/run_quantized.py --profile flow-experimental \
  --prompt '中国的首都是哪里？请用一句话回答。'
```

The same option works with `--prompt-file`, including stdin. The entry writes
a short experimental-choice notice to stderr, while stdout remains streamed
answer text. Result metadata records `text_profile=flow-experimental` and the
actual pinned native command/environment. Input/output/cancellation behavior
and all existing limits are described in [local use](DEEPSEEK41_LOCAL_RUN.md).

The choice is experimental. In 7aa's two fixed tasks and repeats, full-session
time improved by 23.5–27.6%, with 6–16% slower short requests. Each profile/order
had only two sessions. These results do not support a universal speedup or a
default replacement. The 1 second value limits the swap loop condition;
subsequent RAM preload can make total settling longer.

### Fixed real-entry smoke

The plan was fixed before native measurement: default single short/cap64,
experimental single short/cap64, experimental multiline long-file/cap128
launched from `/tmp`, then one experimental resident engine with three valid
questions/cap128. The resident input first includes a blank and one overlong
6000-character question, followed by short, one-line warehouse retrieval,
repeat short and `/exit`. The rejection produces no native GEN; subsequent
questions share one loaded engine with fresh KV.

Interactive input accepts one physical line per question. Warehouse newlines
are replaced by spaces before the smoke and its rendered IDs are pinned.
The file question is 257 tokens, the one-line version 264, short questions 14.
No timing-based rerun or selection is used. This is functional entry
verification, not another profile performance comparison.

| Run | Input mode / cwd | Startup s | Valid requests | Input rejections |
| --- | --- | ---: | ---: | ---: |
| window-short | prompt / worktree | 40.49 | 1 | 0 |
| flow-short | prompt / worktree | 32.45 | 1 | 0 |
| flow-long-file | file / /tmp | 32.24 | 1 | 0 |
| flow-interactive | interactive / worktree | 41.91 | 3 | 1 |

| Run / request | Path | Input tokens | Output cap | Output tokens | TTFT s | Request s | Finish |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| window-short / text | windows | 14 | 64 | 6 | 22.16 | 26.28 | stop |
| flow-short / text | flow | 14 | 64 | 6 | 20.30 | 29.00 | stop |
| flow-long-file / text | flow | 257 | 128 | 32 | 97.06 | 119.52 | stop |
| flow-interactive / text-1 | flow | 14 | 128 | 6 | 21.30 | 30.57 | stop |
| flow-interactive / text-2 | flow | 264 | 128 | 32 | 86.03 | 106.92 | stop |
| flow-interactive / text-3 | flow | 14 | 128 | 6 | 7.08 | 7.80 | stop |

All four CLI processes return zero. Each has one READY and one DONE per valid
question; the resident process has exactly three DONE despite one rejected
input. All six requests have zero KV reuse, correct content/format and EOS
before their caps. Four Chinese factual answers and two strict unique-key
JSON answers are complete. CLI stdout exactly matches recorded completions
with per-answer newlines; EOS/protocol text is kept out of answer output.
Normal `/exit` records a closed session and releases the native process and
capacity hold. File input from `/tmp` verifies asset paths are independent
of cwd. This smoke covers these two task contents only.

The shared offline trace reader expects benchmark status; the gate passes
an in-memory `completed` status alias for parsing only. Saved raw entry
results and statuses remain unchanged and hash-pinned.

### Tests, resources and preserved scope

The 28 Python regressions pass, including four new profile tests and existing
fake-child resource/timeout/interrupt/cleanup cases. All 99 earlier option,
environment and sequence controls pass. Another 19 real-asset/no-GPU entry
controls cover omitted/explicit windows and experimental profiles across
prompt/file/stdin/interactive input, exact native command/env/pack pins,
low host memory, bad engine/pack pins and overlong single-question rejection.

Maximum observed GPU use is 19.531 GiB, maximum process HWM 35.34 GB, and minimum host available 15.06 GB. All bounds pass; engine swap and STORE read failures are zero. One-second/boundary samples plus HWM do not capture every instantaneous peak. GPU excludes the capacity hold but includes context/other users; engine swap is not system swap.

All six native source trees/binaries, three core archives, eight pack files,
99 runtime/header files and prior evidence through 7aa are preserved. The
default entry's native command/environment matches its pre-edit record.
Only the public selector, shared-runner named-profile admission, new profile
tests and the two documentation files change. The main checkout remains
clean. No download, commit, push or PR is part of this gate.

`step7ab-smoke.json` records the fixed commands, source hashes, raw-result
hashes and actual native paths. `step7ab-entry-summary.json` and adjacent
answer reviews retain the six outputs, rejection and streaming checks.
The gate-only patch, tests/controls and preservation results are indexed
in `step7ab-validation.json`.

**7ab complete; user confirmed.** The opt-in entry remains available; the proposed long-output validation is measured in 7ac below.

## Step 7ac: longer explanation and code-output validation

**Longer generation and the tested code functions work within the existing
bounds. Strict explanation format and one topic-coverage case fail. The
experimental profile stays opt-in; windows remains the default.**

This gate reuses the unchanged real interactive entry, tokenizer and exact
native7i/7w CUDA 12.8/SM75 binaries. Default windows and flow-experimental
each receive the original long-explanation99 and long-Python121 token tasks
in both orders. Prompts/IDs match 7k; cap256, context1024, fresh KV,20 GiB GPU,
40 decimal GB LRU and8 GiB host floor remain fixed. One resident process
answers two independent questions and exits with `/exit`.

The four-process/eight-answer plan was fixed before measurement:
window-explain-code, flow-explain-code, flow-code-explain,
window-code-explain. No prompt rewrite, cap increase, cache drop or selective
rerun is used. Failed quality checks retain their outputs and timings.
Native builds/workspace/cache behavior differ between profiles. One answer
per task/profile/order is a quality/control observation, not a stable
performance estimate or causal isolation of the1000 ms budget.

### Criteria fixed before measurement

Content, requested format and natural completion are graded independently.
Explanations require six ordered 1–6 items, the six requested topics with
reasons, complete coherent sentences and no intro/summary. Each item must
have 25–35 Han characters, excluding punctuation, Latin letters and numbers.
The fixed CJK-range regex and exact vectors are in `step7ac-criteria.json`;
its hash is pinned in the native report and checked after the runs.

Code requires only `sum_even(nums)` and `summarize_even(nums)`, no imports
or outside prose, and accurate single-line Chinese docstrings. The policy
accepts raw Python or one optional Python fence with no outside prose;
wrappers are recorded rather than silently removed from raw output.

Eight fixed integer-list vectors per function cover empty, mixed, negatives,
odd-only, zeros, duplicates, large integers and all-negative inputs. Exact
stats keys/values and input preservation are checked; mean uses the fixed
1e-12 relative/absolute tolerance. Complete source/AST is read before running
it in an isolated interpreter with restricted pure builtins and CPU/memory
limits. Unsupported constructs stay pending review, not automatically wrong
or passed. Natural EOS counts as completion, including at the cap; a length
finish without EOS remains incomplete even if a code fragment parses.

### Full-answer results

| Run / task | Output tokens | Content | Strict format | Natural completion | Function checks |
| --- | ---: | --- | --- | --- | ---: |
| window-explain-code / long_explanation | 139 | pass | pass | pass | — |
| window-explain-code / long_python | 192 | pass | pass | pass | 16 passed |
| flow-explain-code / long_explanation | 153 | pass | fail | pass | — |
| flow-explain-code / long_python | 187 | pass | pass | pass | 16 passed |
| flow-code-explain / long_python | 192 | pass | pass | pass | 16 passed |
| flow-code-explain / long_explanation | 159 | fail | fail | pass | — |
| window-code-explain / long_python | 192 | pass | pass | pass | 16 passed |
| window-code-explain / long_explanation | 155 | pass | fail | pass | — |

All eight answers stop at EOS, generating 139–192 tokens without reaching the
256 limit. Content requirements pass 7/8 and requested format 5/8. All four
code answers pass syntax, the two requested signatures, Chinese docstrings
and 16 function/vector calls each: 64/64 pass, with input preservation. This
is a bounded two-function check, not general code-generation quality.

Only one of four explanations meets every requested length bound:

| Run | Han counts in items 1–6 | All 25–35? |
| --- | --- | --- |
| window-explain-code | 29, 28, 31, 28, 29, 32 | pass |
| flow-explain-code | 27, 30, 35, 31, 32, 37 | fail |
| flow-code-explain | 30, 32, 33, 30, 35, 37 | fail |
| window-code-explain | 30, 31, 36, 29, 33, 36 | fail |

Both experimental explanations have 37 Han characters in item 6, exceeding 35.
The code-first default explanation has 36 in items 3 and 6. The default path
also exhibits length misses, so they cannot be assigned only to the
experimental scheduling policy. Counts are retained without trimming the
answers or relaxing the bound.

One experimental answer, flow-code-explain/item3, also has incomplete topic
coverage: it states that loaded weights use stored bytes and that activations
need space, but does not clearly explain why the quantized weight volume
itself can still exceed VRAM. This is a prompt-coverage judgment, not a claim
that activation overhead is factually false. Other requested topics are
covered. The full manual reasoning and completion hashes remain in
`step7ac-manual-reviews.json`; full raw answers remain adjacent to reviews.

### Timing and cache observations

These are retained measurements, not equal-output speedup estimates.
Output wording/lengths differ, and there is one sample per task/profile/order.

| Run / task | Path | TTFT s | Request s | Decode tokens/s | Whole-request reads GB |
| --- | --- | ---: | ---: | ---: | ---: |
| window-explain-code / long_explanation | windows | 70.97 | 118.91 | 2.878 | 51.57 |
| window-explain-code / long_python | windows | 57.94 | 136.60 | 2.428 | 59.98 |
| flow-explain-code / long_explanation | flow | 57.92 | 131.92 | 2.054 | 62.48 |
| flow-explain-code / long_python | flow | 52.45 | 141.84 | 2.081 | 66.14 |
| flow-code-explain / long_python | flow | 60.62 | 158.82 | 1.945 | 74.39 |
| flow-code-explain / long_explanation | flow | 63.52 | 120.75 | 2.761 | 57.09 |
| window-code-explain / long_python | windows | 82.39 | 166.83 | 2.262 | 72.50 |
| window-code-explain / long_explanation | windows | 56.73 | 102.55 | 3.361 | 44.74 |

The experimental decode rate is lower at all four matched positions in these
observations. Explanation-first explanation/code rates are 2.878/2.428 on
windows and 2.054/2.081 experimentally; code-first code/explanation rates
are 2.262/3.361 versus 1.945/2.761. This does not extend 7aa's 23.5–27.6%
short-output task-mix improvement to longer generation, nor establish a
universal slowdown magnitude. Windows remains the default.

| Profile | Decode steps | Decode misses / step | STORE wait ms / step | Native decode ms / step | STORE wait / decode wall | All-phase STORE evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| windows | 674 | 15.22 | 280.30 | 381.09 | 73.55% | 15541 |
| flow-experimental | 687 | 20.05 | 364.17 | 464.08 | 78.47% | 10178 |

Both stores reach 4595 allocated slots; this is arena capacity, not occupied
or useful cache entries. The normalized decode counters motivate an offline
fetch/cache attribution next. STORE wait overlaps CPU lane work and
may overlap GPU work; it is not pure SSD wait or GPU-idle time. Submitted
bytes/hit ratios do not describe all physical IO paths. Whole-request reads
include prefill, settling and decode. Stream contains read waits/handoff;
inclusive settling includes RAM preload. Do not add nested timers twice
or treat absent flow timers on windows as zero. Differing routing/placement,
output tokens and OS state prevent budget-only attribution.

### Resources, controls and preservation

Maximum observed GPU use is 19.531 GiB, maximum process HWM 38.29 GB, minimum host available 11.98 GB. All four processes pass the bounds. Engine swap and STORE read failures are zero. One-second/boundary samples plus HWM cannot capture every instantaneous peak; GPU excludes the hold but includes context/other users. Engine swap is not system-wide swap.

All processes have one READY/two DONE, no KV reuse, exact pinned input IDs
and cap256. CLI stdout matches the recorded completions; sessions close and
native PIDs/capacity holds are released. Saved entry status remains completed;
trace parsing uses an in-memory status alias only. Raw reports are unchanged.

Sixteen new offline controls pass: four profile/cap256 forwarding checks,
ten extraction/Han-boundary checks and two code-checker calibrations, which
accept a correct pure implementation and reject a wrong sum. Prior 7ab's 28
Python regressions and 118 controls are preserved, not rerun, because runtime
source is byte-identical. The fixed vectors/criteria are not relaxed.

Only roadmap and local-use documentation change. All 211 pre-existing
canonical files except those two, 99 runtime/header files, six native source
trees/binaries, three core archives, eight pack files and prior evidence
through 7ab are preserved. No native/backend build, download, commit, push or
PR is part of this gate. The main checkout remains clean.

`step7ac-validation-runs.json` pins commands, criteria/source hashes and raw
results. `step7ac-answer-reviews.json` retains all quality outcomes, manual
notes and code/vector examples. `step7ac-summary.json` retains timing and
cache/resource observations. Controls, jobs, gate-only documentation patch
and preservation checks are indexed in `step7ac-validation.json`.

**7ac complete; user confirmed by authorizing 7ad.** Keep windows as default
and flow-experimental opt-in. The authorized offline fetch/cache attribution
is recorded below. Strict wording/length compliance remains an output limitation.

## Step 7ad: offline decode fetch and cache attribution

**The saved long-decode gap is accounted for mainly by more demand misses
and longer `ensure()` tails per step. Conservative flow retention is a
specific next hypothesis to test; a speedup from changing it is unproven.**

This gate only reads the four 7ac native processes/eight answers and the
pinned 7i/7w source. It launches no model or benchmark, changes no runtime,
operator, weight or default, and updates only this roadmap. Earlier 7aa
workloads are not pooled into these rates.

### Decode accounting and the same-output control

Rates below use actual fed decode steps, 674 windows and 687 flow, rather than
output counts including EOS or averages of per-request ratios.

| Saved7ac observation | windows | flow-experimental |
| --- | ---: | ---: |
| Native decode wall ms / step | 381.09 | 464.08 |
| STORE ensure-tail wait ms / step | 280.30 | 364.17 |
| CPU lane minus ensure-tail ms / step | 75.62 | 74.88 |
| Wall minus CPU lane ms / step | 25.18 | 25.03 |
| Demand misses / step | 15.22 | 20.05 |
| Ensure-tail ms / miss, aggregate proxy | 18.42 | 18.17 |
| Submitted logical MB / miss | 7.407 | 7.410 |
| Submitted logical MB / step | 112.746 | 148.548 |
| Rounded decode process physical MB / step | 175.789 | 212.917 |
| Decode-only STORE evictions | 9510 | 7777 |

Flow-minus-windows wall is 82.989 ms/step; ensure-tail wait is 83.867,
CPU-minus-wait is -0.735, and wall-minus-CPU is -0.143. This arithmetic identity
locates the observed gap in the measured wait component. The per-miss proxy
is slightly lower for flow while miss volume is higher; these logs do not
show intrinsically slower CPU arithmetic or establish higher device latency
per read. The table's evictions are decode only;7ac's 15541/10178 include
prefill as well. Fewer evictions do not imply fewer misses when the arena grows.

For the 121-token Python task, three answers have identical 192 output IDs.
The two **first-request** code answers have the same prompt, output and 191
decode steps: flow 514.17 versus windows 442.05 ms/step, a 72.119 ms gap;
ensure-tail accounts for 71.797 ms/step of that gap. This removes output length
and emitted-token differences for one control. GPU placement, hidden routing,
RAM/page-cache state and process ordering remain different, so it is not a
controlled estimate of a retention-policy effect.

### What the counters measure

`ExpertStore::ensure()` submits absent experts before starting `wait_ns`.
The recorded tail includes completion waits, locks and bookkeeping; initial
admission/submission is outside it. The CPU lane timer includes `ensure()`
and CPU expert compute/reduction. Subtractions above are accounting terms,
not disjoint CPU/GPU timelines or measured GPU-idle time. Wait/miss is not
individual SSD latency, and payload/wait is not storage bandwidth.

A miss means no store mapping at demand time. Hits require ready, non-guess
entries; hits+misses does not cover every routed expert or inflight state.
Submitted bytes are logical expert payload queued by demand, preload and
prefetch, not physical read bytes. Internal no-slot/cancellation counters and
per-layer/expert misses are not present in the current STORE protocol.

Native final `STAT disk_mb` **does** provide a decode-scoped Linux
`/proc/self/io read_bytes` delta, divided by fed steps and rounded to 0.001 MB.
The table weights those final rates by steps; rounding is at most 0.0005 MB
per step. It includes Engram, promotions and other process reads and is
sampled just before final STORE/DONE. It cannot isolate STORE physical reads,
device/queue latency or exact prefill reads. Whole-request read bytes remain
a separate measurement. Do not subtract logical STORE bytes from physical
reads to infer another path's traffic.

### Cache seed and the native admission path

| First request | Flow retained experts | Flow prefill allocated slots | Flow RAM preload | Windows prefill demand misses | Windows prefill allocated slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation99 | 1427 | 1472 | 0 | 3957 | 3968 |
| Python121 | 1698 | 1728 | 0 | 4484 | 4544 |

Both cold flow handoffs report no-room, early-source and invalid counts 0.
Their synchronous handoff copies take 13.136/15.260 seconds inside stream
execution. These observations suggest testing admission before adding memory
or speculative reads; extra copying is already material and cannot be assumed
free. `slots` is allocated capacity, not live occupancy or future usefulness.

The native flow path first updates heat with actual routed counts, then
retains currently routed experts only when heat>=4. Heat starts from the
routing profile, scaled to about 512 hits per layer, and decays;4 does not mean
four accesses in this prompt. `retain_free()` copies all three complete
matrices after successful reads and before staging release, only into free
slots, without another read or eviction and under the existing host floor.
Windows demand-fill has no corresponding heat filter.

Post-prompt RAM preload independently selects non-VRAM experts with heat>=4.
Unlike free-slot retention, it can replace entries via `start_fill()`.
Cold flow preload is 0; the later requests preload 876/1525 experts, taking
13.915/24.019 seconds within inclusive settling 14.916/25.021 seconds.
The 1000 ms setting bounds the preceding swap loop, not total settling.

All later requests start decode with 4595 allocated slots, yet the flow miss
gap remains. Cold seed size therefore cannot explain every request. No saved
process reports arena shrink. Initial VRAM expert counts differ, 1727 windows
versus 1642 flow; nevertheless flow's weighted final VRAM-hit share is higher,
42.82% versus 40.65%, while its store misses are higher. Initial expert counts
alone do not explain the result. The logs lack the rejected-heat distribution,
ready/free occupancy and later reuse needed for exact per-expert attribution.

The SSD store disables router lookahead unless `DS4_AHEAD` is explicitly set;
all saved decode prefetch counters are 0. This is no measurement of predictor
accuracy. Native source warns of SSD contention in a different environment;
those historical figures are not a 2080 Ti result. Keep lookahead unchanged.

### One proposed next experiment and preserved scope

Propose 7ae: a default-off **flow free-slot retention heat threshold** control,
comparing current 4 with 1 in the **same newly built binary**. Preserve the
actual-route/read-success/staging-lifetime checks and existing single-device,
pageable, non-copy eligibility. Keep preload heat 4, LRU/budgets, GPU promotion,
settle 1000, chunk 256, chooser, lookahead, mixed/first-window and weights fixed.
Reuse existing native operators and `retain_free()`; no new math kernel is
needed. The wrapper strips inherited DS4 variables, so the experiment needs
an explicit validated benchmark option and recorded metadata. Default native
threshold 4 and production windows/public profile choice stay unchanged.

After boundary/default/forwarding and retain-lifetime controls plus the SM75
build, pin the criteria and run at most four resident processes/eight original
cap256 answers: heat4-explain-code, heat1-explain-code, heat1-code-explain,
heat4-code-explain. Preserve 20 GiB GPU, 40 decimal GB LRU, 8 GiB host floor,
18 workers, context 1024 and fresh KV. Report first/later requests separately,
including retained/copied experts, first-token and full request/session time,
decode misses/wait/reads, host/GPU peaks and unchanged code/content/format
checks. Retain failures and all outcomes; no cache drop, selective rerun,
threshold fitting or default promotion. Extra copies may offset decode gains;
free-only retention cannot replace entries in a full store. If the hypothesis
fails, keep 4 and record the failure.

`step7ad-analysis.json` pins all eight raw records and rate/ID controls.
`step7ad-source-review.json` records ten findings with hashes and line anchors;
the main/IO files and eight relevant native function bodies match between
7i and7w. Eight new offline arithmetic/data controls pass. Prior runtime
tests are preserved, not rerun: no backend/source/binary changed. The gate
patch and preservation checks retain the other 210 canonical files, six native
source trees/binaries, three core archives, eight pack files and prior evidence
through 7ac. No build, model run, download, commit, push or PR occurs in 7ad.

**7ad complete; user confirmed by authorizing 7ae.** Keep windows default and
flow opt-in. The bounded native retention experiment is recorded below.

## Step 7ae: opt-in flow retention heat 4 versus 1

**The threshold-1 experiment improves the tested long-output request totals
by 4.58%, mainly through first-request decode. Later normalized decode is
slightly worse. Keep native default 4 and the public profiles unchanged.**

This is a Strata native cache-admission change, reusing existing operators
and `ExpertStore::retain_free()`. It adds no quantization or math kernel.
The same new CUDA 12.8/SM75 binary serves both modes on the RTX 2080 Ti under
the 20 GiB target, 40 decimal GB LRU store, 8 GiB host floor and 18 workers.
The original explanation99/Python121 input IDs, cap256/context1024 and fresh
KV remain fixed; expert and OS caches persist within each two-request session.

### Implementation and fixed scope

`tools/deepseek41/patches/flow_store_retention.patch` applies after the 7w
first-window patch. Two native files change: a small exact-value parser in
`ds4_prefill_schedule.hpp`, and one model field/eligibility comparison plus
an explicit-1 diagnostic in `ds4_model.cu`. Only exact
`DS4_FLOW_STORE_MIN_HEAT=1` lowers the threshold; unset, explicit4 and invalid
values keep4. The setting is captured per model. Heat includes profile
initialization, routed counts and decay, not just this prompt's access count.

The actual-route check, complete/successful read requirement, early-DMA
source guard, staging event and retain-before-release site stay unchanged.
`retain_free()` itself is byte-identical: copy three owned matrices into free
slots, respecting host floor, with no new read or eviction. Single-device,
pageable and non-copy eligibility stays unchanged. GPU placement policy,
preload heat4, settle1000, chunk256, chooser, lookahead and mixed/first-window
off settings remain fixed. Downstream cache/placement state can still differ.

The benchmark-only `--flow-store-min-heat 1|4` requires the full pinned
configuration and `--resident-order explain-code|code-explain`; unsupported
resources, partial options, public text/interactive input and first-only
probes fail before output/GPU. Inherited DS4 variables are stripped; the
chosen threshold is explicitly forwarded and recorded. Existing defaults,
public `run_quantized.py`, local-use docs and pinned 7i/7w binaries remain
unchanged. No user-facing profile is promoted to this new binary.

The new binary is `build-deepseek-retention-sm75/strata-dsv41`, SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
`step7ae-candidate-build.json` pins complete configure/build commands and
logs; all ten CUDA cubins target SM75. HIP/SYCL toolchains are unavailable,
so those backends are not validated. The patch applies/reverses exactly;
the full 1138-file candidate source is pinned.

### All four sessions and first/later requests

Criteria/vectors and the four-process/eight-answer order were fixed before
measurement: heat4-explain-code, heat1-explain-code, heat1-code-explain,
heat4-code-explain. No cache drop, prompt/cap change, fit or selective rerun.
All processes use the same binary and exit0, with one READY/two DONE each.

| Two-request order | Heat4 request total s | Heat1 request total s | Observed reduction |
| --- | ---: | ---: | ---: |
| Explanation, Python | 271.77 | 260.32 | 4.21% |
| Python, explanation | 280.49 | 266.65 | 4.94% |
| Both sessions | 552.25 | 526.97 | 4.58% |

These request totals exclude model startup/shutdown. Native engine wall
including startup/shutdown totals643.22 versus621.38 seconds, 3.40% lower.
Startup totals84.09 versus87.64 seconds; startup is not counted as decode.
Each mode produces680 output tokens including EOS and676 fed decode steps,
but **none** of the four same-position/task pairs has identical output IDs.
Equal aggregate counts do not establish identical routing or numerical parity.

| Position / task | Heat4 first token s | Heat1 first token s | Heat4 request s | Heat1 request s | Heat1-minus-4 decode ms / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| First / explanation99 | 54.92 | 57.87 | 129.45 | 116.06 | -123.46 |
| First / Python121 | 62.46 | 62.66 | 161.47 | 150.00 | -75.07 |
| Later / explanation99 | 63.49 | 63.09 | 119.02 | 116.65 | +5.01 |
| Later / Python121 | 53.02 | 55.68 | 142.31 | 144.26 | +8.77 |

The later explanation finishes sooner with fewer tokens while normalized
decode is slower. The later Python request is slower overall. Do not describe
the result as every request or every first-token latency improving.

| Step-weighted decode observation | Heat4 | Heat1 |
| --- | ---: | ---: |
| Wall ms / step | 470.95 | 425.55 |
| Ensure-tail wait ms / step | 371.43 | 324.56 |
| CPU lane minus ensure-tail ms / step | 74.50 | 75.92 |
| Demand misses / step | 20.36 | 17.76 |
| Ensure-tail ms / miss, aggregate proxy | 18.25 | 18.28 |
| Rounded decode process physical MB / step | 218.250 | 198.031 |

Decode wall falls9.64%, misses12.76% and ensure-tail12.62%; total first-token
latency instead rises2.31%. First-request decode is511.95 to414.62 ms/step;
later decode is429.71 to437.35 ms/step. This supports the cold-seed hypothesis
within these fixtures, not a universal threshold. Wait is nested in CPU lane;
copy is nested in stream; preload is inside inclusive settling. Neither
wait/miss nor payload/wait measures individual device latency/bandwidth.
Final STAT physical MB covers all process reads, not STORE-only traffic;
its rounding is0.001 MB/step. Do not add/subtract nested or unlike counters.

### Cold seed, full store and output quality

| First prompt | Heat4 retained / copy s | Heat1 retained / copy s | Heat4 allocated slots | Heat1 allocated slots |
| --- | ---: | ---: | ---: | ---: |
| Explanation99 | 1427 / 12.815 | 3944 / 28.308 | 1472 | 3968 |
| Python121 | 1698 / 15.302 | 4498 / 30.931 | 1728 | 4544 |

Cold retention no-room/early/invalid counts are0 and RAM preload is0. The
larger copy timer does not equal the incremental first-token cost; stream
waits/IO state also differ. Allocated slots measure arena capacity, not live
occupancy or future reuse. Decode-only evictions rise7739 to11257 despite
fewer misses, partly because threshold1 starts with more seeded slots.

All later prefills end at4595 allocated slots. Heat1 retains0 additional
experts and reports2455/2500 no-room attempts. Later RAM preload reads1082/1511
hot experts in17.046/23.441 seconds; heat4 reads884/1506 in14.030/23.934
seconds. These waits remain a measured next target, not proof preload is
unnecessary: removing it may move waiting into decode. The positive1000 ms
budget does not cap total settling. No arena shrink is logged.

All8 answers reach EOS below cap. The four code answers pass64/64 fixed
function/vector calls, including empty/negative/large values and input
preservation. This is four code answers, not64 independent model tasks.
Content passes6/8; requested format7/8. Both later explanation answers, one
per threshold, say VRAM capacity determines whether a model can load without
qualifying device residency, misleading for this RAM/SSD offload setting.
Heat1-first explanation item6 has38 Han characters, above35. Full outputs,
manual reasons, hashes and failed checks are retained under unchanged7ac
criteria. No quality improvement, causal quality regression or official parity
is claimed from these samples.

### Controls, parser repair and preservation

42 Python tests pass:28 existing shared benchmark/public-entry tests,4 new
threshold-guard tests, and10 audit tests including2 new diagnostic tests.
123 mock controls pass:99 legacy,19 public-entry and5 threshold/default
forwarding controls. The inherited new setting does not alter old command/env
bytes. Two real native tests pass with ASan/UBSan: exact threshold parsing and
existing ExpertStore ownership/free-only/floor/duplicate/inflight cases.

After process2 completed successfully, the strict offline audit parser rejected
the new `experimental min_heat=1` log. The driver stopped before process3.
The fix accepts only a unique, flow-only, active-handoff line bound to method
heat1, rejects missing/invalid/duplicate/conflicting records, and preserves
default audit rows exactly. Both completed raw results/logs were retained and
reaudited; only the remaining two planned processes then ran. The old failed
driver/report are preserved. This pause and uncontrolled OS state limit timing
inference. No native answer was rerun, normalized away or excluded.

Maximum observed GPU use is19.531 GiB, process HWM38.79 GB, and minimum host
available12.08 GB. Engine swap, STORE read failures and invalid handoffs are0.
All four processes pass bounds and release their PIDs/capacity holds. Samples
plus HWM do not observe every instantaneous GPU/host peak; process swap does
not measure system-wide swap.

Seven canonical files change: roadmap, benchmark, new native patch, two new
threshold tests, and the necessary audit parser/test update. The other207
pre-existing canonical files,99 runtime/header files, six prior native source
trees/binaries, three core archives, eight pack files and prior evidence through
7ad remain unchanged. Main checkout stays clean. No weights/downloads, commit,
push or PR. `step7ae-validation.json` indexes commands, source/build pins,
raw outputs, reviews, controls, summaries, gate patch and preservation checks.

**7ae complete; user confirmed by authorizing 7af.** Keep heat1 benchmark-only
and retain production defaults. The bounded existing RAM-preload switch
experiment, including its resource failure, is recorded below.

## Step 7af: RAM preload on/off at native flow heat 1

**The benchmark allowance and preflight checks are complete. The resource
gate fails: seven answers complete, and the final preload-on answer is
stopped after engine swap reaches288 KiB. The full four-session comparison
is unavailable. Keep all defaults unchanged and diagnose the resource event.**

This gate reuses the exact 7ae CUDA12.8/SM75 binary, SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
There is no native/backend build, operator change or new cache implementation.
Only existing `DS4_SETTLE_STORE=1|0` differs between modes. Native flow heat1,
positive GPU swap-loop budget1000, chunk256, LRU40 decimal GB,20 GiB GPU,
8 GiB host floor,18 workers, context1024 and fresh KV stay fixed.

### Narrow benchmark allowance and preflight

The benchmark now accepts `--settle-store off` only in the explicitly pinned
heat1 experiment with original `explain-code|code-explain` two-task orders
and the complete fixed resource/selector configuration. Heat4/off, missing
flags, other budgets/orders, public text/interactive sources and altered
resources still fail before output/GPU. The off method records
`ram_preload_disabled=true`; the old on path retains its method/command/env.
Inherited DS4 variables are removed before explicit settings are forwarded.
Public windows/flow profiles, native source/binaries, model/pack and local-use
docs stay unchanged.

Source confirms that the positive budget controls the preceding GPU swap
loop, which may exit early. `DS4_SETTLE_STORE=0` then skips only hot RAM-store
list selection/preload, retaining GPU settling. No preload timer is emitted;
this is an explicitly disabled code path, not a measured0 ms. The on path
still uses heat4 for RAM preload. Timer nesting, free-only retention,
staging/read-success guards, geometry, quantization/math and protocol remain
unchanged. Entire `run_engine()` sampling/guards/cleanup and its kB-to-byte
provider are byte-identical to the preserved pre-edit version.

43 Python tests pass:12 benchmark,16 public-entry/session/profile,5 retention
constraint and10 unchanged audit tests.128 mock/synthetic preflight controls
pass:99 legacy,19 public,5 retained threshold/default and5 new on/off/absent
preload-timer controls. Prior native ASan/UBSan and SM75 build evidence is
preserved, not rerun. No new backend or numerical-parity result is claimed.

### Fixed plan and retained partial result

Criteria, original99/121 prompt IDs, cap256 and this order were fixed before
measurement: preload-on-explain-code, preload-off-explain-code,
preload-off-code-explain, preload-on-code-explain. Four processes start and
eight GEN requests are sent. No cache drop, prompt/cap rewrite, fitting,
selective rerun or extra request is used.

| Session | Complete answers | First token s: first / later | Complete request total s | Resource result |
| --- | ---: | ---: | ---: | --- |
| On: explanation, Python | 2/2 | 52.11 /51.31 | 243.73 | Pass |
| Off: explanation, Python | 2/2 | 55.81 /37.03 | 236.20 | Pass |
| Off: Python, explanation | 2/2 | 60.53 /39.28 | 250.18 | Pass |
| On: Python, explanation | 1/2 | 64.04 /63.39* | unavailable | Zero-swap guard stops final answer |

`*` The interrupted answer's first token is recovered from recorded T-event
time minus GEN-send time. This reconstruction matches all seven complete
DONE-derived first-token metrics within1e-8 s. It is a captured prefix in a
failed session, not a completed latency result. No final native prefill,
decode, request-wall or STORE measurement exists for that answer; those
fields remain null. The51 captured tokens contain two complete items and
the start of item3, without EOS/DONE. The stream checkpoint text is shorter;
all51 captured IDs are decoded separately without editing raw evidence.

Only explanation-then-Python has two successful complete sessions:243.73
versus236.20 seconds, an observed3.09% reduction. This is one pair, not a
whole-gate speedup. The reverse-order on session has no completed total.
Its285.45 seconds until forced exit cannot be ranked against a finished
session or used to estimate the missing answer.

The complete later-code comparison shows the intended tradeoff: off first
token37.03 versus on51.31 seconds, but decode takes94.029 versus89.823 seconds.
Its normalized decode is505.53 versus480.33 ms/step, while complete request
time is131.06 versus141.14 seconds. Outputs differ,187 versus188 tokens.
Part of the removed preload waiting appears during decode; no extrapolated
break-even length or steady throughput claim follows.

Both on first prompts explicitly log0 hot RAM-preload reads. First-code
output IDs are identical on/off, yet first-token/request times differ.
Those first-request differences cannot be attributed to omitted preload
reads; OS, IO and adaptive placement states remain uncontrolled. All timings
are from this gate;7ae timings are not pooled or multiplied into a gain.

### What triggered the resource stop

The final on session completes its code answer, then starts explanation99.
It logs stream38670 ms, RAM preload1533 experts/23684 ms and inclusive
settling24686 ms. During decode, the first positive engine `VmSwap` sample
appears at process elapsed282.628 seconds:

| Trigger observation | Value |
| --- | ---: |
| Engine swap | 294912 bytes =288 KiB |
| Host available at trigger | 13650202624 bytes =13.65 GB |
| GPU use excluding capacity hold | 20971257856 bytes =19.531 GiB |
| Previous sample engine swap | 0 |
| Guard-to-request elapsed observation | 86.675 s, incomplete |

Every host-available/GPU sample in that process remains inside the existing
8 GiB/20 GiB bounds. The engine-swap branch of the OR guard fires, and the
monitor deliberately terminates the process, returncode-15. Raw error says
“Host memory floor reached or engine used swap”; source/sample inspection
identifies the swap branch. The process's minimum host-available value12.14 GB
is a whole-run minimum, not the13.65 GB trigger value.

The logs do not capture swapped mapping identity, kernel/cgroup pressure or
which pages were exchanged. They do not establish that model weights were
swapped or RAM preload caused the event. No native-memory/default/OS policy
is changed on that speculation. All conditions remain fixed; the failed run
is retained rather than ignored, relaxed or replaced.

Across all four processes, observed maximum GPU use is19.531 GiB, process
HWM38.61 GB and minimum host available11.87 GB. Three whole sessions pass
all bounds; one fails zero-swap. Seven completed requests report STORE read
failures0; the interrupted request has no final STORE counters. All four
native PIDs and capacity holds are released; GPU returns to167 MiB after
cleanup. Sampling/HWM do not capture every instantaneous peak, and process
swap is not system-wide swap.

### Quality, preservation and next diagnostic gate

Seven full answers reach EOS below cap. Four complete code answers pass64/64
fixed function/vector calls, including the DONE-complete code prefix in the
later failed session. Among the seven complete answers, content6/7 and
format6/7 pass. Off-later explanation overstates VRAM capacity as a model
loading condition for RAM/SSD offload; off-first item4 has24 Han below25.
The interrupted prefix receives no final content/format judgement, and is
not counted as a completed model answer. No criteria are relaxed or causal
quality difference claimed.

`step7af-validation-runs.json` remains the original failed driver report.
`step7af-observed-runs.json` pins raw results/events/memory/logs and separates
seven completed requests from the censored answer. For the failed session,
only its DONE-complete prefix is audited through an explicit in-memory
status/returncode alias; raw process status remains failed/-15. The other
three audits stay exactly unchanged. Two new offline controls check event
clock reconstruction and the precise resource predicate. Full outputs,
manual notes, vector checks and unavailable metrics remain visible.

Only roadmap, benchmark and retention-guard Python test change. The other
211 pre-existing canonical files,99 runtime/header files, all seven native
source trees/binaries, three core archives, eight pack files and evidence
through7ae remain unchanged. Main checkout stays clean. No build/download,
weight mutation, commit, push or PR. `step7af-validation.json` indexes this
implementation, preflight checks, resource failure, partial observations,
quality review, gate patch and preservation checks.

**7af implementation/analysis and the resource failure are acknowledged;
user confirmed the 7ag diagnostic follow-up.** The original zero-swap gate
remains failed. Its raw results are not replaced by the diagnostic below.

## Step 7ag: bounded resource-failure diagnostics and one independent replay

**Failure-only diagnostics and the one-session replay are complete. Swap did
not reproduce in this attempt. No real native failure snapshot was triggered;
the original 7af288 KiB swap/page-ownership issue remains unresolved.**

This adds an opt-in Python diagnostic layer around the existing guard. It
reuses the exact 7ae native binary, SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
No native/operator/weight/pack/default/OS policy change or backend build.
The single separate replay keeps the failed7af on/code-explain configuration:
heat1, RAM preload on, GPU settle1000, chunk256,40 decimal GB LRU,20 GiB GPU,
8 GiB host floor,18 workers, context1024, original121/99 IDs and cap256.
Fresh KV remains per request; the native model/cache stays resident for two.

### Failure-only collection with the original guard intact

`--resource-diagnostics` is accepted only for the pinned heat1/preload-on/
code-explain diagnostic configuration. Invalid order, resources or settings
fail before output/GPU. The option is forwarded only to the Python runner;
its native command/environment stay exactly equal to7af. Omitted options
retain old method metadata and omit the new runner keyword. Public text and
interactive entry/profile pins remain unchanged.

The existing sample is flushed before a resource failure. Only then does an
enabled runner lazily call `resource_diagnostics.py` in a metadata helper with
a3-second timeout. It captures the owned child before termination. Helper
failure/timeout is recorded, then the same original resource error is raised.
The old host-floor/swap/GPU predicates and thresholds, exception strings,
protocol processing, engine termination and GPU-hold cleanup remain intact.
Removing only the optional parameter/helper/calls recovers the complete prior
`run_engine()` bytes; main budget-release/result finally is byte-identical.

The helper reads fixed Linux metadata paths and writes only under the
benchmark's new resource-diagnostic directory:

- Process status/stat/cgroup, maps and smaps_rollup; smaps summaries retain
  mapping headers/names and numeric metrics only for mappings reporting swap.
- Host meminfo/vmstat/memory pressure and read-only swappiness/reclaim settings.
- Available cgroup-v2 memory/current/high/max/swap/events/pressure/stat data
  from at most four hierarchy levels.

Ordinary proc reads are capped at64 KiB, maps256 KiB, smaps8 MiB and swapped
mapping summaries128 records; cgroup files16 KiB each. Truncated prefixes,
missing files and unsupported membership stay explicit, not zero. Atomic
JSON checkpoints can preserve partial metadata if the helper times out.
These are sequential post-trigger reads, not an atomic kernel snapshot;
the child may advance and counters/mappings may change during collection.
Cgroup ancestors can include other processes. No raw process-memory contents
are read, and no kernel/cgroup policy is written.

An initial CPU test found that trimming a partial last line also removed the
preceding complete line's newline. A one-expression reader fix preserves
that byte; the original failed test log is retained. No model had started.
Twelve new tests pass for bounded reads, mapping summaries, cgroup limits,
missing data, timeout/helper failure, live current-Python-process smaps access,
original guard errors and owned-child cleanup. Alongside33 shared tests,
45 Python tests pass.130 mock/synthetic controls preserve native command/env,
old profiles/retention modes and opt-in forwarding. Prior10 audit tests are
unchanged and preserved, not rerun. The live metadata test is not evidence
of a native swap incident.

### Single diagnostic outcome: not reproduced

Exactly one model process receives the original code121 then explanation99
requests. It returns0 with one READY/two DONE and two EOS-complete answers,
192/148 output tokens including EOS. Engine wall is309.44 seconds, recorded
only as diagnostic context. There is no replacement7af sample, rerun, third
request or performance estimate.

| Observed diagnostic resource | Value |
| --- | ---: |
| Memory samples | 284 |
| Maximum observed engine swap | 0 bytes |
| Maximum GPU use excluding hold | 19.531 GiB |
| Process HWM | 38.82 GB |
| Minimum host available | 13.69 GB |
| Actual guard triggers / native snapshots | 0 /0 |

Every recorded sample is within the unchanged bounds. No resource-diagnostic
field/directory appears, as expected when no guard fires. This is no failed
collection, and no real native swapped-mapping evidence was obtained. One
successful attempt with1-second/boundary sampling does not prove paging can
never recur, identify the old swapped pages or establish a fix. Output,
cache/placement and OS state can differ despite pinned inputs/settings.
The original7af error,294912 swapped bytes, partial51-token answer and failed
full comparison remain immutable.

Both complete diagnostic answers pass manual content review. Code passes16/16
fixed function/vector calls; explanation has37 Han characters in item6,
above35, so requested format passes1/2. All outputs and the original criteria
are retained. Two diagnostic answers do not establish general model quality,
long-context behavior, numerical parity or sustained throughput.

### Review, preservation and proposed next bound

Only four canonical files change: benchmark, new resource-diagnostics module,
new diagnostic test module and this roadmap. The other212 pre-existing
canonical files,99 runtime/header files, seven native source trees/binaries,
three core archives, eight pack files and evidence through7af stay unchanged.
Main checkout stays clean. Native PIDs and the capacity hold are released.
No download, native build, commit, push or PR.

`step7ag-diagnostic-run.json` pins the one command, criteria/source/binary
hashes and raw result. `step7ag-observations.json` pins events/memory/logs and
states why no native snapshot exists. Source-scope proof, tests, controls,
complete-output/AST review,16 function calls and preservation checks are
indexed by `step7ag-validation.json`.

**7ag complete; user confirmed the 7ah follow-up. Swap was not reproduced
and root cause remains unresolved.** Keep diagnostics opt-in and all defaults
unchanged. The confirmed7ah scope is:
one bounded resident diagnostic with the original code/explanation pair
repeated once, four fresh-KV requests in one model, retaining the same
binary/resources/heat1/preload-on/settle1000 and failure-only snapshot. Add
only the controlled benchmark order and no-GPU forwarding checks first.
Longer residence/cache reuse is the variable; no memory-policy tweak or
selective retry. If the event does not recur, retain that outcome without
clearing7af. The independently recorded7ah outcome follows below.

## Step 7ah: four-request resident resource diagnostic

**The original code/explanation pair was repeated once in one native model.
All four questions reached EOS; swap did not reproduce. The original7af
288 KiB failure remains open, with page ownership and kernel cause unknown.**

The single change from7ag is resident diagnostic length: two to four requests,
ordered code, explanation, code-repeat, explanation-repeat. The repeated
requests use exactly the original121/99 IDs and cap256. Each resets KV;
expert/OS caches persist. Output IDs need not be identical because native
placement/numerical paths and cache/OS state may differ.

### Implementation and fixed scope

`--resident-order code-explain-repeat` requires the complete pinned
heat1/preload-on/settle1000 configuration and `--resource-diagnostics`.
Public text/interactive inputs, incomplete resources and first-request-only
are rejected before GPU/output. A single existing runner receives all four
prepared requests. Existing orders keep their command/env/method behavior;
only this new order receives repeated-diagnostic metadata.

All existing non-main benchmark functions, including `run_engine`, remain
byte-identical. The diagnostic helper, sampling, guard predicates/thresholds,
original errors,3-second capture limit and cleanup remain byte-identical.
Reuse the exact7ae CUDA12.8 SM75 native binary SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
No math operator, native source/build, weights, pack, public defaults or OS
policy change. Limits remain20 GiB GPU,40 decimal GB LRU,8 GiB host floor,
18 workers, chunk256/context1024, heat1/on and GPU settle1000.

Three focused tests verify repeated fixtures/caps, complete opt-in guards and
public-input rejection. Together with45 existing tests,48 Python tests pass.
131 mock/synthetic controls include one four-request forwarding check and
all130 prior controls; no GPU/model is started by these checks. The prior10
audit tests are preserved, not rerun. Native/operator code did not change,
so no new native build or backend test result is claimed.

### Observed outcome and quality

Exactly one native process runs four requests, returns0 and produces one
READY/four DONE. All reach EOS with 187/153/187/152 output tokens including EOS,
fresh KV and no store read failure. Engine wall
552.65 seconds is diagnostic context only. There is no selective
retry, extra question, replacement7af sample or comparative speedup estimate.

| Observed diagnostic resource | Value |
| --- | ---: |
| Memory samples | 516 |
| Maximum observed engine swap | 0 bytes |
| Maximum GPU use excluding hold | 19.531 GiB |
| Process HWM | 39.98 GB |
| Minimum host available | 12.63 GB |
| Actual guard triggers / native snapshots | 0 /0 |

Every recorded sample remains inside the unchanged bounds. The failure-only
helper is not invoked, so no diagnostic field/directory is expected. This
absence is not a collector failure. The prior two-request replay and this
four-request success do not prove sustained stability, exclude unsampled
paging or identify the old swapped mappings.7af's error,294912-byte swap,
partial51-token final answer and failed full comparison remain immutable.

Both complete code answers are fully read and AST-reviewed before isolated
execution;32/32 fixed function/vector calls pass. Across all four answers,
content passes 3/4 and requested format 2/4.
The first explanation again treats VRAM capacity as an unconditional loading
condition without CPU/RAM/SSD offload. Both explanations have37 Han characters
in item6; the first also has36 in item3, above the fixed35 limit.

| Explanation | Han counts, items1–6 |
| --- | --- |
| long_explanation | 31, 32, 36, 33, 32, 37 |
| long_explanation_repeat | 29, 30, 34, 29, 35, 37 |

Semantic/format failures are retained in `step7ah-manual-reviews.json` and
`step7ah-answer-reviews.json`; no threshold relaxation or rerun. These four
answers do not establish general quality, numerical parity or long-context
and sustained-throughput behavior.

### Preservation and next review

Only three canonical files change: benchmark, new repeated-diagnostic test
module and this roadmap. The other214 pre-existing canonical files,
99 runtime/header files, seven native source trees/binaries, three core
archives, eight pack files and historical evidence through7ag stay unchanged.
Main checkout stays clean. Native PIDs and GPU hold are released. No download,
commit, push or PR.

`step7ah-diagnostic-run.json` pins the single command, original fixtures,
source/criteria/binary hashes and raw result. `step7ah-observations.json`
retains events/memory/log hashes, per-request context and all answer reviews.
Scope/source proof,48 tests/131 controls,32 function calls, historical
preservation and the gate patch are indexed by `step7ah-validation.json`.

**7ah complete; user confirmed the 7ai offline follow-up. Swap was not
reproduced;7af remains failed and root cause unresolved.** Keep heat1
benchmark-only and production windows as default. The confirmed7ai scope is: an offline host-memory ownership/budget audit for the
64 GB RAM/20 GiB GPU target. Trace existing RAM-store/mmap/staging/embedding/
Engram allocations; reconcile saved7af/7ag/7ah memory observations while
separating capacity, residency, reclaimability, overlaps and unknowns. Use
that evidence to select one next bounded memory/cache change. No model run,
native build, default/guard/OS-policy change in that proposed audit.
The independent offline7ai audit follows below.

## Step 7ai: offline host-memory ownership and budget audit

**The offline audit is complete.40 GB limits the expert arena, not total host
memory. Known non-store requested/full-cache capacities add2.265 GB, with
other allocations still unknown. No model was started and the old swap
cause remains unidentified.**

The reviewable ledger is [DEEPSEEK41_HOST_MEMORY_AUDIT.md](DEEPSEEK41_HOST_MEMORY_AUDIT.md).
It traces the pinned7ae native/SM75 code and pack metadata, reconciles six
saved7af/7ag/7ah sessions and separates static capacity from observed residency.
No native/operator/runtime/public entry/guard/default/OS policy change.

The fixed40-layer mixed-format geometry gives a maximum8,687,616-byte blob,
8,704,000-byte store slots,4595 slots at40 decimal GB and39,994,880,000 bytes
of allocatable arena. Its40,108,032,000-byte virtual reservation includes an
uncommitted rounded tail. Growth is64 slots/557,056,000 bytes. Slot count
records allocation capacity, not occupied experts, RSS or swap-page identity.
The current store is pageable; flow retention does not enable store zero-copy.

Known non-store capacities include1.324 GB BF16 embedding,269.255 MB full
Engram payload blocks plus unsized metadata,474.219 MB pinned flow staging,
26.063 MB swap buffers,47.202 MB chunk handoff and103.466 MB CPU work vectors.
The CPU work allocation uses capacity8 picks while actual routing uses6.
The417.006 MB GPU ring is already in GPU accounting; mapped file/page-cache
bytes are not another full private231.32 GB model copy. Small allocations,
allocator/thread/driver/kernel/file-cache/cgroup effects remain partly unknown.
No full host bound or measured2.265 GB extra RSS is asserted.

All six old sessions stay above the8 GiB sampled host floor; the minimum is
11.867 GB.14 old requests were started,13 complete and one interrupted.
Only the original7af session records294912 bytes of swap, at13.650 GB available;
its own12.137 GB session minimum is a different timestamp. The audit preserves
whole samples for each extremum and all26 saved STORE snapshots.
No original result is replaced, no performance percentage is derived and
7af's failed gate stays failed;7ag/7ah stay not reproduced.

Static review also distinguishes the wrapper's8 GiB floor/zero-swap stop from
native store growth's8 GiB availability check and its4 GiB shrink trigger
(every16 windows, target about5 GiB, trailing-chunk ownership constraints).
Native availability also considers cgroup-v2 max/current/file; wrapper logs
only host availability plus process RSS/HWM/swap. Historical mapping/pressure
and RssAnon/RssFile/locked-page data are absent. These differences do not
establish a cause or fix for the original swap event.

A full-known-capacity scenario plus8 GiB floor needs50.850 GB at40GB cache,
versus saved initial host availability50.719–52.714 GB. Its remainder is
-0.131–1.864 GB before unmeasured costs. This is capacity arithmetic, not an
observed overcommit: inactive/unfaulted pages, slot slack and live Engram rows
can make residency smaller. No conservative zero-swap40GB guarantee follows.
At36GB,4136 slots reduce arena capacity3,995,136,000 bytes and leave3.864–5.859 GB
in the same scenario. This is not a measured4GB RSS reduction or safe/faster
candidate result.32GB is arithmetic context only.

31 offline evidence checks pass over44 input hashes,40-layer expert metadata,
six saved sessions and26 STORE snapshots. Existing48 tests/131 controls and
quality reviews remain historical evidence, not rerun or newly claimed.
Only this roadmap and the new ledger change; the other216 pre-existing
canonical files and all99 runtime/header files remain byte-identical.
Seven native source trees/binaries, three core archives, eight pack files and
prior evidence through7ah stay pinned. Main checkout remains clean.
No download, native build, model run, commit, push or PR.

`step7ai-memory-audit.json` records capacity formulas, per-owner known/unknowns,
whole timed extrema, source/log hashes and counterfactual budgets.
`step7ai-audit-checks.json` records the independent saved-data consistency
checks. The patch, source/file preservation and all audit artifacts are
indexed by `step7ai-validation.json`.

**7ai complete; user confirmed the 7aj comparison. Root cause remains
unresolved.** The confirmed7aj scope is:
a benchmark-only36/40GB cache-budget comparison with the same native/pack,
heat1/preload-on/GPU1000, four original fresh-KV questions per model and all
existing resource/quality criteria. First complete no-GPU scope controls;
at most two models/eight requests,40GB then36GB, stop on any failure without
retry. One session per budget cannot establish stable performance or eliminate
OS-cache/order effects. Keep defaults/guards/OS policy unchanged and do not
replace7af or claim its fault fixed. The independently recorded7aj outcome
follows below.

## Step 7aj: bounded40/36 GB resident cache-budget comparison

**The two fixed sessions complete under the unchanged resource guards.
This is a host-headroom/cache-cost observation, not a default promotion or
resolution of7af.**

36GB reduces observed peak RSS39.987->36.701GB and increases separate-session minimum host availability12.823->15.864GB. Four-request wall493.784->535.568s (+8.46%); decode355.290->398.348ms/step (+12.12%), misses15.300->17.050/step (+11.44%) and ensure-tail266.626->311.061ms/step (+16.67%). Both use observed swap0 and19.531GiB GPU. One session per budget, all4 same-position output IDs differ; descriptive headroom/cache-cost tradeoff only.

### Narrow benchmark allowance and controls

Only the repeated four-request resource diagnostic accepts36GB: pinned
heat1/preload-on/settle1000, original121/99/121/99 IDs, cap256, fresh KV per
question and resident expert/OS caches. Public text/interactive/profile inputs,
ordinary36GB cache override, other orders, incomplete configuration and
first-request-only fail before output/GPU.32/40GB controls and public defaults
retain their prior command/env/method data.36GB adds no new method fields;
only the existing cache-size value changes. Untrusted store/floor/zero-copy
and other DS4 overrides are stripped as before.

All existing non-main functions, `run_engine`, sampling, predicates/thresholds,
original errors,3-second failure-only helper and cleanup remain byte-identical.
The exact7ae CUDA12.8 SM75 binary is reused, SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
No native/math/pack/weight/public-default/OS-policy change or native build.
Limits remain20GiB GPU,8GiB host floor,18 workers,chunk256/context1024,
GPU settle1000 and zero engine swap. Store LRU ceiling is the one variable.

Three new guard tests plus48 existing tests pass.136 mock/synthetic controls
cover all131 old controls, two40/36GB forwarding checks and three definition-only
plan success/failure/exception controls. A new test constructor initially
assumed the GPU-cap flag was already present; its ValueError was fixed by
appending absent flags. The failed log is retained; no model had started.
Only that three-test module was rerun. Prior10 audit tests remain preserved,
not rerun. No backend build/test result is newly claimed.

### Fixed sessions, timing and cache observations

Exactly two processes,40GB then36GB, each one READY/four DONE and four EOS
answers. All eight requests are preserved; no retry, extra request or cache
drop. Initial OS/cache/placement and outputs can differ. There is only one
session per budget, so the following percentages describe these observations.

| Metric | 40GB | 36GB | Observed36 minus40 |
| --- | ---: | ---: | ---: |
| Sum of four full requests, s | 493.784 | 535.568 | 8.46% |
| Sum first-token latency, s | 259.649 | 266.287 | 2.56% |
| Native prefill total, s | 259.637 | 266.277 | 2.56% |
| Native decode total, s | 234.136 | 269.283 | 15.01% |
| Decode ms/step | 355.290 | 398.348 | 12.12% |
| STORE ensure-tail ms/step | 266.626 | 311.061 | 16.67% |
| STORE misses/step | 15.300 | 17.050 | 11.44% |
| Rounded process decode physical MB/step | 152.365 | 163.559 | 7.35% |
| Whole-request process physical GB | 224.347 | 239.937 | 6.95% |
| Post-prompt RAM preload total, s | 93.177 | 95.257 | 2.23% |
| Engine lifetime incl startup/cleanup, s | 540.607 | 582.098 | 7.67% |


The first/later/repeat cohorts distinguish a fresh process's first question
from later resident-cache requests. Later includes indices1–3; repeat includes2–3 and
is a subset, not another independent sample.

| Cohort | Request total change | Decode ms/step change | Ensure-tail/step change | Misses/step change |
| --- | ---: | ---: | ---: | ---: |
| first | 7.66% | 9.20% | 12.28% | 12.15% |
| later | 8.80% | 13.76% | 19.21% | 11.01% |
| repeat | 5.64% | 7.08% | 10.23% | 12.00% |

| Request | Output tokens40/36 incl EOS | Wall seconds40/36 | First-token seconds40/36 | Decode ms/step40/36 | Same output IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 /long_python | 187/192 | 147.240/158.523 | 62.480/63.476 | 455.706/497.634 | False |
| 1 /long_explanation | 147/149 | 121.263/139.054 | 65.037/67.618 | 385.108/482.680 | False |
| 2 /long_python_repeat | 187/192 | 126.566/134.472 | 68.386/72.253 | 312.797/325.752 | False |
| 3 /long_explanation_repeat | 142/147 | 98.715/103.519 | 63.746/62.939 | 248.006/277.944 | False |

Smaller capacity is accompanied by more decode misses, ensure-tail and process
reads. First flow retains4498/4136 blobs, and36GB logs362 no_room events;
later retained0 under both full allocated capacities. Post-prompt RAM preload
totals93.177/95.257s, so smaller capacity does not yield a measured total
preload saving here. These observations are consistent with cache pressure,
without isolating all time to SSD or eliminating output/placement/OS/order
effects.

Decode uses output tokens minus the first prefill-produced token. STORE
ensure-tail includes completion/mutex/bookkeeping and nests in CPU-lane time;
it is not a pure SSD/GPU-idle timer. Submitted bytes are logical payload.
Final STAT disk_mb is a rounded3-decimal cumulative process physical MB/fed
step, weighted here by decode steps; it includes Engram, promotions and other
paths. Whole-request read_bytes also includes prefill. Copy nests in stream,
RAM preload nests in settle, and the1000ms limit only bounds the GPU swap
loop. Nested timers are not added as independent latency components.

### Resources, quality and limits

40GB has4595 maximum slots;36GB has4136, a3,995,136,000-byte capacity reduction.
Logged slots remain allocated capacity, not ready occupancy or resident RAM.
Separate session extrema below are not a simultaneous ownership snapshot.

| Budget | Memory samples | Peak RSS GB | HWM GB | Min host available GB | Max GPU GiB | Max engine swap B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 GB | 499 | 39.987 | 39.987 | 12.823 | 19.531 | 0 |
| 36 GB | 532 | 36.701 | 36.701 | 15.864 | 19.531 | 0 |

Every recorded sample stays inside the original bounds. No resource guard
fires, so no failure-only helper/native snapshot is expected or produced.
This is not a collector failure or proof that paging cannot recur.7af's
288KiB swap, trigger/error, partial last answer and failed on/off comparison
remain immutable; none of these two runs replaces that missing sample.

All complete outputs are manually read and code AST/calls approved before
isolated function/vector execution. Original content/format/completion
criteria remain separate; all failures are retained.

| Budget | EOS | Content | Format | Function/vector calls |
| --- | ---: | ---: | ---: | ---: |
| 40 GB | 4/4 | 4/4 | 4/4 | 32/32 |
| 36 GB | 4/4 | 3/4 | 3/4 | 32/32 |

40GB passes4/4 content and4/4 format.36GB passes3/4 content and3/4 format: its first explanation lists gradients without a training qualification in a local-inference context; its repeated explanation item6 has37 Han characters above35. Both failures are retained, with no causal cache/quality attribution. All4 code answers pass64/64 fixed function/vector calls. Outputs total663/680 tokens including EOS,659/676 decode steps; raw latency and normalized decode are both reported.

Eight bounded answers do not establish general quality, numerical parity,
long-context behavior or sustained throughput. Different output IDs and
OS/placement/order effects prevent a stable or isolated cache-only speed claim.

### Preservation and next review

Only three canonical files change: benchmark, new cache-budget test module
and this roadmap. The other216 pre-existing canonical files,99 runtime/header
files, seven native source trees/binaries, three core archives, eight pack
files and historical evidence through7ai stay pinned. Main checkout stays
clean; native PIDs and GPU holds are released. No download, native build,
commit, push or PR.

`step7aj-comparison-run.json` pins order, commands/source/criteria/assets and
both results. `step7aj-observations.json` retains complete per-request/cache/
read/memory/quality data and the comparison limits. Tests, failed test log,
controls, source review, manual approvals, function checks, patch and historical
preservation are indexed by `step7aj-validation.json`.

**7aj complete; user confirmed the7ak reverse-order follow-up.** Keep public defaults and both heat1 budgets benchmark-only. The confirmed scope is: repeat this bounded comparison in reverse order36GB then40GB, using the same original four questions/configuration per model. At most2 models/8 requests; stop on any failure, no retry. This checks order sensitivity before choosing whether to expose a memory-headroom preset; it does not establish a safe cap or close7af.
The independently recorded7ak outcome follows below.

## Step 7ak: reverse-order36/40 GB cache-budget check

**The two fixed sessions complete under the unchanged resource guards.
This is a host-headroom/cache-cost observation, not a default promotion or
resolution of7af.**

In reverse order36->40GB,36GB lowers observed peak RSS40.091->36.571GB and raises separate-session minimum host availability12.666->16.069GB. Four-request total493.062->520.012s (+5.47%); decode344.865->380.271ms/step (+10.27%), misses14.920->16.855/step (+12.97%) and ensure-tail258.556->292.827ms/step (+13.25%). Both budgets observe swap0 and19.531GiB GPU. All4 same-position output IDs differ; these are bounded observations, not a stable speed/safety estimate.

### Reused implementation and new order controls

Only the repeated four-request resource diagnostic accepts36GB: pinned
heat1/preload-on/settle1000, original121/99/121/99 IDs, cap256, fresh KV per
question and resident expert/OS caches. Public text/interactive/profile inputs,
ordinary36GB cache override, other orders, incomplete configuration and
first-request-only fail before output/GPU.32/40GB controls and public defaults
retain their prior command/env/method data.36GB adds no new method fields;
only the existing cache-size value changes. Untrusted store/floor/zero-copy
and other DS4 overrides are stripped as before.

All existing non-main functions, `run_engine`, sampling, predicates/thresholds,
original errors,3-second failure-only helper and cleanup remain byte-identical.
The exact7ae CUDA12.8 SM75 binary is reused, SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
No native/math/pack/weight/public-default/OS-policy change or native build.
Limits remain20GiB GPU,8GiB host floor,18 workers,chunk256/context1024,
GPU settle1000 and zero engine swap. Store LRU ceiling is the one variable.

All canonical benchmark, test, runner, public-entry and native files are
unchanged. Prior7aj51 tests/136 controls are verified and preserved, not
rerun. Five fresh no-GPU controls check both pinned command/env/method/ID
forwardings and reverse-order success/first-failure/exception sequencing.
The ignored driver differs only in artifact prefixes,36->40 session order
and checks for these controls. No new native build or backend test is claimed.

### Fixed sessions, timing and cache observations

Exactly two processes,36GB then40GB, each one READY/four DONE and four EOS
answers. All eight requests are preserved; no retry, extra request or cache
drop. Initial OS/cache/placement and outputs can differ. There is only one
session per budget, so the following percentages describe these observations.

| Metric | 40GB | 36GB | Observed36 minus40 |
| --- | ---: | ---: | ---: |
| Sum of four full requests, s | 493.062 | 520.012 | 5.47% |
| Sum first-token latency, s | 253.038 | 267.893 | 5.87% |
| Native prefill total, s | 253.029 | 267.883 | 5.87% |
| Native decode total, s | 240.026 | 252.120 | 5.04% |
| Decode ms/step | 344.865 | 380.271 | 10.27% |
| STORE ensure-tail ms/step | 258.556 | 292.827 | 13.25% |
| STORE misses/step | 14.920 | 16.855 | 12.97% |
| Rounded process decode physical MB/step | 145.273 | 163.122 | 12.29% |
| Whole-request process physical GB | 223.242 | 237.281 | 6.29% |
| Post-prompt RAM preload total, s | 91.275 | 97.897 | 7.25% |
| Engine lifetime incl startup/cleanup, s | 540.230 | 566.139 | 4.80% |


The first/later/repeat cohorts distinguish a fresh process's first question
from later resident-cache requests. Later includes indices1–3; repeat includes2–3 and
is a subset, not another independent sample.

| Cohort | Request total change | Decode ms/step change | Ensure-tail/step change | Misses/step change |
| --- | ---: | ---: | ---: | ---: |
| first | 9.40% | 7.55% | 10.04% | 11.25% |
| later | 3.84% | 10.24% | 13.35% | 12.41% |
| repeat | 5.36% | 11.34% | 15.31% | 13.52% |

| Request | Output tokens40/36 incl EOS | Wall seconds40/36 | First-token seconds40/36 | Decode ms/step40/36 | Same output IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 /long_python | 187/192 | 144.275/157.834 | 58.276/62.858 | 462.362/497.260 | False |
| 1 /long_explanation | 166/139 | 122.020/123.264 | 62.793/68.306 | 358.952/398.250 | False |
| 2 /long_python_repeat | 193/187 | 124.889/135.062 | 67.452/72.167 | 299.154/338.147 | False |
| 3 /long_explanation_repeat | 154/149 | 101.879/103.852 | 64.517/64.563 | 244.198/265.468 | False |

All8 requests use the native flow path. First flow retains4136/4498 blobs at36/40GB, with362/0 no_room events; later retained0 under full allocated capacities.36GB has more decode misses, ensure-tail and process reads, and RAM preload97.897s versus91.275s at40GB. No invalid handoffs or STORE read failures are observed. This is consistent with cache pressure, without attributing all time to SSD or treating allocated slots as occupancy/RSS.

Decode uses output tokens minus the first prefill-produced token. STORE
ensure-tail includes completion/mutex/bookkeeping and nests in CPU-lane time;
it is not a pure SSD/GPU-idle timer. Submitted bytes are logical payload.
Final STAT disk_mb is a rounded3-decimal cumulative process physical MB/fed
step, weighted here by decode steps; it includes Engram, promotions and other
paths. Whole-request read_bytes also includes prefill. Copy nests in stream,
RAM preload nests in settle, and the1000ms limit only bounds the GPU swap
loop. Nested timers are not added as independent latency components.

### Direction versus the prior order

The prior7aj order is40->36GB; this gate is36->40GB. Each gate is analyzed
separately. The two runs per budget across both gates are not a randomized
stable estimate, and later/ repeat cohorts reuse each process's own cache.
Dates, output IDs and OS/placement/thermal/background state remain possible
confounders; no cache drop, selective replacement or pooling as a repair of7af.

| Cohort | Request change prior/current | Decode ms/step change prior/current | Misses/step change prior/current |
| --- | ---: | ---: | ---: |
| all | 8.46% /5.47% | 12.12% /10.27% | 11.44% /12.97% |
| first | 7.66% /9.40% | 9.20% /7.55% | 12.15% /11.25% |
| later | 8.80% /3.84% | 13.76% /10.24% | 11.01% /12.41% |
| repeat | 5.64% /5.36% | 7.08% /11.34% | 12.00% /13.52% |

The prior40->36GB gate observes36GB request-time +8.46%, normalized decode +12.12%, RSS -3.286GB and minimum available +3.042GB. The reversed36->40GB gate observes +5.47%, +10.27%, -3.519GB and +3.403GB respectively. The headroom/cost directions agree, while percentages and outputs differ; this does not isolate order effects or prove a safe cap. Gates started2026-10-27 09:00:38 and09:26:45 UTC. Keep them separate, without replacing7af or pooling them into a universal gain.

### Resources, quality and limits

40GB has4595 maximum slots;36GB has4136, a3,995,136,000-byte capacity reduction.
Logged slots remain allocated capacity, not ready occupancy or resident RAM.
Separate session extrema below are not a simultaneous ownership snapshot.

| Budget | Memory samples | Peak RSS GB | HWM GB | Min host available GB | Max GPU GiB | Max engine swap B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 GB | 523 | 36.571 | 36.571 | 16.069 | 19.531 | 0 |
| 40 GB | 504 | 40.091 | 40.091 | 12.666 | 19.531 | 0 |

Every recorded sample stays inside the original bounds. No resource guard
fires, so no failure-only helper/native snapshot is expected or produced.
This is not a collector failure or proof that paging cannot recur.7af's
288KiB swap, trigger/error, partial last answer and failed on/off comparison
remain immutable; none of these two runs replaces that missing sample.

All complete outputs are manually read and code AST/calls approved before
isolated function/vector execution. Original content/format/completion
criteria remain separate; all failures are retained.

| Budget | EOS | Content | Format | Function/vector calls |
| --- | ---: | ---: | ---: | ---: |
| 40 GB | 4/4 | 4/4 | 3/4 | 32/32 |
| 36 GB | 4/4 | 3/4 | 4/4 | 32/32 |

36GB passes3/4 content and4/4 format: repeated explanation item2 makes VRAM capacity an unconditional condition for running on GPU without host/offload qualification.40GB passes4/4 content and3/4 format: first explanation item6 has37 Han characters above35. Failures are independent and retained; no causal budget/quality claim. All4 code answers pass64/64 function/vector calls. Output totals700/667 tokens including EOS and696/663 decode steps at40/36GB; normalized decode is reported beside full latency.

Eight bounded answers do not establish general quality, numerical parity,
long-context behavior or sustained throughput. Different output IDs and
OS/placement/order effects prevent a stable or isolated cache-only speed claim.

### Preservation and next review

Only this roadmap changes canonically. The other218 pre-existing canonical
files,99 runtime/header files, seven native source trees/binaries, three core
archives, eight pack files and evidence through7aj stay pinned. Main checkout
stays clean; native PIDs and GPU holds are released. No download, native build,
commit, push or PR.

`step7ak-comparison-run.json` pins order, commands/source/criteria/assets and
both results. `step7ak-observations.json` retains complete per-request/cache/
read/memory/quality data and the comparison limits. The five new controls,
source review, manual approvals, function checks, patch and historical
preservation are indexed by `step7ak-validation.json`.

**7ak complete; the user confirmed7al.** The authorized follow-up exposes the
measured36GB/heat1 native configuration as an explicit memory-experimental
local text profile, preserving windows/flow defaults and guards. The scope
requires no-GPU input/pin/streaming/cleanup checks before at most2 smoke models
and3 accepted requests. This is a RAM-headroom choice, not a speed/default
promotion or7af root-cause fix. Its independently retained outcome follows.

## Step 7al: opt-in36GB local text profile

### Implemented scope and compatibility

The public entry adds explicit `--profile memory-experimental`. It selects
exact7ae retention binary SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`,
36 decimal GB LRU,heat1,chunk256,flow/on,mixed/first-window/off,RAM preload/on
and GPU-settle1000ms. It shares the existing20GiB GPU target,8GiB host
MemAvailable floor,zero engine swap requirement,18 workers,context1024,
cap1–256,fresh KV per question and cleanup.1000ms bounds the swap loop;
it does not cap subsequent RAM preload or request time.

The profile requires its exact engine/pack pins and complete configuration
before loading. Prompt,file,stdin and resident input reuse the existing
streaming/tokenizer/protocol. Ambient DS4 overrides are filtered. It enables
the existing3-second best-effort failure-only metadata collector without
changing stop conditions or cleanup. Normal success creates no diagnostic
snapshot. No native failure was induced here; no new native failure-capture
claim follows.

`windows` remains default. Legacy windows/flow-experimental and benchmark
commands,environments and method records pass133 rerun compatibility controls.
The shared run_engine,stream/input classes,signal/runtime-error cleanup tail,
diagnostic module and budget-release finally block remain byte-identical.
The native operators,source,binaries and three core archives are unchanged.
No build,download,commit,push,PR or OS-policy change was performed.

### Verification and real public-entry smoke

Before smoke,57 Python tests and148 no-GPU controls pass:133 prior controls
rerun,12 new real-asset mock input/pin/resource/cleanup controls,and3 new
smoke-plan stop-on-failure controls. The57 tests include6 new profile tests.
Mock/synthetic faults check existing stop/cleanup handling,not an observed
native swap reproduction. Ten old offline audit tests are preserved,not rerun.

The fixed plan uses at most2 native processes/3 accepted requests,with no
retry and immediate cancellation of remaining runs after a failure. All runs
complete successfully. The first uses the script's absolute path from `/tmp`.
The second ignores a blank line,rejects one6000-character over-context input,
then answers the original99/121-token tasks in one resident process and exits
via `/exit`. The rejected input sends no native generation request. Each
accepted question has fresh KV. CLI stdout exactly matches saved completions;
both native processes and GPU holds are released.

| Mode/task | Input token IDs | Cap | Output tokens including EOS | Request to first token | Request wall time | Decode rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single RAM-cache sentence |18|64|34|26.92s|49.44s|1.47tok/s|
| Resident memory explanation |99|256|151|55.57s|113.76s|2.58tok/s|
| Resident two Python functions |121|256|192|58.03s|154.47s|1.98tok/s|

Input lengths are independently frozen public template token counts. All three
native paths select flow. Every answer reaches EOS within its cap; content
and requested format pass3/3. The six explanation items contain30,30,33,27,35
and35 Han characters within the fixed25–35 range. Both Python functions and
the full AST/call list are manually inspected before isolated execution;
all16 function/vector calls pass,including empty,odd-only,negative,duplicate,
zero,large integers and input preservation. One optional Python fence follows
the previously fixed acceptance rule. No quality criterion was relaxed.

The single/resident model startup-to-ready times are42.64/32.12s,and full
engine wall times94.10/303.94s. Single/resident peak RSS is23.70/35.37GB and
minimum host MemAvailable28.88/17.41GB,all decimal. Maximum observed GPU use
excluding the2GiB capacity hold is19.53GiB;engine swap stays0 and STORE read
failures stay0.36GB's STORE slots_max4136 is allocated capacity,not measured
occupancy or whole-process RSS.1-second/boundary samples cannot exclude
transient peaks. No resource failure means0 diagnostic snapshots,not a failed
collector. This three-question functionality smoke makes no performance
comparison or sustained safety claim.

### Review boundary and remaining V1 work

The profile offers the measured RAM-headroom choice to the user. The prior
7aj/7ak same-engine36/40GB comparisons retain their own different output IDs,
order and quality failures. Their5.47–8.46% fixed-workload latency cost is not
a prediction against default windows or old flow-experimental. It is not a
stable universal speed result or proof that36GB prevents swap. The original
7af288KiB swap interruption,partial answer and unknown page ownership remain
unchanged; the incomplete preload A/B remains incomplete.

Canonical changes are the benchmark/profile validator,public entry,new6-test
profile module,local run guide and this roadmap:5 files.215 other pre-existing
canonical files and all99 runtime/header files stay unchanged. Historical
raw evidence through7ak,seven native source trees/binaries,three archives
and eight pack files are preserved. `step7al-smoke.json` pins the public
commands,source,assets and results; `step7al-observations.json` retains the
request/resource/audit data. Manual reviews,function checks,all test/control
logs,the scoped patch and historical preservation are indexed by
`step7al-validation.json`.

**7al complete;the user confirmed7am.** The authorized follow-up is one
resident36GB model,at most8 accepted questions across4 frozen task families
repeated once,cap256/context1024 and a30-minute total ceiling. It stops on any
runtime/guard/driver failure without retry and retains partial evidence.
Its independently retained outcome follows;larger context is outside7am.

## Step 7am: bounded eight-question resident stability and quality

### Fixed scope and verification

This gate reuses the unchanged7al public `memory-experimental` entry and exact
7ae retention binary SHA256
`d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`.
36 decimal GB LRU,heat1,preload/on,chunk256,GPU-settle1000ms,18 workers and
20GiB GPU/8GiB host-floor/zero-swap guards stay unchanged. No runtime,operator,
native source,binary,build,weight,default or OS-policy change occurs.

One resident process receives explanation99,Python121,one-line warehouse264
and multi-step arithmetic97 public-template token IDs,in that order repeated
once. Each question has fresh KV and cap256 within context1024. Inputs,IDs,
criteria and order are frozen before native startup. The existing3-second
failure-only collector is retained. There is no cache drop,selective rerun
or second native process.

Before starting,18 selected existing Python regressions pass:6 memory-profile
and12 diagnostic tests. Seven new no-GPU driver controls cover one launch,
nonzero return,deadline/graceful and forced cleanup,exception cleanup,owned-PID
signal refusal and input/context bounds. The30-minute total ceiling reserves
15s for cleanup;generation is cancelled by the remaining1785s deadline.
Native and wrapper sessions differ,so emergency cleanup verifies the recorded
native PID's parent/executable instead of assuming wrapper-group termination
also kills native. These fault paths are synthetic controls,not real native
fault captures. The previous7al57-test/148-control suite remains preserved;
18 tests are rerun here,39 tests and148 controls are not rerun.

### Observed outcome

One native model answers8/8 questions with EOS;the session closes normally
through `/exit`. Public stdout matches saved completions exactly. Wrapper
wall time is896.37s (14m56s),engine wall894.79s and startup-to-ready41.24s.
The30-minute deadline and emergency cleanup are not triggered. There are806
output tokens including8 EOS and798 decode steps;all eight native paths
choose flow. This is a single bounded functionality/stability run,not an A/B.

|Request|Task|Input tokens|Output including EOS|First token|Request wall|Decode tok/s|Content/format|
|---|---|---:|---:|---:|---:|---:|---|
|1|long_explanation|99|152|55.53s|113.07s|2.62|fail/fail|
|2|long_python|121|187|57.70s|149.22s|2.03|pass/pass|
|3|long_retrieval|264|32|98.05s|119.30s|1.46|pass/pass|
|4|multi_arithmetic|97|38|45.51s|59.47s|2.65|pass/fail|
|5|long_explanation|99|140|62.12s|104.20s|3.30|fail/pass|
|6|long_python|121|187|70.57s|142.02s|2.60|pass/pass|
|7|long_retrieval|264|32|93.34s|110.89s|1.77|pass/pass|
|8|multi_arithmetic|97|38|40.21s|52.06s|3.12|pass/fail|

Across842 saved1-second/boundary samples,peak RSS/HWM is37,772,279,808 bytes
(37.77 decimal GB);minimum host MemAvailable15,143,473,152 bytes (15.14GB).
Maximum GPU use excluding the2GiB capacity hold is19.53GiB. Engine swap and
STORE read failures stay0;all observed samples satisfy existing bounds.
First-request decode reaches4136 allocated STORE slots,also the36GB maximum.
That is capacity,not occupancy or RSS. Whole-process memory includes more
than the cache. Per-request RSS peaks vary and the last peak is below request7;
this run does not establish a leak-free plateau or a universally safe cap.
Samples cannot exclude transient peaks. Normal success creates0 diagnostic
snapshots;no native failure-capture validation is claimed.

### Quality findings and limits

Content passes6/8,strict requested format5/8,with0 unassessed EOS answers.
All eight complete texts are manually read. Both complete Python answers and
AST/call lists are approved before isolated execution;32/32 function/vector
calls pass,including input preservation and the fixed empty/odd/negative/
zero/duplicate/large-integer cases. Both six-field warehouse JSON objects
are exact,with unique requested keys and integer quantities45/17/28.
Both arithmetic answers correctly calculate8 groups,16 net and21 with spare.

Explanation requests1 and5 say VRAM capacity decides whether a model can load
without qualifying the GPU-resident working part or host offload. They fail
the same content rule retained in7ak. Request1 has Han counts29,32,36,33,34,37;
items3 and6 exceed35. Request5 counts29,32,32,28,32,31 and passes format.
Arithmetic requests4 and8 provide correct equations after the numbers,but
no Chinese explanatory prose;the frozen rule scores format false. The prompt
asks a brief calculation explanation without explicitly saying Chinese prose;
this stricter preset scoring limit is retained transparently,not reported as
an arithmetic error. No quality criterion is relaxed after seeing output.

The repeated explanation has different output IDs;the other three repeated
answers have identical IDs. Neither this small workload nor those matches
establish numerical parity,general instruction following or causal cache/
quantization-quality effects. First-token waits40.21–98.05s and decode rates
1.46–3.30tok/s remain practical limits. There is no sustained/all-day stability,
full-context quality or old7af root-cause resolution. The original288KiB swap
interruption and incomplete preload A/B remain independently failed/open.

### Preservation and next review

Only this roadmap and local-run guide change canonically. The other218
pre-existing canonical files and all99 runtime/header files stay unchanged.
Evidence through7al,seven native source trees/binaries,three core archives
and eight pack files remain pinned. Main checkout stays clean;the native
PID and GPU hold are released. No download,build,commit,push or PR.
`step7am-session.json` pins the unique command,source,inputs,criteria and raw
result. Observations,842 samples,manual approvals,32 function checks,18-test
logs,7 controls,scoped patch and historical preservation are indexed by
`step7am-validation.json`. Quality failure and resource success are recorded
separately;neither is substituted for the old7af result.

**7am complete;the user confirmed7an.** The authorized follow-up is an
offline-only context/host/GPU budget audit. It preserves the1024 limit,
profiles,guards and raw failed/successful evidence;no model execution,build,
download or context promotion. Its independent outcome follows.

## Step 7an: offline context and memory-budget audit

The current1024 boundary is enforced by the local tokenizer budget,READY
check,method record and inherited native argv. The reused native engine
already accepts --max-context. This gate audits fixed source/pack/saved
records without changing runtime or launching a model. Detailed formulas,
source lines,units,capacity tables and limitations are in
[the context audit](DEEPSEEK41_CONTEXT_AUDIT.md).

Four shared persistent sources(2/8/14 ratio2,20 ratio1) hold compressed KV
and index keys on the single GPU. Their rows are288/80bytes. Eight query
source layers reuse four key caches. Typed CUDA allocations align256.
At fixedchunk256,1024→2048 adds1,605,632bytes(1.53125MiB) to the known
context-sensitive GPU allocation subset. Actual GPU use can change through
expert placement after scratch allocation;this is not a measured peak.

Longer actual prompts can enter native layer-major processing. Each chunk
state slot is23,601,152bytes. At cap256,the maximum prompt grows768→1792,
three→seven slots,adding94,404,608bytes(90.03MiB) if LM is enabled. Its
conditional temporary layer cache is bounded above by one full largest
384-expert layer,3,336,044,544bytes. The264-token7am requests already used
2-chunk LM with a rounded3.0GB layer-cache log. This cache is not newly
introduced by2048;unknown queues/events/vector/allocator/driver costs remain.
Native RAM/IO gates may decline LM. State/cache are freed before settling.

A hypothetical36GB full-store+known7ai nonstore capacities+2k max-prompt LM
state+full largest layer+8GiB floor totals50,356,250,824bytes. The saved7am
initial host available52,493,484,032 leaves2.14GB for omitted costs in that
scenario,not a safe cap,actual RSS forecast or changed admission rule.
Native default context32768 and pack metadata1048576 are not hardware
validation. Default YOCO skipping may activate at3692 prompt tokens;
2048 stays below it while4096/cap256 can cross it. This supports testing2k
first while keeping long-prefix correctness separately open.

45 offline checks pass against22 pinned inputs and7 native source files.
No runtime unit tests,GPU allocation/quality measurements,native model,
request,build,download or OS-policy change occurs. The corrected audit is
cross-checked against all40 old expert blobs;an earlier draft missing two
per-role8KiB gaps is retained transparently. Only this roadmap and the new
context-audit document change canonically.219 other pre-existing files and
all99 runtime/header files remain unchanged. Historical evidence through7am,
native source/binaries,archives and pack are pinned;main checkout stays clean.
Artifacts and the scoped patch are indexed by `step7an-validation.json`.
No old7af failure is replaced or resolved.

**7an complete,awaiting user confirmation.** Proposed7ao implements explicit
--context1024|2048 only for memory-experimental,default1024. Thread the selected
capacity through all input paths,native argv,strict READY and method metadata;
retain legacy commands/env/pins,output1–256,20GiB/8GiB/zero-swap guards and
cleanup. Complete no-GPU compatibility/boundary/error tests first. Then at
most1 native model/3 fresh-KV questions(cap128,total30min):short control,
pre-frozen>1024-token multi-position retrieval(aim1400–1700),same long question
repeated. Stop on any runtime/guard/driver/configuration failure,no retry;
retain partial output and bounded failure metadata. No build/default/context
promotion,performance A/B,4096 validation or7af fix.7ao has not started.
Stop here for this step's review.

## Next review gates

These are proposed steps, not authorization to run them without confirmation.

| Step | Deliverable | Acceptance gate | Status |
| --- | --- | --- | --- |
| 1 | Branch and isolated worktree | Main checkout clean; baseline reader tests pass | Complete |
| 2 | Core metadata and expert byte access | Correct original expert IDs, shard offsets and bounded copies; bad files rejected | Complete; user confirmed |
| 3 | One expert through existing native compute | Actual 5120/2304 expert passes an independent reference with the declared MXFP4/Q8 baseline | Complete on CPU; user confirmed |
| 3b | Same expert on the consumer GPU | Check existing GPU kernel formats and rounding; compare the real expert against the CPU baseline on the RTX 2080 Ti | Complete; user confirmed |
| 4 | Expert cache and original top-6 execution | Same GPU result on cold/warm/eviction; bounded staging/device buffers; measure bytes moved | Complete; user confirmed |
| 5a | Shared expert and complete text MoE | Real shared weights match the declared reference; complete MoE preserves routing and cache invariance | Complete; user confirmed |
| 5b | mHC and normalization around MoE | Fixed 1017 residual mixing/normalization matches a pinned reference; compose with MoE | Complete; user confirmed |
| 5c | Layer-0 attention and KV state | Validate position/mask behavior and incremental state against the reference | Complete; user confirmed |
| 5d | Assemble one native decoder layer | Combine confirmed components; validate short prefill and decode | Complete; user confirmed |
| 6a | Layer-2 ratio-2 KV compressor | Correct projection, per-channel pooling, group positions and incomplete-group state | Complete; user confirmed |
| 6b | Compressed KV post-processing and bounded cache | Fixed compressed RoPE, group-16 FP4/E4M3 conversion and correct append/reset positions | Complete; user confirmed |
| 6c | Layer-2 indexer key generation and cache | Project/normalize unrotated latents, apply indexer RoPE and group-32 FP4/E8M0 conversion, validate append/reset | Complete; user confirmed |
| 6d | Layer-2 indexer query and causal selection | Validate query projection, score reduction and top-k visibility against the pinned reference | Complete; user confirmed |
| 6e | Layer-2 attention over window and selected compressed KV | Combine both KV sources with correct indices, mask, sink and output math | Complete; user confirmed |
| 6f | Complete layer-2 attention projections and orchestration | Connect Q/KV/output projections to the confirmed compressor, indexer and core; preflight cache capacity; validate prefill/decode | Complete; user confirmed |
| 6g | Share compressed KV and indices across layers | Establish layer-2 source/layer-3 consumer ownership, positions and reset; reject stale shared state | Complete; user confirmed |
| 6h | Assemble the layer-2 decoder | Join confirmed attention with mHC, text MoE and residual paths; validate prefill/decode/reset | Complete; user confirmed |
| 6i | Assemble the layer-3 decoder and the layer-2/3 pair | Pass residuals, delayed coefficients and shared attention state between the two full decoders; validate prefill/decode/reset | Complete; user confirmed |
| 6j | Layer-1 Engram lookup and sequence state | Pin the table/lookup layout and causal token-state contract; validate bounded real rows against the reference before assembling layer 1 | Complete; user confirmed |
| 6k | Layer-1 Engram projection and residual gate | Consume confirmed lookup values; validate original projection, stream-wise normalization and gated residual output before decoder assembly | Complete; user confirmed |
| 6l | Assemble the layer-1 decoder with Engram | Apply Engram before mHC/attention/MoE with correct token history and delayed coefficients; validate prefill/decode/reset | Complete; user confirmed |
| 6m | Connect the complete layer-0/1/2/3 text path | Pass actual residuals and same-token coefficients through all four decoders, with Engram history and layer-2/3 shared state; validate prefill/decode/reset | Complete; user confirmed |
| 6n | Extend the shared-consumer path through layer 7 | Reuse the layer-2 KV/index publication for layers 4–7, preserving actual handoffs, capacity checks and reset behavior | Complete; user confirmed |
| 6o | Add the layer-8 KV/index source | Establish the next source identity, weights/cache/publication and reset; reject cross-group state while preserving layer-0..7 behavior | Complete; user confirmed |
| 6p | Extend the source-8 group through layer 13 | Reuse the shared-consumer path for original layers 9–13 and connect actual handoffs while preserving both source groups and reset behavior | Complete; user confirmed |
| 6q | Layer-14 Engram layout and lookup | Pin the second table and original hash/sequence contract; reuse the Engram reader and validate bounded real rows before decoder assembly | Complete; user confirmed |
| 6r | Layer-14 Engram projection and residual gate | Reuse the confirmed projection/gate implementation with original layer-14 weights; validate the standalone Engram boundary before decoder assembly | Complete; user confirmed |
| 6s | Add the layer-14 KV/index source | Extend the confirmed source components to original source 14; validate weight/cache/publication identity and sequence behavior with the explicit independent reference | Complete; user confirmed |
| 6t | Assemble the layer-14 decoder with Engram | Join confirmed Engram and source-14 attention with mHC/MoE; validate standalone prefill/decode/reset before extending the prefix | Complete; user confirmed |
| 6u | Connect the prefix through original layer 14 | Feed the actual layer-13 residual and delayed coefficients into the new Engram decoder; validate prefill/decode/reset and preserve existing source groups | Complete; user confirmed |
| 6v | Extend the source-14 group through layer 19 | Reuse source-14 publication for original consumers 15–19; validate identities, real handoffs and whole-prefix reset | Complete; user confirmed |
| 6w | Establish the layer-20 source boundary | Pin ratio-1 compression and candidate-state semantics before extending the native decoder path | Complete; user confirmed |
| 6x | Native layer-20 ratio-1 compressor | Reuse native projection/normalization infrastructure; compare BF16 projection and pre-RoPE output against the pinned source-20 fixture | Complete; user confirmed |
| 6y | Source-20 ratio-1 compressed KV | Reuse RoPE/FP4 conversion with one-token positions; validate bounded append/reset before indexer integration | Complete; user confirmed |
| 6z | Source-20 ratio-1 index keys | Reuse projection/normalization and index-key RoPE/FP4 conversion; validate a bounded cache from unrotated source-20 latents | Complete; user confirmed |
| 6aa | Source-20 index queries and candidate selection | Reuse query/scoring components; validate causal top-512 and pinned candidate-block rules before shared publication | Complete; user confirmed |
| 6ab | Source-20 attention assembly and shared state | Connect projections, bounded KV/index caches and sparse attention; publish indices/candidates with ownership, position and reset checks | Complete; user confirmed |
| 6ac | Source-20 decoder assembly | Join confirmed source-20 attention with mHC, MoE and residual paths; validate prefill/decode/reset | Complete; user confirmed |
| 6ad | Source-20 group and prefix integration | Reuse source-20 publication for consumers 21–23; connect the group with actual residual/coefficient handoffs and reset | Complete; user confirmed |
| 6ae | Candidate-consuming source24 and group24–27 | Own index queries constrained to source20 candidates; reuse source20 KV/keys, validate ownership and connect the group | Complete; user confirmed |
| 6af | Reuse the candidate index path for original28–31 | Connect the next index-source group through original weights/configuration, preserving source20 ownership and earlier groups | Complete; user confirmed |
| 6ag | Reuse the candidate index path for original32–35 | Connect the next group through original weights and configuration; preserve source20 ownership and earlier candidate groups | Complete; user confirmed |
| 6ah | Complete the original backbone through layer39 | Preserve source20 and all candidate groups; verify ordinary37–39 versus separate MTP40–42, and full40-layer residual/coefficient execution | Complete; user confirmed |
| 6ai | Embedding and output head | Map token IDs to the residual input and final residual/coefficient outputs to logits using pinned weights and independent references | Complete; user confirmed |
| 6aj | Original-weight token-to-logits integration | Supply actual embedding inputs through all 40 original backbone layers; verify actual expert coverage, logits and state before short text generation | Complete; user confirmed |
| 6ak | Minimal short text generation | Connect pinned tokenizer and greedy sampling; validate a bounded generated sequence and required original weight coverage | Complete; user confirmed |
| 6 | Short text generation | Add indexer/shared KV ownership, remaining layers and Engram paths; validate logits and generation with real weights | Bounded short-generation control complete; user confirmed |
| 7a | Common consumer startup memory budget | Account for the fixed profile before loading weights; reject insufficient limits; validate original generation with capacity restricted to 20 GiB | Complete; user confirmed |
| 7b | Bounded CPU expert cache and SSD reads | Account for resident/staging bytes; verify cold/warm/eviction behavior with original weights and preserve the common budget | Complete; user confirmed |
| 7c | Common expert file index/source | Unify GGUF/raw expert ranges and original identities; retain bounded reads, cache and admission; preserve original generation | Complete; user confirmed |
| 7d | Unified model manifest and independent entry | Load dense, Engram and expert sources with explicit revision/coverage; preserve original generation without test-code dependencies | Complete; user confirmed |
| 7e | Bounded IQ1_M sample and upstream audit | Verify real mixed-format expert/Engram samples on SM75, bounded downloads and direct SSD reads | Complete; user confirmed |
| 7f | Upstream engine SM75 build and intake | Reuse pinned strata-dsv41 loader/runtime; check hardware and package requirements before complete weights | Complete; user confirmed |
| 7g | Complete IQ1_M base and first upstream text run | Verify five full files, native pack coverage and one short generation with memory observations; DSpark none | Complete; user confirmed |
| 7h | Real-prompt usability baseline | Five prompts, up to 64 output tokens each; measure first-token latency, speed, memory and SSD/cache behavior | Complete; user confirmed |
| 7i | Native SSD/cache diagnostics | Expose existing counters, validate current-parent/fixed-path output controls and 20 GiB capacity; document dynamic-path variation | Complete on CUDA; user confirmed |
| 7j | RAM-store budget comparison | Compare native 32/40 GB ceilings; measure answers, memory headroom, misses, evictions, reads and waits under the same GPU bound | Complete; user confirmed |
| 7k | Bounded longer-generation checks | Keep the explicit 40 GB candidate and 20 GiB GPU bound; check longer inputs and 128–256-token output limits, answers and memory | Complete; user confirmed |
| 7l | Bounded local text entry | Reuse the tokenizer and native serve process; accept Chinese input with pinned assets, memory/output limits, streaming and cleanup | Complete; user confirmed |
| 7m | Resident independent-question entry | Load once and reuse weights/expert caches across bounded independent questions; retain admission, streaming and cleanup | Complete; user confirmed |
| 7n | Native RAM-store policy comparison | Compare LRU/LFU with fixed prompts and unchanged RAM/GPU bounds; measure answers, reads, waits and latency | Complete; user confirmed, retain LRU |
| 7o | Native chunked-prefill audit and bounded trial | Check SM75 compatibility and workspace before trying a small chunk under the same memory limits; record latency and answers | Complete; user confirmed, retain windows |
| 7p | Native post-prompt RAM-preload isolation | Compare the existing preload switch under the same bounds; review TTFT, decode, reads and answers together | Complete; user confirmed, tradeoff and format failures retained |
| 7q | Bounded reuse of streamed hot experts | Reuse existing cache infrastructure; verify ownership, memory bounds and actual read savings before promotion | Complete; user confirmed, default-off prototype |
| 7r | Resident four-request cache-pressure comparison | Separate first/later/repeated-query effects; review capacity, reads, memory, output quality and both run orders | Complete; user confirmed |
| 7s | Production window versus flow-store resident comparison | Same-day pinned paths, both orders, fixed requests, quality, resources and cache pressure | Complete; user confirmed |
| 7t | Native first-flow/later-window experiment | Successful-feed transition, original paths and workspace, default-off compatibility, same-day resident comparison | Complete; user confirmed |
| 7u | First-prompt length and request order | Same short/long task multiset, ABBA per order, native path logs, quality and resources | Complete; user confirmed |
| 7v | Native chooser on first requests | Fresh-process short/long probes, both orders, predicted versus observed costs, quality and resources | Complete; user confirmed |
| 7w | First-short-feed window guard | Default-off compatibility, equal-workspace 14/16/17/257 matrix, success-only state and later-feed controls | Complete; user confirmed |
| 7x | Offline prefill cost and cache-state audit | Saved-trace IO/timing attribution, measurement semantics, source review and bounded next experiment | Complete; user confirmed |
| 7y | Positive native cache-settle budget | Same binary, 1000/6000 ms, two first-request fixtures and both orders; full latency, decode, reads, quality and resources | Complete; user confirmed |
| 7z | Resident positive cache-settle budget | Same binary and four-request short/long/repeat sessions, both orders; first/later latency, decode, reads, cache pressure and quality | Complete; user confirmed |
| 7aa | Production windows versus 1000 ms flow | Pinned production/candidate profiles, same four-request tasks and both orders; full latency, decode, reads, quality and resources | Complete; user confirmed |
| 7ab | Opt-in native flow text entry | Explicit single/resident profile, shared guards/streaming/cleanup, default compatibility and four-process/six-request smoke | Complete; user confirmed |
| 7ac | Longer explanation and code quality | Original99/121 tasks, cap256, four resident controls/eight answers; content/format/completion, function vectors and cache pressure | Complete; user confirmed |
| 7ad | Offline decode fetch and cache attribution | Saved8-answer waits/reads/cache seed, native counter semantics, same-output control and one bounded next hypothesis; no new model run | Complete; user confirmed |
| 7ae | Opt-in flow retention heat4 versus1 | Same binary, exact99/121 tasks in both orders,4 processes/8 answers; first/later latency, reads/waits, quality and bounds | Complete; user confirmed |
| 7af | RAM preload on/off at heat1 | Same binary/GPU settle1000,4 processes/8 started requests;7 complete/1 resource interrupted; no rerun or default promotion | Resource gate failed; diagnostic follow-up user-confirmed |
| 7ag | Failure-only resource metadata and one replay | Bounded read-only helper,45 tests/130 controls, same binary/on/heat1 one process/two answers; old failure immutable | Complete; not reproduced/root unresolved; follow-up user-confirmed |
| 7ah | Four-request resident resource diagnostic | Same binary/settings, original pair repeated once,48 tests/131 controls; one process/four EOS answers, failed7af retained | Complete; not reproduced/root unresolved; follow-up user-confirmed |
| 7ai | Offline host-memory ownership/budget audit | Known/unknown capacity ledger, six saved sessions/26 STORE snapshots,31 offline checks; no new model run | Complete; root unresolved; follow-up user-confirmed |
| 7aj |40/36GB resident cache-budget comparison | Same native/heat1/on/GPU1000,2 models/8 EOS answers,51 tests/136 controls; memory/latency/reads/quality and order limits retained | Complete; follow-up user-confirmed; no default promotion |
| 7ak | Reverse-order36/40GB cache-budget check | Same native/configuration,2 models/8 EOS answers,5 new controls; quality and order sensitivity reviewed | Complete; follow-up user-confirmed; no default promotion |
| 7al | Opt-in36GB local text profile |57 tests/148 no-GPU controls,2 public-entry models/3 EOS answers; input recovery,quality,resource bounds and cleanup | Complete; follow-up user-confirmed; defaults retained |
| 7am | Bounded eight-question resident stability/quality |18 regressions/7 controls,1 model/8 EOS answers,32 function checks;resource pass/content6of8/format5of8 | Complete; follow-up user-confirmed;quality failures retained |
| 7an | Offline context/memory-budget audit |45 checks,22 pinned inputs,shared-cache/scratch/LM/YOCO and entry-boundary review;0 native runs | Complete; awaiting confirmation;2048 candidate unmeasured |
| Full import | Complete quantized package and coverage | Verify complete source hashes, original coverage and memory before generation | IQ1_M/Engram base complete in 7g; first text smoke passes |
| 7 | Consumer GPU optimization | Measure prefill overlap, cache policy, graph capture and DSpark one change at a time | Proposed |

Steps through **7ae** are user-confirmed. **7af** retains its zero-swap
resource failure; the user authorized diagnostic follow-up. **7ag** is
user-confirmed without reproducing swap. **7ah** is user-confirmed with four
EOS answers and no observed swap. **7ai** completes the offline host-memory
ledger and is user-confirmed. **7aj** completes the bounded40/36GB comparison
and is user-confirmed. **7ak** completes the reverse-order check and is
user-confirmed. **7al** exposes the opt-in36GB profile and is user-confirmed.
**7am** completes the bounded8-answer run and is user-confirmed with retained
quality failures. **7an** completes the offline context audit and awaits review;
2048 remains a proposed candidate and the original7af root cause is unresolved.
The native prefix covers layers **0–39**, with bounded generation and explicit
model loading. The complete IQ1_M/Engram base passes bounded checks under a
20 GiB GPU target. Gate 7n retains LRU, and 7o retains windows. Gates7q–7u
measure opt-in handoff and mixed-path behavior, including short-first costs.
Gate 7v identifies the original chooser's short/long first-request tradeoff.
Gate 7w tests a default-off, one-use short-feed window guard in the same chunk
workspace and does not support enabling it. Gate 7x attributes the saved
prefill variation to stream waits and cache settling, with explicit limits
on the cache estimates. Gate 7y measures the existing positive settle budget
on fresh short/long requests. Gate 7z extends that comparison to resident
first/later/repeated requests. Gate 7aa compares the resulting candidate with
the current production windows profile on the same tasks and resource bounds.
Gate 7ab exposes the measured candidate as an opt-in local text profile.
Gate 7ac records longer-generation code success, explanation-format/coverage
failures and the limits of extending the earlier timing result. Gate 7ad
attributes the saved decode gap to miss volume/ensure-tail waits and proposes
a single opt-in free-slot retention threshold experiment, without claiming
a measured gain or changing runtime defaults. Gate 7ae implements that opt-in
threshold experiment: the fixed long-output request totals improve, mostly
on first-request decode, with later-request and output-quality limits.
Heat1 remains benchmark-only through7ak;7al makes36GB/heat1 an explicit text
choice. Gate7af tests the existing RAM-preload switch;
its final answer is resource-interrupted, so no full comparison or promotion
follows. Gate7ag adds bounded failure-only diagnostics and performs one
independent replay; swap does not recur in that attempt, so the original
resource issue stays open. Gate7ah extends the diagnostic to four fresh-KV
requests in one model without reproducing swap;7af stays failed. An offline
host-memory ownership/budget audit in7ai identifies capacity outside the
expert cache and separates it from old residency. A bounded36/40GB budget
comparison in7aj measures the observed headroom/cache-cost tradeoff. Gate7ak
checks its order sensitivity separately, preserving output/date/cache limits.
Gate7al adds the explicit memory-experimental text profile and checks its public
inputs,streaming,resident recovery and cleanup. Gate7am extends it to one
bounded8-question resident session;resource bounds pass while content/format
failures remain. Gate7an audits context-dependent allocations and proposes
a bounded2048 trial. Further work awaits confirmation;no default,safe-cap,
larger-context validation or root-cause claim follows.
Production windows remains the default.

Longer-context quality, official GPU parity and sustained throughput still
require later validation. The short answers, three longer tasks and existing
component controls do not establish those results.
