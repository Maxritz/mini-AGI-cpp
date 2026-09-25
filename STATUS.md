# mini-AGI-cpp STATUS

Live audit record per the COMPLETE-IT contract (§5, §30).
Authority order: tests > golden/reference > Python > C++ contracts > docs.
Baseline: clean MinGW build, `ctest`: 8/8 PASS (mininpz, tensor,
config_tokenizer, store, model, pool_backward, backward, wavef). DX12 off in
that build by design.

## Dependency graph (current state)

```
[Tensor] COMPLETE
    ↓
[Store] COMPLETE
    ↓
[Paging] COMPLETE
    ↓
[CPU Forward inc. MoE + golden] COMPLETE
    ↓
[CPU Backward] COMPLETE for weights (attn/RoPE/RMSNorm/SwiGLU/pool verified
               by numerical gradient check; loss KL + aux still open)  ← CURRENT
    ↓
[Optimiser] PARTIAL (BLOCKED on resume bug + no trunk/pool split)
    ↓
[DX12 infra] PARTIAL (BLOCKED: root-signature mismatch)
    ↓
[DX12 dense/MoE/backward] BROKEN (BLOCKED on infra)
    ↓
[Training loop] PARTIAL (experts+gate now learn; no resume/eval)
    ↓
[Continual learning] MISSING in C++ (Python-only)
    ↓
[End-to-end] PARTIAL (train→save works; reload/resume missing)
```

## Subsystem verdicts

| Subsystem | Verdict | Evidence |
|---|---|---|
| Tensor | COMPLETE | `test_tensor` passes: shape/dtype/ownership/matmul/rmsnorm/softmax/RoPE/topk. Single `mt::Tensor` abstraction, no duplicate. `is_view_` exists; views exercised via slices in model fwd. |
| Configuration | COMPLETE | `test_config_tokenizer` passes; `Config::from_manifest` typed round-trip verified. |
| Tokenizer | COMPLETE | Byte-level 265 + 9 specials matches `minagi/tokenizer.py`; encode/decode round-trip in tests. |
| Model creation | COMPLETE | `tools/create.cpp` + `model_create.*` build fresh dirs; golden weights load. |
| Model forward (trunk) | COMPLETE | `src/model.cpp`: RMSNorm→fused qkv→RoPE→causal mask (P=kv_len−T)→proj, SwiGLU `w2(silu(w1)*w3)`, residuals, tied head — matches `minagi/model.py`. |
| Attention | COMPLETE (CPU) | Cache concat + writeback matches `Attention` (`dict{k,v}` on dim 2); mask convention matches (`ki<=qi+P`). Validated via golden. |
| RoPE | COMPLETE (CPU) | `build_rope`/`apply_rope` match `inv=1/(theta^(arange(0,hd,2)/hd))`, even/odd rotation. CPU backward added: the inverse is the rotation by −theta, verified by `test_backward`. |
| RMSNorm | COMPLETE fwd / COMPLETE bwd | Fwd matches (fp32 reduction, eps 1e-6, no bias). Bwd now includes the normalisation Jacobian (`dx = r·du − r³·(x/D)·Σdu·x`); omitting it left every gradient flowing *through* a norm wrong. Verified in `test_backward`. |
| SwiGLU | COMPLETE fwd / COMPLETE bwd | Fwd matches. Bwd uses `sig*(1+x*(1-sig))` via `silu_deriv1` (`pool.cpp:43`); the old `sa1+(1-sa1)*a1` form is gone. Dense + pooled paths both verified in `test_backward` / `test_pool_backward`. |
| Router | COMPLETE | Fwd (softmax→topk→renorm→gate-scale→capacity dispatch→combine) validated vs golden. Bwd carries the full softmax Jacobian over all resident slots, not just the top-k, plus the renormalisation Jacobian and the gate path. Verified in `test_pool_backward` (router max rel err 0). |
| Expert pool | COMPLETE fwd / COMPLETE bwd (resident experts + gate learn; non-resident experts correctly get no grads until swapped in) |
| Expert dispatch | COMPLETE | Switch-style sort-by-expert, `cap`/capacity drop counting matches `pool.py PooledMLP`; forced-drop behaviour in `test_wavef` check4. |
| Paging | COMPLETE | Tiers LRU/fetch/flush/counters, `swap_to` identity-matched rearrange, `choose`/`choose_by_demand`, `build_keys`, read_only discipline — all validated in `test_wavef` checks 2/5/8/9/10. Never silently all-resident (resident=3 < 6 in golden). |
| Recurrence | COMPLETE fwd / COMPLETE bwd (weights) | Fwd: adapter `[I|I]`, recur/coda sites, ponder loop match `recur.py`. Bwd: per-(step,block) attention, RoPE, ln1/ln2, dense + pooled MLP all implemented and numerically verified. The adapter is applied to `pre_out` at every step, so the prelude gradient is a sum over steps — routing only step 0 was a real bug (see gap history). |
| Halting | COMPLETE fwd / COMPLETE bwd | `lam=sigmoid(halt)` forced 1/0 on last/pre-min steps; golden halt rows exact. Bwd: the `d_lam` recursion was implemented but missing the batch mean, so every halting gradient was Mx too large (an 8x overestimate at production M=8); fixed in `step_ce` by folding in `1/M`, and `halt.weight`/`halt.bias` are now verified by a cached-lam batch sweep (`test_backward` case 2b) that fails M2/M3 by 2× with the defect and passes with the fix. The `d_lam -> d_yf` edge into `ln_f` is implemented and covered transitively via the trunk checks. |

| Decoder | COMPLETE | `pick_next` (adapt trace, rep penalty, n-gram ban, temp<=0 argmax) matches `decode.py`; golden gen ids exact (32/32). |
| Loss | PARTIAL | Halt-weighted CE mixture real, and the reported loss is now the batch mean (it previously summed without the `/M` its own gradient applied, reporting M× the objective being differentiated). MISSING: ponder KL (`ponder_beta`), pool aux/z-loss. |
| Backward | COMPLETE for weights | `test_backward` checks every trunk + recurrent weight against a directional finite difference of the loss, in three cases: dense trunk, free-halting trunk, and the pooled recurrent site (router/depth_emb/gate/expert grads all through `backward()`). Three real bugs were found and fixed by it (see Completed). NOT verified: the halting gradient (gap #11) - too small to measure in float32 at this scale. |
| Optimiser | PARTIAL | Real AdamW (clip, decoupled wd, bias correction, m/v save/reload). Bugs: `clip_gradients()` runs a full step + pins clip (`optimizer.cpp:118-128`); single-tensor `step()` is a silent no-op (`:16-27`); `load_state_dict` restores `|t` only, resume restarts bias correction vs stale moments (`:144-211`); `train.cpp` runs wd=0.0 under the AdamW name. |
| Checkpointing | COMPLETE (write+read) | Atomic `.tmp`→rename, core/routers/experts/manifest/telemetry round-trip in `test_wavef` check7; ZIP64 streaming handles >4 GiB. |
| Store | COMPLETE | `store::load` (core+routers+experts+legacy flat `.npy`, moment skip, bad-dir failure) validated in checks 1/11; `save_paged` router-bundle routing fixed this session. |
| DX12 context | PARTIAL | Device/queue/allocator/list/fence creation works (smoke §1). No adapter selection (RX 9070 XT not picked/logged); descriptor heap allocated never used; (near-)zero transitions/barriers. |
| DX12 engine | PARTIAL | PSO compile/dispatch/upload/readback plumbing exists; pushes constants to correct root param (12). No precompile/cache; `set_pso` before root signature (reversed). |
| Dense GPU kernels | BROKEN | Root-signature 3-way mismatch (docs say 2 params, impl has 13 UAV + constants@12, `dx12_dense.cpp:57` pushes @1); SRV binds on UAV-only params; `GpuBuffer` vs `Dx12Buffer` split with no converter — dense cannot consume engine buffers. Kernel-level: softmax reduction broken for cols>16/rows>1; RoPE ignores pos_offset/stride3; embedding/copy slot-0 double-bind + in-place alias; attn/MLP composite miswiring (K source, residual binds, scores register u9 vs SRV 9). |
| MoE GPU kernels | BROKEN | `pool.hlsl:104` assignment stride `*4` should be `*16`; kept-compaction mismatch (`dx12_pool.cpp:421` vs `:430-442` vs shader `total_kept` loops); missing inter-pass barriers. |
| Backward GPU kernels | STUB (orphaned) | 8 `main_grad_*` kernels, zero C++ callers; `CSGradScatterAdd` references undeclared `total_kept`/`D`; `CSGradAttention` assumes H=1; W/X grad registers collide (`backward.hlsl:206` vs `:228`). |
| CPU/GPU equivalence | MISSING | Smoke test does CPU-golden + synthetic pool-shape only; no per-op CPU-vs-DX12 numerics exist. |
| Training loop | PARTIAL | `tools/train.cpp`: toy byte-stream → forward → CE → trunk+router+expert+gate step → checkpoint + timer growth/prune. Expert filename fixed to `e%05d.npz` so reload finds them; residents flushed before checkpoint. No eval, no resume, no config.yaml, hardcoded dims. |
| Continual learning | MISSING (C++) | Streaming/interleave/replay/Plasticity/AutoGrow-brakes/revert/forgetting-guards exist only in Python (`train.py`/`stream.py`/`plasticity.py`/`live.py`). C++ has mechanics (`grow`/`prune`) with fixed-step policy. |
| Serving/inference | COMPLETE (infer) | `minagi_decode` loads + greedy-decodes; GGUF converter path smoke-verified (bit-exact dequant, finite peaked logits). No learning serve path in C++. |
| Tests | FULL | 8/8 pass. `test_backward` covers dense trunk (case 1), free-halting trunk (case 2), and pooled recurrent site (case 3), plus case 2b: a cached-lam batch sweep that verifies `halt.weight`/`halt.bias` by differentiating the loss w.r.t. the cached `lam_n` in double precision — the only method that resolves the halting signal in float32. `test_pool_backward` covers pooled MLP in isolation; case 3 asserts that a pre-`begin_segment()` write throws (gap #12 closed). Remaining gaps: no optimiser resume test, no DX12 numerics, no train→reload→resume. |

## Backward gap table

CLOSED this session — each was found by `test_backward`, not by inspection:

| # | Gap | Status |
|---|---|---|
| 1 | Recur/prelude attn grads were zeros | **CLOSED** — per-(step,block) attention backward (`attn_backward`), incl. softmax Jacobian, RoPE inverse, cached-key handling |
| 2 | Expert + gate grads skipped; `d_pool_x` discarded | **CLOSED** — expert SwiGLU + gate grads; `d_pool_x` folded through the ln2 norm into the block input |
| 3 | Wrong SiLU-derivative form in pool backward | **CLOSED** — `silu_deriv1` (`pool.cpp:43`), pinned by `test_pool_backward` |
| 4 | Router grad simplified; `d_depth_emb` zeros | **CLOSED** — full softmax Jacobian over all resident slots + renorm Jacobian + gate path |
| 5 | Adapter step-0 grad never accrued | **CLOSED, but the real bug was wider** — see below |
| 6 | RMSNorm Jacobian ignored; `d_xn2` dropped | **CLOSED** — full Jacobian added; this alone fixed every gradient flowing *through* a norm |
| 9 | Dense-fallback MLP (`use_pool=false`) had no backward branch | **CLOSED** — `dense_mlp_backward` shared by the prelude and non-pooled recur sites |

STILL OPEN:

| # | Gap | Location | Fix direction |
|---|---|---|---|
| 7 | Ponder KL + pool aux loss absent | `backward.cpp` (no refs) | Add KL(`ponder_beta`, `halt_prior`) + aux terms to match `recur.py:338-343`, `pool.py:435-436` |
| 8 | BPTT window / Poisson depth sampling absent | `backward.cpp`, `train.cpp` | `bptt_window` detach + `train_steps_mean` sampling per `recur.py:229-233,271-273` |
| 10 | GPU backward orphaned + uncompilable kernel | `backward.hlsl:424-425` etc. | Fix `total_kept`/`D` bindings; wire first `grad_*` PSO only after CPU backward is validated |
| 11 | ~~Halting gradient unverified~~ | `test_backward.cpp` case 2b + `backward.cpp` | CLOSED. The halting gradient CAN be checked; the float32-floor belief was wrong. The real defect was a missing `1/M` on `d_lam` (and `dc_prev`), making every halting gradient M× too large. Verified by a batch sweep: analytic/numeric ratio was exactly 1/M (1.0/0.5/0.333 at M=1/2/3) — a signature no float32 noise can fake. Fix: fold `inv_M` into `step_ce` so both `d_lam` and `dc_prev` inherit it. Regression test: `check_halting_scaling` perturbs the *cached* lam and re-evaluates the loss in double (immune to the float32 forward floor), sweeping M in {1,2,3}; the defect fails M2/M3 by 2×, the fix passes both. `halt.weight`/`halt.bias` perturb-as-parameter fails because the loss moves by one ulp, but the cached-lam probe resolves it to <8% rel err. NOTE: an earlier M=1-only probe masked this — always test the batch sweep. |
| 12 | ~~Pooled path not checked end-to-end~~ | `test_backward.cpp` case 3 | **CLOSED** - router, depth_emb, gate and per-expert grads all checked through `backward()` with `use_pool=true`. Note a fresh `PagedPool` has `slots_=[-1]*resident` and every slot maps back to expert 0, so the case needs `coder.begin_segment()` first or routing collapses and every gradient is zero. |

### The three bugs the gradient check found

Worth recording because none were visible by inspection, and two had been written and "verified" by a passing test suite:

1. **Prelude gradient was routed from step 0 only.** The adapter reads `pre_out` at *every* step, but the gradient into `pre_out` was accumulated inside `if (n == 0)`. Measured error: `tok_emb` off by 1.33×, prelude `proj` by 1.85×. Fix: accumulate across steps, run the prelude backward once after the time loop.
2. **RMSNorm backward omitted the normalisation Jacobian.** The weight grads were right, which is why nothing looked wrong; every gradient flowing *through* a norm was wrong.
3. **The loss summed without the `/M` its own gradient applied**, so the reported number was M× the objective being differentiated. Visible as a constant factor in every parameter.

Plus one forward bug, found because the gradient check made the model trainable:
4. **`random_init` zeroed `attn.proj`**, which the Python reference initialises `normal_(0, 0.02/sqrt(2*depth))`. A zeroed `proj` zeroes the attention output, so the whole recurrence was a no-op on its first pass. Training loss on a 3-step smoke run went from 22.1 to 5.55 after the fix.

### A test-side bug that masked a real defect

5. **`ce_of` in the halting-probe used `std::exp` where `std::log` belonged**, producing `ce ≈ 0.919` instead of the true `1.922` — exactly a ~2.09× factor, which silently canceled against the missing-batch-mean error and made a defect-reintroduction look green for a full turn. A correct CE is `-(log(exp(l[tgt]-mx)) - log(Σexp))` = `-(log p_tgt)`; the test had the first `log` as an `exp`, turning `-log(0.146)` into `-(0.925 - log(6.32)) = 0.919` instead of `-log(0.146) = 1.922`. Every agent in the `/debate-3` round named indexing/layout as the cause; none read line 213. **Lesson: when two near-identical reconstructions agree to 6 digits, they may share one typo rather than both be right.** Caught because the standalone probe was rebuilt to print component values, then the test's `ce_of` was compared against it. Removed by deleting `tests/eps_probe.cpp`; the corrected `ce_of` lives on in `test_backward.cpp` case 2b.

### A silent-zero class of bug worth guarding

5. **A fresh `PagedPool` has `slots_ = [-1]*resident`, and `resident_rows()` maps every `-1` to expert 0** (`src/paged.cpp:363`). Writing the resident tensors without first calling `Coder::begin_segment()` therefore produces a forward in which all traffic routes to one expert and every other expert's gradient is exactly zero — no crash, no NaN, just a wrong answer. `test_backward`'s pooled case hit this and reported all-zero grads; the fix was one `begin_segment()` call, but the real defect was that the write path permitted it. `PagedPool::mutable_w1/w2/w3` now throw `std::runtime_error` when `slots_valid()` is false, with `slots_valid()` exposed for callers that want to branch. `train.cpp` and the test both go through the guard; the 3-step smoke run still passes.

**Debugging note (float32 has a floor).** The halting gradient cannot be checked at this model scale and no amount of tuning fixes it: `dL/d(lam)` is ~5e-6 against a loss of ~1.9, i.e. ~3e-8 relative, while float32 holds ~1e-7. A central difference returns ~1e-6 for *any* step size, and the apparent "ratios" (0.35, 0.66, 1.33) swing non-monotonically with step count — that is the noise floor, not a signal. Two plausible-looking fixes to the `d_lam` recursion were tried and both were **refuted** by an isolated toy check (adding a `cum_prev` factor to the `CE·lam` term: ratio 1.59/1.57/1.79; dividing by `M`: same). The un-modified recursion matches a central difference to ratio 1.000000 on every step. **Lesson: when a gradient check disagrees, first confirm the instrument can resolve the signal before changing the math.** Closing gap #11 needs a float64 loss or a much larger model, not another edit to `backward.cpp`.

### Throwaway diagnostics

There is no CMake target for ad-hoc probes; compile by hand against the static lib:

```sh
g++ -std=c++17 -O2 -I src tests/<probe>.cpp build/libminimagi.a -o build/<probe>.exe
```

Add a real target instead if these start recurring.

### Current TODO ledger

```text
DONE P0 src/backward.cpp step_ce: batch mean (/M) added to CE so d_lam and dc_prev are scaled; halting grads no longer M x too large. Verified by batch sweep (ratio 1.000000 at every M).
DONE P0 src/backward.cpp:1040-1080 comment rewrite to warn against M=1-only reference checks.
DONE P0 tests/test_backward.cpp case 2b: check_halting_scaling - cached-lam batch sweep over M={1,2,3}, fails by 2x with the defect, passes with the fix (rel<8%).
DONE P0 tests/test_backward.cpp: ce_of exp-vs-log typo fixed (line 213).
DONE P0 tests/test_backward.cpp case 3: begin_segment() precondition + pool_slots_valid assertion; pre-segment writes now throw.
DONE P0 src/paged.hpp: mutable_w1/w2/w3 now assert_slots_valid() and throw std::runtime_error.
DONE P1 src/paged.cpp: assert_slots_valid + slots_valid() defined.
DONE P1 src/recur.hpp: begin_segment() comment upgraded to REQUIRED.
DONE P1 STATUS.md: gap #11 rewritten, not deleted - records the float32-floor-vs-real-defect confusion so the next reader doesn't repeat it.
DONE P1 STATUS.md: added ce_of exp/log typo under bug history #5.
P1   src/backward.hpp: d_lam_per_step field - could add an invariant: sum_t d_lam[n][t] == dL/d(sum_t lam_n[t]); cheap, float64-free, catches future normalization slips.
P2   CMakeLists.txt: add a guarded minagi_diag target for ad-hoc probes instead of hand-compiling.
```


## Baseline record (2026-09-25)

```text
BUILD:        PASS (MinGW Ninja, warnings only)
CPU TESTS:    PASS 8/8 (mininpz, tensor, config_tokenizer, store, model,
                       pool_backward, backward, wavef)
DX12 TESTS:   NOT RUN (MinGW build is DX12-off by design; needs fresh MSVC build)
GOLDEN TESTS: PASS (test_wavef check6: logits 1e-4, halt rows exact, gen ids 32/32, summaries 2e-4)
GRADIENT TEST:PASS (test_backward: every trunk + recurrent weight matches a
                   directional finite difference within float32 noise)
POOL GRADIENT:PASS (test_pool_backward: dx/router/depth/gate/ew1/ew3/ew2)
TRAIN SMOKE:  PASS (minagi_train 3 steps: loss 5.55, checkpoint written)
```

## Completed this session (branch `gguf-convert-smoke`, pushed to origin)

- `tools/gguf_convert.cpp` (new): GGUF v2/v3, Q4_K/Q6_K dequant, GQA fusion, typed cfg, 32-thread, sanity scan.
- `src/mininpz.cpp`: unconditional-ZIP64 streaming writer + dual-format seek reader.
- `src/recur.cpp`: dense recur-site branch on `use_pool`.
- `src/store.*`: `save_paged` router-bundle parameter.
- Verified: output_norm exact, dequant bit-exact vs independent Python, decode finite peaked logits.
