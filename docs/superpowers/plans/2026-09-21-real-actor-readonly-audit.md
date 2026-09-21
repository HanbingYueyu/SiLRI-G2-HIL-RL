# G2 Real Actor Read-Only Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the complete dual-RGB SiLRI Actor inference path on the RTX 3090 without constructing or invoking any robot command path, then approve six explicit freshness limits from three qualifying live sessions.

**Architecture:** A standalone `ActorInferenceProbe` owns trusted-local checkpoint validation, deterministic full-frame resize, CUDA inference, synchronization, action validation, and timing. The existing read-only freshness audit accepts this probe as an optional callable and evaluates source ages only after real inference completes. A separate offline approval tool consumes three immutable summaries and evidence streams and emits an explicit, provenance-bearing limits artifact; it never enables motion or becomes a default configuration.

**Tech Stack:** Python 3.10, PyTorch/CUDA, NumPy, GDK read-only APIs, existing PTP snapshot IPC, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-real-actor-readonly-audit-design.md`

## Global Constraints

- Real hardware remains `allow_motion=False` throughout every task and live run.
- Never construct or import `GdkCommandPort`, `MotionBackend`, a Gym environment, or a learner client from the probe/audit/approval modules.
- Never send hold, motion, gripper, mode-switch, or Gym-step commands.
- Never call `phc2sys`, change CLOCK_REALTIME/PHC, or let Actor/audit control PTP.
- The only accepted live inference device is CUDA with a device name containing exactly `NVIDIA GeForce RTX 3090`.
- Input keys and shapes are exactly state `(7,)`, `left_wrist`/`right_aux` HWC uint8 RGB, and Actor input images `(1,3,128,128)`.
- Full-frame bilinear resize is a timing-audit choice, not approval of the final task ROI or visual policy quality.
- Every action is validated and discarded; no action value crosses an environment or command boundary.
- Every summary retains `motion_authorized=false`, `thresholds_approved=false`, `source_clock_identity_proven=false`, and `actions_discarded=true` until the separate approval artifact is produced; the approval artifact still retains `motion_authorized=false`.
- No production threshold defaults are added; callers must explicitly load an approved artifact.
- A threshold proposal requires three independent 120-second sessions with at least 1000 accepted samples each and matching checkpoint SHA-256, policy configuration, and GPU identity.
- Preserve the user's three already-staged Chinese files and unrelated submodule/worktree state.

---

### Task 1: Trusted CUDA Actor Inference Probe

**Files:**
- Create: `g2_local/actor_inference.py`
- Create: `tests/test_g2_actor_inference.py`
- Modify: `g2_local/policy.py`

**Interfaces:**
- Consumes: trusted-local checkpoint path, `create_policy(device='cuda')`, and an observation mapping with `state`, `left_wrist`, and `right_aux`.
- Produces: `ActorInferenceProbe(checkpoint: Path, *, device: str, warmup_steps: int)`, immutable `InferenceResult`, `warmup(obs)`, `infer(obs) -> InferenceResult`, `metadata() -> dict`, and idempotent `close()`.

- [ ] **Step 1: Write failing checkpoint and device tests**

```python
def test_probe_rejects_wrong_schema_config_gpu_and_actor_keys(tmp_path, fake_cuda):
    for fault in ('schema', 'camera', 'state', 'action', 'gpu', 'actor_keys'):
        with pytest.raises(ValueError, match=fault):
            make_probe(tmp_path, fake_cuda, fault=fault)

def test_checkpoint_hash_and_version_are_immutable_metadata(probe):
    meta = probe.metadata()
    assert meta['checkpoint_sha256'] == sha256(probe.checkpoint)
    assert meta['checkpoint_version'] == 3
    assert meta['gpu_name'] == 'NVIDIA GeForce RTX 3090'
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_actor_inference.py -q`

Expected: FAIL because `g2_local.actor_inference` does not exist.

- [ ] **Step 3: Implement strict trusted-local construction**

Use exact type checks, SHA-256 streaming reads, `torch.load(..., map_location='cpu', weights_only=False)` only after documenting the trusted-local boundary, exact config validation, strict extraction of `actor.*` keys, `create_policy('cuda').eval()`, and exact `load_state_dict`. Reject CPU, non-CUDA devices, unexpected GPU names, missing/extra Actor keys, and non-finite parameters. Do not import runtime Actor, Gym, command, learner, or motion modules.

- [ ] **Step 4: Write failing preprocessing/inference/timing tests**

```python
def test_full_rgb_pipeline_resizes_runs_synchronizes_and_discards_action(probe, observation):
    result = probe.infer(observation)
    assert probe.policy.last_shapes == {'left_wrist': (1,3,128,128),
                                        'right_aux': (1,3,128,128), 'state': (1,7)}
    assert probe.cuda.synchronize_calls >= 2
    assert result.action_shape == (1, 6)
    assert result.action_discarded is True
    assert result.total_ns >= result.forward_ns > 0
```

Also cover wrong NumPy dtype/shape/contiguity, NaN state/action, output outside the SiLRI clamp, CUDA exceptions, close after partial construction, and `BaseException` cleanup.

- [ ] **Step 5: Implement deterministic inference**

Convert HWC uint8 images to NCHW float `[0,1]`, transfer the full frame, resize with explicit bilinear arguments to 128x128, run under `torch.inference_mode()`, synchronize around measured CUDA work, validate exact output, copy only the six diagnostic values to CPU, then discard them. Record CPU preparation, H2D/resize, forward/synchronize, total, and CUDA memory bytes as exact nonnegative integers.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_actor_inference.py tests/test_silri_runtime.py -q
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests -q
git diff --check
git -C lerobot diff --check
```

Commit only the three Task 1 files as `feat: add read-only CUDA actor inference probe`.

---

### Task 2: Integrate Real Inference into Freshness Audit

**Files:**
- Modify: `g2_local/freshness_audit.py`
- Modify: `tests/test_g2_freshness_audit.py`
- Create: `tests/test_g2_actor_audit.py`

**Interfaces:**
- Consumes: `ActorInferenceProbe`, `GdkReader`, and `SnapshotClient`.
- Produces: CLI options `--actor-checkpoint PATH --device cuda --warmup-steps N`; real inference timing distributions and fixed `actions_discarded=true`. `--inference-delay-s` remains diagnostic-only and is mutually exclusive with Actor mode.

- [ ] **Step 1: Write failing ordering and exclusion tests**

```python
def test_snapshot_and_age_are_measured_after_real_inference(audit_rig):
    report = audit_rig.run(actor_probe=OrderedProbe())
    assert audit_rig.events == ['observe', 'infer_start', 'infer_end', 'snapshot', 'measure']
    assert report['actions_discarded'] is True

def test_actor_mode_never_builds_motion_gym_or_learner():
    imports = module_imports('g2_local.freshness_audit', 'g2_local.actor_inference')
    assert not imports & {'GdkCommandPort', 'MotionBackend', 'G2LocalEnv',
                          'LearnerServiceStub'}
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_actor_audit.py tests/test_g2_freshness_audit.py -q`

Expected: FAIL because the audit has no Actor mode.

- [ ] **Step 3: Implement dependency-injected Actor mode**

Validate all CLI arguments and output existence before checkpoint, CUDA, GDK, or socket construction. Require Actor mode for the new现场 workflow; make it mutually exclusive with nonzero simulated delay. Warm one complete observation for exactly `warmup_steps`, record warm-up separately, then acquire a fresh observation for every formal sample. Call the probe before reading the snapshot so all ages include inference. Preserve transactional evidence and existing close/failure semantics.

- [ ] **Step 4: Extend evidence and summary**

Add literal-unit distributions for `cpu_prepare_ms`, `h2d_resize_ms`, `actor_forward_ms`, `actor_inference_ms`, and CUDA memory MiB. Include checkpoint/GPU/config metadata, requested/completed warm-ups, exact accepted/rejected counts, and action min/max aggregates without retaining an executable action channel. All JSON uses finite values and existing byte/row bounds.

- [ ] **Step 5: Verify and commit**

Run focused audit/probe/freshness tests, then all tests and both diff checks. Commit only the three Task 2 files as `feat: audit real actor inference without motion`.

---

### Task 3: Three-Session Evidence Validator and Explicit Limit Proposal

**Files:**
- Create: `g2_local/freshness_approval.py`
- Create: `tests/test_g2_freshness_approval.py`
- Create: `configs/g2_freshness_limits.schema.json`

**Interfaces:**
- Consumes: exactly three audit evidence directories and six explicit proposed limits.
- Produces: `validate_sessions(paths) -> ApprovalEvidence`, `approve_limits(evidence, limits, output)`, and an immutable JSON artifact containing the six values and provenance. It never supplies defaults or changes motion state.

- [ ] **Step 1: Write failing qualification tests**

```python
def test_requires_three_independent_qualifying_sessions(tmp_path):
    with pytest.raises(ValueError, match='three independent'):
        validate_sessions([session(tmp_path, samples=1000)] * 3)

def test_rejects_short_small_mismatched_or_faulted_sessions(session_set):
    for fault in ('duration', 'samples', 'checkpoint', 'gpu', 'sequence',
                  'lease', 'rejected', 'residual_process', 'nonfinite'):
        with pytest.raises(ValueError, match=fault):
            validate_sessions(session_set.with_fault(fault))
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_freshness_approval.py -q`

Expected: FAIL because the approval module does not exist.

- [ ] **Step 3: Implement strict evidence validation**

Open existing directories read-only with symlink-resistant identity checks. Require status completed, duration exactly 120 seconds or more, at least 1000 accepted samples, zero rejected samples, unique session IDs, matching checkpoint/config/GPU, strict sequence growth, fixed 2.5-second leases for identical source samples, fixed false authorization fields, finite summaries, and recorded clean process checks. Recompute maxima and percentiles from raw evidence instead of trusting summary values alone.

- [ ] **Step 4: Implement explicit proposal validation**

Accept no default values. Require six finite positive numbers. Require age/skew/mapping proposals to exceed every raw worst-case upper bound by an explicit recorded margin; retain TF task-contract ceilings of 0.005 m and 0.02 rad while ensuring observed maxima fit. Write a new exclusive artifact with evidence paths, hashes, GPU, checkpoint SHA-256, approval timestamp, margins, six limits, `thresholds_approved=true`, and `motion_authorized=false`. Never overwrite an artifact.

- [ ] **Step 5: Verify and commit**

Run approval tests, all tests, diff checks, and forbidden-import searches. Commit only the three Task 3 files as `feat: validate evidence for explicit freshness limits`.

---

### Task 4: Operator Documentation and Offline CUDA Smoke

**Files:**
- Modify: `docs/g2-clock-diagnostics.md`
- Modify: `docs/g2-adaptation-status.md`
- Modify: `README.md`
- Test: `tests/test_g2_actor_audit.py`

**Interfaces:**
- Consumes: Tasks 1–3 CLIs.
- Produces: exact monitor/audit/approval commands, evidence locations, stop order, and a no-GDK synthetic CUDA smoke using the actual checkpoint and RTX 3090.

- [ ] **Step 1: Add a subprocess smoke test**

Run the actual checkpoint through the real CUDA probe with synthetic correctly shaped uint8 images. Assert RTX 3090 identity, finite six-dimensional discarded action, nonzero synchronized timing, fixed false authorization flags, and no command/motion/Gym/learner imports.

- [ ] **Step 2: Document the three live sessions**

Use three new output paths, a monitor duration that covers warm-up plus 120 seconds, and Actor audit commands with the exact checkpoint. Document B-then-A Ctrl+C order, residual-process checks, evidence copying, and refusal to approve on any mismatch or fail-closed monitor exit during the formal window.

- [ ] **Step 3: Verify and commit**

Run the CUDA smoke, focused suite, complete suite, diff checks, and process scan. Commit only the four Task 4 files as `docs: add real actor audit workflow`.

---

### Task 5: Three Live Read-Only Sessions and Limit Approval

**Files:**
- Create after successful evidence review: `configs/g2_freshness_limits.json`
- Modify: `docs/g2-adaptation-status.md`

**Interfaces:**
- Consumes: three qualifying现场 evidence directories from Task 4.
- Produces: one explicit approved limits artifact and a status record; still no motion authorization.

- [ ] **Step 1: Preflight each session**

Verify no residual PTP/audit process, output paths do not exist, checkpoint SHA-256 is unchanged, CUDA identifies the RTX 3090, and the hardware E-stop is accessible. Do not instantiate a command port.

- [ ] **Step 2: Run three independent sessions**

For each session, start the fixed non-adjusting monitor, wait for a healthy mapping, run Actor audit for 120 seconds, stop B then A, and record a clean process scan. Never reuse or delete an evidence directory.

- [ ] **Step 3: Review raw evidence and propose limits**

Run the approval validator without an output path first. Examine p99, max, session-to-session spread, inference timings, mapping behavior, source freezes, and TF consistency. Choose explicit margins based on the observed worst case and document the reasoning for each of the six values.

- [ ] **Step 4: Write the approval artifact**

Run `approve_limits` with all six explicit values and a new `configs/g2_freshness_limits.json`. Confirm it records `thresholds_approved=true` and `motion_authorized=false`, references all three immutable evidence directories, and contains no default/fallback behavior.

- [ ] **Step 5: Commit evidence conclusions only**

Commit the approved JSON and status document as `docs: approve explicit G2 freshness limits`. Runtime raw evidence remains local and is referenced by immutable path/hash; no robot command is run.

---

### Task 6: Final Cross-Module Safety Review

**Files:**
- Review: all Task 1–5 changes and the approved spec.
- Modify only if a reproduced Critical/Important finding requires a regression fix.

- [ ] **Step 1: Request independent read-only review**

Check checkpoint trust boundary, preprocessing equivalence, CUDA synchronization, action disposal, import graph, audit ordering, transactional evidence, approval recomputation, no defaults, and the unchanged motion boundary.

- [ ] **Step 2: Reproduce and fix findings with TDD**

For every Critical/Important finding, add the smallest failing test, observe RED, implement one fix, observe GREEN, and request re-review. Do not change code for unreproduced speculation.

- [ ] **Step 3: Run final verification**

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests -q
git diff --check
git -C lerobot diff --check
rg -n 'GdkCommandPort|MotionBackend|G2LocalEnv|LearnerServiceStub|allow_motion\s*=\s*True' \
  g2_local/actor_inference.py g2_local/freshness_audit.py g2_local/freshness_approval.py
ps -eo pid,ppid,pgid,user,comm,args | rg 'ptp4l|phc2sys|clock_monitor|freshness_audit'
```

Expected: complete suite passes, diff checks are silent, forbidden search is empty, and process output contains only the inspection command.

- [ ] **Step 4: Handoff accurately**

Report approved freshness limits and their provenance. Explicitly state that ROI/task vision, E-stop/stop distance, action scale/workspace, motion commissioning, Gym step, and training remain unapproved.
