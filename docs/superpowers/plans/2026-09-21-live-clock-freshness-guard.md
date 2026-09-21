# Live Clock Freshness Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a continuously refreshed, fail-closed PTP clock mapping service and use it to reject stale or temporally ambiguous GDK observations before they enter a Gym step.

**Architecture:** An operator-started foreground monitor owns one fixed, non-adjusting `ptp4l` child and publishes short-lived immutable snapshots over a read-only Unix socket. A stateful observation guard consumes those snapshots plus expanded GDK timestamp evidence; it can only reject data and never changes `allow_motion`, control mode, or robot state.

**Tech Stack:** Python 3.10, stdlib subprocess/socket/threading/json, NumPy/SciPy, Agibot GDK 4.1.5, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-live-clock-freshness-guard-design.md`

## Global Constraints

- Never call `phc2sys`, adjust CLOCK_REALTIME/PHC, change GDK control mode, or send robot/gripper commands.
- The live monitor and all现场 validation run with real `allow_motion=False`.
- Actor/Gym never invokes sudo or controls the PTP process.
- A healthy clock/observation is only a necessary condition; it never sets or widens motion permission.
- Production freshness thresholds have no defaults and must be explicitly supplied after read-only load validation.
- Preserve the user's staged files and unrelated dirty-tree changes; each commit names only its task files.
- Source timestamps use the confirmed `raw_ptp` scale with explicit 37-second correction, while retaining `currentUtcOffsetValid` in evidence.

---

### Task 1: Rolling Clock Snapshot Model

**Files:**
- Create: `g2_local/live_clock.py`
- Create: `tests/test_g2_live_clock.py`
- Modify: `g2_local/clock_mapping.py`

**Interfaces:**
- Consumes: `clock_mapping.parse_ptp(line: str, master: str) -> dict | None` and the established PTP validation limits.
- Produces: `ClockSnapshot`, `ClockWindow(expected_master, boot_id, session_id)`, `ClockWindow.feed_ptp(line, received_mono_ns, wall_ns)`, `ClockWindow.feed_properties(raw, mono_ns)`, and `ClockWindow.snapshot(now_mono_ns, wall_ns) -> ClockSnapshot`.

- [ ] **Step 1: Write failing snapshot/window tests**

```python
def test_window_warms_then_publishes_short_raw_ptp_lease():
    window = ClockWindow('044052.fffe.000010', 'boot', 'run')
    feed_master_properties_and_8_samples(window)
    snap = window.snapshot(16_100_000_000, 1_000_016_100_000_000)
    assert snap.healthy is True
    assert snap.scale == 'raw_ptp'
    assert snap.valid_until_ns == snap.last_sample_mono_ns + 2_500_000_000

def test_repeating_snapshot_does_not_extend_old_lease():
    window = healthy_window()
    first = window.snapshot(16_000_000_000, wall(16))
    later = window.snapshot(17_000_000_000, wall(17))
    assert later.valid_until_ns == first.valid_until_ns

@pytest.mark.parametrize('fault', ['master', 'gap', 'drift', 'residual',
                                    'delay', 'properties', 'wall_jump', 'malformed'])
def test_any_clock_fault_latches_unhealthy(fault):
    window = healthy_window()
    inject_fault(window, fault)
    assert window.snapshot(now(), wall_now()).healthy is False
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_live_clock.py -q`

Expected: FAIL because `g2_local.live_clock` does not exist.

- [ ] **Step 3: Implement immutable snapshots and rolling validation**

Implement `ClockSnapshot` as a frozen dataclass containing every field listed in the
Task 1 Interfaces block. `ClockWindow.feed_ptp` accepts one raw line plus its receive
monotonic/wall timestamps, `feed_properties` accepts one raw PMC response and monotonic
timestamp, and `snapshot` accepts the current monotonic/wall timestamps and returns a
`ClockSnapshot`. Each method validates exact `int` nanoseconds before mutating state.

Use a bounded deque, refit on every accepted offset report, increment sequence only when publishing a new immutable object, retain the last PTP-derived expiry, and latch every structural fault until explicit reconstruction.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_clock_mapping.py tests/test_g2_live_clock.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add g2_local/live_clock.py g2_local/clock_mapping.py tests/test_g2_live_clock.py
git commit -m "feat: add rolling PTP clock snapshots"
```

---

### Task 2: Read-Only Snapshot Socket

**Files:**
- Create: `g2_local/clock_ipc.py`
- Create: `tests/test_g2_clock_ipc.py`

**Interfaces:**
- Consumes: `ClockSnapshot` from Task 1 and a thread-safe `Callable[[], ClockSnapshot]` provider.
- Produces: `SnapshotServer(path: Path, provider, *, request_limit=256)`, `SnapshotClient(path: Path, *, timeout_s: float, expected_master: str)`, `SnapshotClient.read() -> ClockSnapshot`, and idempotent `close()` methods.

- [ ] **Step 1: Write failing protocol and permission tests**

```python
def test_client_reads_valid_snapshot_and_checks_boot_master_sequence(tmp_path):
    server = running_server(tmp_path/'clock.sock', healthy_snapshot(sequence=3))
    client = SnapshotClient(server.path, timeout_s=.05,
                            expected_master='044052.fffe.000010')
    assert client.read().sequence == 3

@pytest.mark.parametrize('request', [b'{}\n', b'{"op":"stop","schema":1}\n', b'x'*257])
def test_server_rejects_every_non_snapshot_request(server, request):
    reply = exchange(server.path, request)
    assert json.loads(reply) == {'schema': 1, 'ok': False, 'error': 'invalid_request'}

def test_client_rejects_expired_wrong_boot_wrong_master_and_sequence_rollback(client, server):
    for snapshot in invalid_identity_or_lease_snapshots():
        server.set_snapshot(snapshot)
        with pytest.raises(ValueError):
            client.read()

def test_socket_and_parent_are_current_user_only(tmp_path):
    server = running_server(tmp_path/'private'/'clock.sock', healthy_snapshot())
    assert stat.S_IMODE(server.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(server.path.stat().st_mode) == 0o600
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_clock_ipc.py -q`

Expected: FAIL with missing `g2_local.clock_ipc`.

- [ ] **Step 3: Implement the bounded JSON protocol**

Implement one request per connection, exact request shape
`{"op":"snapshot","schema":1}`, 256-byte request and 4096-byte response limits,
finite socket timeouts, strict response key/type/range validation, local boot-ID lookup,
sequence monotonicity, lease validation against local monotonic time, and no mutation RPC.
Bind via a newly created mode-0700 directory and chmod the socket to 0600.

- [ ] **Step 4: Verify protocol tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_clock_ipc.py -q`

Expected: PASS with no leaked socket-server threads.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add g2_local/clock_ipc.py tests/test_g2_clock_ipc.py
git commit -m "feat: publish clock snapshots over read-only IPC"
```

---

### Task 3: Foreground Clock Monitor and PTP Lifecycle

**Files:**
- Create: `g2_local/clock_monitor.py`
- Create: `tests/test_g2_clock_monitor.py`
- Modify: `g2_local/clock_probe.py`
- Modify: `docs/g2-clock-diagnostics.md`

**Interfaces:**
- Consumes: `ClockWindow`, `SnapshotServer`, fixed PTP identity/interface, and explicit `--max-seconds` in `[60, 43200]`.
- Produces: `ptp_monitor_command(max_seconds, uds, uds_ro) -> list[str]`, `MonitorRuntime.run() -> int`, and CLI `python -m g2_local.clock_monitor`.

- [ ] **Step 1: Write failing command/lifecycle tests**

```python
def test_fixed_command_is_non_adjusting_get_only_and_finitely_bounded():
    cmd = ptp_monitor_command(3600, '/var/run/g2-live-a', '/var/run/g2-live-a-ro')
    joined = ' '.join(cmd)
    assert '--free_running=1' in cmd and '-S' in cmd and '-s' in cmd
    assert 'phc2sys' not in joined and '--kill-after=3s' in cmd
    assert '3600s' in cmd and '--uds_ro_file_mode=0666' in cmd

def test_child_exit_publishes_unhealthy_and_monitor_exits(runtime):
    runtime.process.finish(7)
    assert runtime.run() == 7
    assert runtime.provider().healthy is False

def test_monitor_only_signals_its_known_process_group(runtime):
    runtime.request_stop()
    runtime.run()
    assert runtime.process.signalled_groups == [runtime.process.pgid]

def test_existing_evidence_or_socket_is_rejected_before_sudo(runtime, tmp_path):
    runtime.output.mkdir()
    with pytest.raises(FileExistsError):
        runtime.run()
    assert runtime.process.started is False
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_clock_monitor.py -q`

Expected: FAIL with missing monitor module/API.

- [ ] **Step 3: Implement the operator-owned foreground monitor**

Use a fixed `sudo -n /usr/bin/timeout --signal=INT --kill-after=3s 43200s
/usr/bin/stdbuf -oL -eL /usr/sbin/ptp4l` argv with no shell, substituting only the
validated explicit duration in the range 60–43200 seconds. Create one session-specific
PTP UDS pair, query only the read-only UDS with unprivileged `pmc`, write bounded JSONL
events for PTP lines/properties/mapping state changes, and publish an unhealthy snapshot
before normal shutdown. Never restart the PTP child automatically.

- [ ] **Step 4: Reuse parser code without changing the diagnostic probe contract**

Move only shared parsing/command validation helpers out of `clock_probe.py`; keep
`clock_probe --seconds 45 --master 044052.fffe.000010` behavior and its existing tests unchanged.

- [ ] **Step 5: Run monitor, probe, and lifecycle tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_clock_monitor.py tests/test_g2_clock_probe.py tests/test_g2_live_clock.py -q`

Expected: PASS; no test invokes real sudo or hardware.

- [ ] **Step 6: Document exact startup and shutdown commands**

Document `sudo -v && bash run_g2_python.sh -m g2_local.clock_monitor --master
044052.fffe.000010 --max-seconds 43200`; the CLI creates a new random runtime directory;
state that Ctrl+C stops only the owned session, system time is unchanged, and Actor starts separately.

- [ ] **Step 7: Commit only Task 3 files**

```bash
git add g2_local/clock_monitor.py g2_local/clock_probe.py tests/test_g2_clock_monitor.py docs/g2-clock-diagnostics.md
git commit -m "feat: add supervised live clock monitor"
```

---

### Task 4: Expand GDK Observation Timestamp Evidence

**Files:**
- Modify: `g2_local/gdk_backend.py`
- Create: `tests/test_g2_gdk_reader.py`
- Modify: `g2_local/clock_probe.py`

**Interfaces:**
- Consumes: installed GDK `TF`, camera frames, joint states, and existing motion pose.
- Produces: unchanged observation keys plus `last_info` fields `read_start_*`, `read_end_*`, `sdk_clock_ns`, `source_timestamp_ns`, `tf_queries`, `tf_position_error_m`, and `tf_rotation_error_rad`.

- [ ] **Step 1: Write failing reader-contract tests with complete fake GDK data**

```python
def test_observe_exposes_all_source_times_without_changing_policy_observation(reader):
    obs = reader.observe()
    assert set(obs) == {'state', 'left_wrist', 'right_aux'}
    assert reader.last_info['source_timestamp_ns'] == {
        'left_wrist': 1001, 'right_aux': 1002, 'joint': 1003, 'tf': 1004}
    assert len(reader.last_info['tf_queries']) == 2

def test_tf_runtime_failure_rejects_observation_instead_of_reusing_old_value(reader):
    reader.tf.fail_next = True
    with pytest.raises(RuntimeError, match='transform'):
        reader.observe()
    assert reader.last_info['source_timestamp_ns']['tf'] == 1004

def test_reader_waits_boundedly_for_both_tf_directions_on_startup(reader_factory):
    reader = reader_factory(tf_failures_before_ready=2)
    assert reader.tf.lookup_calls == 4
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_gdk_reader.py -q`

Expected: FAIL because current `GdkReader.last_info` lacks joint/TF evidence.

- [ ] **Step 3: Implement evidence capture in one observation transaction**

Construct and warm `gdk.TF` in `GdkReader`, retain both raw direction queries, compare the
installed-SDK direction matching motion pose, validate finite/unit quaternions, and never cache
source timestamps as fallback. Factor the existing probe to call the same evidence helper so
diagnostic and live code cannot disagree on TF orientation.

- [ ] **Step 4: Run reader and prior hardware-boundary tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_gdk_reader.py tests/test_g2_command_port.py tests/test_g2_clock_probe.py -q`

Expected: PASS without importing real GDK in unit tests.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add g2_local/gdk_backend.py g2_local/clock_probe.py tests/test_g2_gdk_reader.py
git commit -m "feat: expose GDK source timestamp evidence"
```

---

### Task 5: Stateful Observation Freshness Guard

**Files:**
- Create: `g2_local/freshness.py`
- Create: `tests/test_g2_freshness.py`

**Interfaces:**
- Consumes: `SnapshotClient.read()`, expanded `GdkReader.last_info`, and optional `after: float` seconds from `CommandStream.wait_sent`.
- Produces: `FreshnessLimits` with six required finite positive fields and `ObservationFreshnessGuard.__call__(obs, info, after) -> bool`; exposes immutable `last_decision` for diagnostics.

- [ ] **Step 1: Write failing explicit-config and interval tests**

```python
def limits():
    return FreshnessLimits(camera_age_s=.100, state_age_s=.050,
                           camera_skew_s=.050, tf_position_error_m=.005,
                           tf_rotation_error_rad=.020, mapping_error_s=.005)

def test_no_limit_has_a_production_default():
    with pytest.raises(TypeError):
        FreshnessLimits()

def test_fresh_sources_commit_strictly_increasing_state(guard):
    assert guard(obs(), info(stamps=(1001, 1002, 1003, 1004)), after=None) is True
    assert guard.previous_source_ns == {'left_wrist': 1001, 'right_aux': 1002,
                                        'joint': 1003, 'tf': 1004}
@pytest.mark.parametrize('fault', ['expired_map', 'map_error', 'camera_age', 'state_age',
                                    'skew', 'future', 'frozen', 'reversed', 'tf_pose'])
def test_each_fault_rejects_without_committing_failed_timestamps(guard, fault):
    before = guard.previous_source_ns.copy()
    assert guard(obs(), info_with_fault(fault), after=None) is False
    assert guard.previous_source_ns == before

def test_post_action_requires_every_source_interval_strictly_after_send():
    guard = healthy_guard()
    assert guard(obs(), info(source_time_after_send=True), after=10.0) is True
    assert guard(obs(), info(interval_crosses_send=True), after=10.0) is False
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_freshness.py -q`

Expected: FAIL with missing freshness module.

- [ ] **Step 3: Implement interval conversion and transactional state**

Convert each raw PTP source timestamp to a local monotonic interval using the snapshot reference
offset, drift, wall-minus-monotonic origin, and empirical error. Check snapshot error against the
explicit limit before evaluating ages. Stage all timestamp updates locally and assign them to the
guard only after every test passes. Store decision codes such as `mapping_expired`,
`camera_stale:left_wrist`, `source_frozen:joint`, and `not_after_command:tf`.

- [ ] **Step 4: Run freshness tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_freshness.py -q`

Expected: PASS for every boundary and one-nanosecond outside-boundary case.

- [ ] **Step 5: Commit only Task 5 files**

```bash
git add g2_local/freshness.py tests/test_g2_freshness.py
git commit -m "feat: add fail-closed observation freshness guard"
```

---

### Task 6: Wire the Guard into MotionBackend/Gym Failure Semantics

**Files:**
- Modify: `g2_local/motion_backend.py`
- Modify: `tests/test_g2_motion_backend.py`
- Modify: `g2_local/env.py`
- Modify: `tests/test_g2_gym.py`

**Interfaces:**
- Consumes: the Task 5 callable guard through the existing `observation_guard(obs, info, after)` boundary.
- Produces: observable rejection reason in raised errors/info while retaining permanent backend lockout and existing stop semantics.

- [ ] **Step 1: Write failing end-to-end fail-closed tests**

```python
def test_mapping_failure_before_plan_sends_nothing_and_stops_backend():
    driver, port = live_guard_backend(guard_rejects='mapping_expired')
    with pytest.raises(RuntimeError, match='mapping_expired'):
        driver.execute(np.zeros(6))
    assert port.sent == []
    assert driver.stopped is True

def test_ambiguous_successor_stops_once_and_never_auto_recovers():
    driver, port = live_guard_backend(reject_after_send='not_after_command:tf')
    with pytest.raises(RuntimeError, match='not_after_command:tf'):
        driver.execute(np.zeros(6))
    assert port.stop_calls == 1
    make_clock_healthy()
    with pytest.raises(RuntimeError, match='reconstruct'):
        driver.observe()
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_motion_backend.py tests/test_g2_gym.py -q`

Expected: new reason-propagation assertions FAIL while existing stop tests remain green.

- [ ] **Step 3: Preserve diagnostic reason without weakening the boolean contract**

When a guard returns anything other than exactly `True`, read its immutable `last_decision.code`
when available and include it in the existing freshness exception. Do not let the guard stop the
port itself; keep `_abort()` as the single stop owner. Do not add automatic reconstruction to Gym reset.

- [ ] **Step 4: Run all execution-chain tests**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_motion_backend.py tests/test_g2_command_stream.py tests/test_g2_gym.py -q`

Expected: PASS, including zero sends before a rejected initial observation and one terminal stop after a rejected successor.

- [ ] **Step 5: Commit only Task 6 files**

```bash
git add g2_local/motion_backend.py g2_local/env.py tests/test_g2_motion_backend.py tests/test_g2_gym.py
git commit -m "feat: enforce freshness failures in Gym execution"
```

---

### Task 7: Read-Only Load Audit and Operator Documentation

**Files:**
- Create: `g2_local/freshness_audit.py`
- Create: `tests/test_g2_freshness_audit.py`
- Modify: `docs/g2-clock-diagnostics.md`
- Modify: `docs/g2-adaptation-status.md`

**Interfaces:**
- Consumes: running snapshot socket, `GdkReader`, explicit duration `[30, 1800]`, and optional finite simulated inference delay.
- Produces: a new evidence directory with raw JSONL and `summary.json`; always reports `motion_authorized=false` and never constructs `GdkCommandPort` or `MotionBackend`.

- [ ] **Step 1: Write failing bounded-audit tests**

```python
def test_audit_summary_reports_distributions_without_approving_thresholds(tmp_path):
    report = summarize_audit(rows(), snapshots())
    assert report['motion_authorized'] is False
    assert report['thresholds_approved'] is False
    assert report['camera_age_ms']['left_wrist']['p99'] == 48.0

def test_audit_rejects_missing_monitor_frozen_sources_and_existing_output(tmp_path):
    with pytest.raises((ConnectionError, ValueError, FileExistsError)):
        run_audit(output=tmp_path, client=disconnected_client(), reader=frozen_reader())

def test_audit_cli_has_duration_bound_and_no_motion_dependency():
    with pytest.raises(ValueError):
        validate_audit_duration(1801)
    assert 'g2_local.gdk_backend.GdkCommandPort' not in audit_imports()
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_freshness_audit.py -q`

Expected: FAIL with missing audit module.

- [ ] **Step 3: Implement the read-only audit**

Capture source-age intervals, camera skew, mapping error/residual/drift, GDK read duration/gaps,
TF/motion differences, monitor snapshot gaps and simulated-inference timing. Report min/p50/p95/p99/max
using literal units. Do not emit a boolean recommendation to enable motion and do not write production config.

- [ ] **Step 4: Document the two-terminal workflow**

Document terminal A (`clock_monitor`) and terminal B (`freshness_audit`), exact Ctrl+C order,
evidence locations, how to verify no residual process, and the separate future procedure for
approving explicit thresholds. Record that the operator now reports access to hardware E-stop but
that controlled E-stop/stop-distance validation remains unperformed.

- [ ] **Step 5: Run the focused and complete suites**

Run:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_live_clock.py tests/test_g2_clock_ipc.py tests/test_g2_clock_monitor.py tests/test_g2_gdk_reader.py tests/test_g2_freshness.py tests/test_g2_freshness_audit.py tests/test_g2_motion_backend.py -q
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests -q
git diff --check
git -C lerobot diff --check
```

Expected: all tests pass; only the two already documented Gym infinite-Box warnings may remain.

- [ ] **Step 6: Perform a read-only smoke test only after software verification**

Start the monitor with `--max-seconds 120`, then run the audit for 60 seconds with
`allow_motion=False`. Verify snapshot sequence advances, the lease never extends without a new
PTP sample, both processes exit, no `ptp4l/phc2sys/clock_monitor` remains, and evidence reports
`motion_authorized=false`. Do not instantiate a command port or send a hold.

- [ ] **Step 7: Commit only Task 7 files**

```bash
git add g2_local/freshness_audit.py tests/test_g2_freshness_audit.py docs/g2-clock-diagnostics.md docs/g2-adaptation-status.md
git commit -m "feat: add read-only freshness load audit"
```

---

### Task 8: Final Safety and Requirements Review

**Files:**
- Review: `docs/superpowers/specs/2026-09-21-live-clock-freshness-guard-design.md`
- Review: all files changed in Tasks 1–7

**Interfaces:**
- Consumes: all task deliverables.
- Produces: review findings resolved, current verification evidence, and a handoff that does not claim motion authorization.

- [ ] **Step 1: Request a read-only code review against the approved spec**

Reviewer must check clock-scale math/signs, snapshot lease semantics, privilege/process cleanup,
socket permissions and parser limits, transactional guard state, command-before-guard ordering,
stop ownership, test realism, and absence of any path that enables motion.

- [ ] **Step 2: Reproduce each Critical/Important finding before changing code**

For each valid finding, add the smallest failing regression test, run it RED, implement one fix,
and run it GREEN. Push back with code evidence on findings that conflict with the approved spec.

- [ ] **Step 3: Run final verification from a clean process state**

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests -q
git diff --check
git -C lerobot diff --check
ps -eo pid,ppid,user,comm,args | rg 'ptp4l|phc2sys|clock_monitor|freshness_audit'
```

Expected: complete suite passes; diff checks are silent; process listing contains only the inspection command.

- [ ] **Step 4: Verify safety statements directly**

Confirm by code search and tests that the new monitor/audit import neither `GdkCommandPort` nor a
robot command API, `free_running=1` is mandatory, `phc2sys` is rejected, all snapshot health paths
leave real `allow_motion=False`, and threshold values appear only in tests/docs or explicit caller config.

- [ ] **Step 5: Commit review fixes, if any, with explicit paths**

Stage only the concrete files changed for validated findings, chosen from
`g2_local/live_clock.py`, `g2_local/clock_ipc.py`, `g2_local/clock_monitor.py`,
`g2_local/gdk_backend.py`, `g2_local/freshness.py`, `g2_local/motion_backend.py`,
`g2_local/freshness_audit.py` and their matching test files, then run:

```bash
git commit -m "fix: harden live freshness safety boundaries"
```

Do not create an empty commit when review finds no issues.
