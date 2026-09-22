# G2 Real SiLRI Training Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable, resumable two-process Real Actor–Learner runtime that trains SiLRI from freshness-gated G2 transitions with motion-triggered SpaceMouse intervention, Y/F labels, and upstream visual resets.

**Architecture:** GDK, Gym, operator input, and episode state live only in Real Actor; policy optimization and replay live only in Learner; the existing loopback gRPC service carries bounded transition batches and versioned Actor parameters. Focused modules own configuration, operator events, episode state, motion assembly, Actor, Learner, and launch lifecycle.

**Tech Stack:** Python 3.10, PyTorch 2.7, Gymnasium, NumPy, gRPC/protobuf, LeRobot SiLRI and ReplayBuffer, AgiBot GDK, Linux hidraw/termios, JSON/JSONL.

**Spec:** `docs/superpowers/specs/2026-09-22-real-silri-training-runtime-design.md`

## Global Constraints

- Policy action is normalized `(x,y,z,roll,pitch,yaw)`; gripper is fixed.
- Upstream vision owns grasp, hole localization, coarse approach, and reset. RL never executes a fixed reset trajectory.
- Motion needs config intent plus CLI `--allow-motion`; loading JSON alone cannot create a command port.
- PTP/freshness, control mode, workspace, stop evidence, E-stop, and motion permission remain independent gates.
- SpaceMouse movement engages intervention; only fresh neutral held for the configured interval releases it. Read error, unplug, malformed input, or stale nonzero input aborts.
- Left button selects translation/rotation. A press-and-release chord of both buttons starts only from `WAITING_FOR_RESET`.
- `Y` is terminal success; `F` is terminal failure; timeout is truncation without a fabricated label.
- Critic trains `executed_action`, never the raw policy or human proposal.
- Automated tests use isolated fakes and send no real robot command.
- Preserve the user's staged Chinese files and GDK logs; commits use exact pathspecs.

## Review Focus

- Silent HID may not release active intervention; unplug/read failure must abort.
- Y/F or a chord during an in-flight step must produce at most one terminal transition after a valid successor.
- Learner disconnect/backpressure must stop Actor before queues grow unbounded or parameters become indefinitely stale.
- Duplicate transitions, rolled-back parameters, and mismatched run/config identity must not mutate replay or policy.
- Resume restores software state only and always returns physically to `WAITING_FOR_RESET`.

## File Responsibility Map

- `g2_local/training_config.py`: exact schema, immutable manifest, commissioning evidence, and motion-permission conjunction.
- `g2_local/operator_control.py`: bounded keyboard input, SpaceMouse start chord, and upstream context inbox.
- `g2_local/real_episode.py`: serialized episode/step tokens and terminal outcome state.
- `g2_local/motion_env.py`: the only commissioned assembly path from clock/GDK factories to a motion-capable Gym environment.
- `g2_local/real_actor.py`: inference, action arbitration, real transition construction, bounded uplink, and Actor shutdown.
- `g2_local/real_learner.py`: identity validation, dual replay, SiLRI updates, parameter publication, and checkpoint/resume.
- `g2_local/real_train.py`: role-specific CLI, preflight, run evidence, train/eval lifecycle, and exit codes.
- `tests/test_g2_real_training_integration.py`: formal two-process proof with injected hardware boundaries; it is not a robot-performance test.

---

### Task 1: Versioned Configuration and Immutable Manifest

**Files:**
- Create: `g2_local/training_config.py`
- Create: `configs/g2_real_training_readonly.json`
- Create: `tests/test_g2_training_config.py`
- Modify: `g2_local/config.py`

**Interfaces:**
- Consumes: `HingeInsertTaskConfig`, `LocalTaskConfig`, `FreshnessLimits`.
- Produces: `ObservationConfig`; `InterventionConfig`; `OptimizationConfig`; `RuntimeConfig`; `CommissioningConfig`; `MotionRuntimeConfig`; `load_training_config(path: Path, *, cli_allow_motion: bool) -> LoadedTrainingConfig`; `LoadedTrainingConfig.write_manifest(output: Path, *, run_id: str, role: str) -> Path`; `config_hash`; `motion_permitted`.

- [ ] **Step 1: Write failing exact-schema and permission tests**

```python
def valid_payload():
    return {
        'schema': 1, 'mode': 'train', 'requested_motion': False,
        'task': {'control_hz': 10.0, 'max_episode_steps': 80,
                 'fix_gripper': True, 'action_scale': [.0015, .0015, .0015,
                                                       .026, .026, .026],
                 'success_reward': 10.0, 'failure_reward': -1.0,
                 'step_reward': -.05, 'reward_source': 'human',
                 'target_xy_range_m': .05, 'ee_xyz_range_m': .003,
                 'ee_rpy_range_rad': .008726646259971648},
        'motion': {'workspace_low': [.20, .20, .70],
                   'workspace_high': [.40, .50, 1.00], 'control_mode': 1,
                   'command_timeout_s': .25, 'send_timeout_s': .05,
                   'stop_timeout_s': 1.0, 'reader_timeout_s': 2.0,
                   'command_lifetime_s': .1, 'send_rate_hz': 50.0,
                   'adapter_root': '/home/flyfuture/g2_hinge_assembly'},
        'freshness': {'camera_age_s': .1, 'state_age_s': .05,
                      'camera_skew_s': .05, 'tf_position_error_m': .005,
                      'tf_rotation_error_rad': .02, 'mapping_error_s': .005},
        'observation': {'camera_keys': ['left_wrist', 'right_aux'],
                        'image_size': 128,
                        'camera_rois': {'left_wrist': [0, 0, 1280, 1056],
                                        'right_aux': [0, 0, 1280, 1056]},
                        'raw_rgb_logging': False},
        'intervention': {'axis_map': [-2, -1, -3], 'left_button': 0,
                         'right_button': 1, 'engage_deadzone': .12,
                         'release_deadzone': .08, 'release_hold_s': .25,
                         'report_max_age_s': .25},
        'optimization': {'online_capacity': 100000, 'human_capacity': 50000,
                         'min_online_transitions': 256, 'online_batch_size': 128,
                         'human_batch_size': 128, 'utd_ratio': 1,
                         'actor_lr': 3e-4, 'critic_lr': 3e-4,
                         'expert_lr': 3e-4, 'lagrange_lr': 3e-4,
                         'target_update_interval': 1, 'publish_interval': 1,
                         'checkpoint_interval': 1000},
        'runtime': {'seed': 1234, 'device': 'cuda', 'learner_host': '127.0.0.1',
                    'learner_port': 50175, 'queue_capacity': 64,
                    'queue_put_timeout_s': 1.0, 'transport_timeout_s': 5.0,
                    'context_max_age_s': 30.0, 'parameter_heartbeat_s': 1.0,
                    'learner_silence_timeout_s': 5.0,
                    'operator_poll_interval_s': .01},
        'commissioning': {'profile': 'unapproved', 'clock_socket': '/run/g2/clock.sock',
                          'expected_master': '044052.fffe.000010',
                          'evidence': []}}

def write_config(tmp_path, payload=None):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(valid_payload() if payload is None else payload))
    return path

def test_config_needs_cli_and_approved_evidence_before_motion(tmp_path):
    payload = valid_payload(); payload['requested_motion'] = True
    loaded = load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)
    assert loaded.motion_permitted is False
    assert loaded.task.action_scale == (.0015, .0015, .0015, .026, .026, .026)

def test_manifest_is_canonical_and_refuses_existing_output(tmp_path):
    loaded = load_training_config(write_config(tmp_path), cli_allow_motion=False)
    manifest = loaded.write_manifest(tmp_path / 'run', run_id='run-1', role='learner')
    assert json.loads(manifest.read_text())['config_sha256'] == loaded.config_hash
    with pytest.raises(FileExistsError):
        loaded.write_manifest(tmp_path / 'run', run_id='run-1', role='learner')

def bad_workspace(payload):
    payload['motion']['workspace_high'] = [.1, .1, .1]
    return payload

def missing_freshness(payload):
    del payload['freshness']
    return payload

def reversed_deadzone_hysteresis(payload):
    payload['intervention']['release_deadzone'] = .2
    return payload

def unknown_key(payload):
    payload['unknown'] = 1
    return payload

@pytest.mark.parametrize('mutation', [bad_workspace, missing_freshness,
                                      reversed_deadzone_hysteresis, unknown_key])
def test_invalid_or_ambiguous_config_is_rejected(tmp_path, mutation):
    payload = mutation(valid_payload())
    with pytest.raises(ValueError):
        load_training_config(write_config(tmp_path, payload), cli_allow_motion=False)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_training_config.py -q`

Expected: collection fails because `g2_local.training_config` does not exist.

- [ ] **Step 3: Implement exact parsing and hashing**

```python
@dataclass(frozen=True)
class ObservationConfig:
    camera_keys: tuple[str, str]
    image_size: int
    camera_rois: Mapping[str, tuple[int, int, int, int]]
    raw_rgb_logging: bool

@dataclass(frozen=True)
class InterventionConfig:
    axis_map: tuple[int, int, int]
    left_button: int
    right_button: int
    engage_deadzone: float
    release_deadzone: float
    release_hold_s: float
    report_max_age_s: float

@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    learner_host: str
    learner_port: int
    queue_capacity: int
    queue_put_timeout_s: float
    transport_timeout_s: float
    context_max_age_s: float
    parameter_heartbeat_s: float
    learner_silence_timeout_s: float
    operator_poll_interval_s: float

@dataclass(frozen=True)
class MotionRuntimeConfig:
    limits: LocalTaskConfig
    control_mode: int
    command_timeout_s: float
    send_timeout_s: float
    stop_timeout_s: float
    reader_timeout_s: float
    command_lifetime_s: float
    send_rate_hz: float
    adapter_root: Path

@dataclass(frozen=True)
class CommissioningEvidence:
    kind: str
    path: Path
    sha256: str

@dataclass(frozen=True)
class CommissioningConfig:
    profile: str
    clock_socket: Path
    expected_master: str
    evidence: Sequence[CommissioningEvidence]

@dataclass(frozen=True)
class OptimizationConfig:
    online_capacity: int
    human_capacity: int
    min_online_transitions: int
    online_batch_size: int
    human_batch_size: int
    utd_ratio: int
    actor_lr: float
    critic_lr: float
    expert_lr: float
    lagrange_lr: float
    target_update_interval: int
    publish_interval: int
    checkpoint_interval: int

@dataclass(frozen=True)
class LoadedTrainingConfig:
    schema: int
    mode: str
    task: HingeInsertTaskConfig
    motion: MotionRuntimeConfig
    freshness: FreshnessLimits
    observation: ObservationConfig
    intervention: InterventionConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    commissioning: CommissioningConfig
    requested_motion: bool
    motion_permitted: bool
    config_hash: str
    canonical_payload: Mapping[str, object]

def load_training_config(path: Path, *, cli_allow_motion: bool) -> LoadedTrainingConfig:
    payload = read_owned_regular_json(path, max_bytes=131072)
    reject_unknown_or_missing_keys(payload, SCHEMA_ONE_KEYS)
    parsed = parse_schema_one(payload)
    approved = parsed.commissioning.verify_files_and_hashes()
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return replace(parsed, motion_permitted=bool(
        cli_allow_motion and parsed.requested_motion and approved),
        config_hash=digest, canonical_payload=MappingProxyType(payload))
```

Implement `read_owned_regular_json`, `reject_unknown_or_missing_keys`, `parse_schema_one`, `canonical_json`, and `CommissioningConfig.verify_files_and_hashes` in this module. `write_manifest` creates a new mode-0700 output directory, writes a mode-0400 canonical manifest containing schema/run/role/config hash and the full parsed payload, fsyncs file and directory, and refuses reuse. The evidence verifier accepts only owned nonsymlink regular files whose SHA-256 equals the configured digest; an approved profile requires named hashes for freshness approval, XYZ/RPY direction-and-scale commissioning, software-stop/lease-expiry evidence, and hardware-E-stop evidence. Reject bool-as-number, unknown/missing keys, nonfinite values, `release_deadzone > engage_deadzone`, ROI keys that differ from the two camera keys, invalid ROI/timing/heartbeat relationships, motion without workspace, and symlinked evidence. Add `failure_reward: float = -1.0` to `HingeInsertTaskConfig` with the same finite-number validation as the other rewards. The checked-in JSON is read-only (`requested_motion=false`) and records axis map `[-2,-1,-3]` without claiming approved workspace/freshness evidence.

- [ ] **Step 4: Run config regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_training_config.py tests/test_g2_config.py tests/test_g2_freshness.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- g2_local/training_config.py g2_local/config.py configs/g2_real_training_readonly.json tests/test_g2_training_config.py
git commit --only -m "feat: add real training configuration manifest" -- g2_local/training_config.py g2_local/config.py configs/g2_real_training_readonly.json tests/test_g2_training_config.py
```

### Task 2: Motion-Triggered SpaceMouse Intervention

**Files:**
- Modify: `g2_local/spacemouse.py`
- Modify: `tests/test_g2_intervention.py`
- Modify: `tests/test_g2_live_input.py`

**Interfaces:**
- Consumes: `LiveInputGate.update(frame, now) -> Proposal`; `InterventionConfig`.
- Produces: `AutomaticIntervention(reader, config, *, clock=time.monotonic)` callable returning `(active, action_or_none)` and exposing `last_frame`.

- [ ] **Step 1: Write failing engage/release/fault tests**

```python
def test_motion_engages_and_only_fresh_held_neutral_releases():
    source, reader, now = automatic_source()
    reader.frame = frame(axes=(.6,0,0,0,0,0), stamps=(1.,1.))
    assert source()[0] is True
    reader.frame = frame(stamps=(1.1,1.1)); now[0] = 1.1
    assert source()[0] is True
    reader.frame = frame(stamps=(1.4,1.4)); now[0] = 1.4
    assert source() == (False, None)

def test_silent_neutral_never_releases_active_intervention():
    source, reader, now = engaged_source()
    reader.frame = frame(stamps=(1.,1.)); now[0] = 5.
    assert source()[0] is True

def test_unplug_latches_fault_instead_of_policy_fallback():
    source = AutomaticIntervention(BrokenReader(), explicit_config())
    with pytest.raises(OSError): source()
    with pytest.raises(RuntimeError, match='latched'): source()
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_intervention.py tests/test_g2_live_input.py -q`

Expected: FAIL because `AutomaticIntervention` is absent.

- [ ] **Step 3: Implement hysteretic takeover**

```python
class AutomaticIntervention:
    def __call__(self):
        frame = self.reader.poll()
        now = self.clock()
        proposal = self.gate.update(frame, now=now)
        moving = max(map(abs, proposal.action)) > self.config.engage_deadzone
        if moving:
            self.active, self.neutral_since = True, None
        elif self.active and self.gate.fresh and self._raw_neutral(frame):
            self.neutral_since = now if self.neutral_since is None else self.neutral_since
            if now - self.neutral_since >= self.config.release_hold_s:
                self.active, self.neutral_since = False, None
        elif self.active:
            self.neutral_since = None
        return (True, proposal.action) if self.active else (False, None)
```

Preserve legacy `HumanInput`. Latch and re-raise all reader/gate failures.

- [ ] **Step 4: Run SpaceMouse regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_spacemouse.py tests/test_g2_live_input.py tests/test_g2_intervention.py tests/test_g2_calibration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- g2_local/spacemouse.py tests/test_g2_intervention.py tests/test_g2_live_input.py
git commit --only -m "feat: add motion-triggered SpaceMouse intervention" -- g2_local/spacemouse.py tests/test_g2_intervention.py tests/test_g2_live_input.py
```

### Task 3: Keys, Start Chord, Context Inbox, and Episode State

**Files:**
- Create: `g2_local/operator_control.py`
- Create: `g2_local/real_episode.py`
- Create: `tests/test_g2_operator_control.py`
- Create: `tests/test_g2_real_episode.py`
- Modify: `g2_local/contract.py`

**Interfaces:**
- Consumes: `AutomaticIntervention.last_frame`; `EpisodeContext`.
- Produces: `TerminalKeyReader.poll() -> Literal['success', 'failure'] | None`; `StartChord.update(frame) -> bool`; `EpisodeContextInbox.read_new() -> EpisodeContext`; `RealEpisodeCoordinator.begin_step() -> StepToken`; `RealEpisodeCoordinator.outcome(observation) -> OutcomeDecision`; `RealEpisodeCoordinator.abort_step(token) -> None`.

- [ ] **Step 1: Write failing event/state tests**

```python
def test_chord_requires_both_buttons_then_full_release():
    chord = StartChord(left_button=0, right_button=1)
    assert chord.update(frame(buttons=(True,True), pressed=(0,1))) is False
    assert chord.update(frame(buttons=(False,False))) is True
    assert chord.update(frame(buttons=(False,False))) is False

def test_y_during_step_is_consumed_once_after_successor():
    machine = running_episode()
    token = machine.begin_step()
    machine.request_terminal('success')
    assert machine.finish_step(token) == OutcomeDecision(10., True, 'human', True)
    with pytest.raises(RuntimeError, match='not running'): machine.begin_step()

def test_duplicate_context_is_rejected(tmp_path):
    inbox = EpisodeContextInbox(tmp_path / 'context.json')
    write_context(inbox.path, episode_id='episode-1')
    assert inbox.read_new().episode_id == 'episode-1'
    with pytest.raises(ValueError, match='duplicate'): inbox.read_new()

def test_stale_or_out_of_range_reset_context_cannot_start():
    machine = waiting_episode(now_ns=40_000_000_000,
                              context_max_age_s=5.0,
                              target_xy_range_m=.05)
    with pytest.raises(ValueError, match='stale reset context'):
        machine.offer_context(context(reset_monotonic_ns=1_000_000_000))
    with pytest.raises(ValueError, match='target offset outside configured range'):
        machine.offer_context(context(reset_monotonic_ns=39_000_000_000,
                                      target_offset_m=(.06, 0., 0.)))

def test_terminal_key_buffer_is_flushed_before_episode_start():
    machine = waiting_episode(keys=('y',))
    machine.offer_context(context())
    machine.confirm_start_chord()
    assert machine.begin_step().step_id == 0
    assert machine.outcome(valid_successor()) == OutcomeDecision(-.05, False,
                                                                  'human', None)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_operator_control.py tests/test_g2_real_episode.py -q`

Expected: collection fails because the modules are missing.

- [ ] **Step 3: Implement bounded key/context input**

```python
class TerminalKeyReader:
    def poll(self):
        labels = {k.lower() for k in self.source.read_available(limit=32)
                  if k.lower() in ('y','f')}
        if labels == {'y'}: return 'success'
        if labels == {'f'}: return 'failure'
        if labels: raise RuntimeError('Conflicting Y/F terminal input')
        return None

class EpisodeContextInbox:
    def read_new(self):
        payload = self._read_owned_regular_json(max_bytes=16_384)
        context = EpisodeContext.from_payload(payload)
        if context.episode_id in self._seen:
            raise ValueError('duplicate episode context')
        self._seen.add(context.episode_id)
        return context
```

Extend context with `visual_reset_monotonic_ns`, optional finite confidence, and bounded upstream frame ID, with explicit backward-compatible defaults. At chord time, validate context age and require target/EE offsets to remain within the configured task ranges; drain pre-episode Y/F input so a stale keystroke cannot terminate the first step.

- [ ] **Step 4: Implement tokenized episode transitions**

```python
@dataclass(frozen=True)
class StepToken:
    episode_id: str
    step_id: int
    nonce: str

def finish_step(self, token):
    self._validate_token(token)
    if self._pending_terminal == 'success':
        return self._finish(OutcomeDecision(self.success_reward, True, 'human', True))
    if self._pending_terminal == 'failure':
        return self._finish(OutcomeDecision(self.failure_reward, True, 'human', False))
    return OutcomeDecision(self.step_reward, False, 'human', None)
```

`RealEpisodeCoordinator.intervention` exposes the Task 2 `AutomaticIntervention` callback used by Gym, while the same source's `last_frame` feeds `StartChord` only in `WAITING_FOR_RESET`. `outcome(observation)` polls the bounded Y/F reader after the valid successor exists and calls `finish_step()` for the currently active token; its observation parameter is retained as call-order evidence. Abort clears pending context/chord/terminal state and cannot emit an unfinished transition.

- [ ] **Step 5: Run state/contract regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_operator_control.py tests/test_g2_real_episode.py tests/test_g2_contract.py tests/test_g2_episode.py tests/test_g2_outcome.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- g2_local/operator_control.py g2_local/real_episode.py g2_local/contract.py tests/test_g2_operator_control.py tests/test_g2_real_episode.py
git commit --only -m "feat: add real episode operator state machine" -- g2_local/operator_control.py g2_local/real_episode.py g2_local/contract.py tests/test_g2_operator_control.py tests/test_g2_real_episode.py
```

### Task 4: Commissioned Live GDK Motion Assembly

**Files:**
- Create: `g2_local/motion_env.py`
- Create: `tests/test_g2_motion_env.py`
- Modify: `g2_local/gdk_backend.py`
- Modify: `g2_local/freshness.py`
- Modify: `g2_local/motion_backend.py`
- Modify: `g2_local/env.py`

**Interfaces:**
- Consumes: config, coordinator, clock client, freshness guard, GDK reader/port, MotionBackend.
- Produces: `create_motion_env(config, coordinator, *, cli_allow_motion, factories=None) -> G2LocalEnv`; `FreshnessLeaseGuard`.

- [ ] **Step 1: Write failing denial/ownership/lease tests**

```python
def test_readonly_config_never_constructs_command_port():
    calls = []
    with pytest.raises(PermissionError):
        create_motion_env(readonly_config(), coordinator(), cli_allow_motion=False,
                          factories=recording_factories(calls))
    assert 'command_port' not in calls

def test_motion_assembly_owns_one_gdk_session_and_explicit_limits():
    env = create_motion_env(commissioned_config(), coordinator(),
                            cli_allow_motion=True, factories=fake_factories())
    assert env.backend.enabled is True
    env.close()
    assert fake_session_release_count() == 1

def test_feedback_lease_expiry_stops_writer_without_new_gym_step():
    env, clock = commissioned_env()
    env.reset(options={'context': context()}); env.step(np.zeros(6))
    clock.advance_beyond_feedback_lease()
    assert wait_until(lambda: env.backend.stream.halt.is_set())

def test_configured_camera_rois_are_cropped_before_resize():
    env = commissioned_env(camera_rois={'left_wrist': (10,20,100,80),
                                        'right_aux': (30,40,120,90)})
    obs, _ = env.reset(options={'context': context()})
    assert obs['left_wrist'].shape == (128, 128, 3)
    assert env.last_crop_boxes == {'left_wrist': (10,20,100,80),
                                   'right_aux': (30,40,120,90)}
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_motion_env.py -q`

Expected: collection fails because `g2_local.motion_env` is missing.

- [ ] **Step 3: Make GDK motion intent explicit**

```python
class GdkReader:
    def __init__(self, adapter_root='/home/flyfuture/g2_hinge_assembly', timeout_s=2.,
                 *, allow_motion=False):
        if type(allow_motion) is not bool:
            raise ValueError('Explicit boolean GDK motion intent required')
        self.controller = G2Controller(gdk, self.robot, allow_motion=allow_motion)
```

Default remains false. One owner closes camera, TF, robot, controller, and GDK only after read/writer lifetimes end.

- [ ] **Step 4: Implement feedback lease and factory**

```python
class FreshnessLeaseGuard:
    def __init__(self, observation_guard, *, feedback_lease_s,
                 clock=time.monotonic):
        if not callable(observation_guard) or not callable(clock):
            raise ValueError('Freshness guard and monotonic clock are required')
        if not math.isfinite(feedback_lease_s) or feedback_lease_s <= 0:
            raise ValueError('Positive feedback lease required')
        self.observation_guard = observation_guard
        self.feedback_lease_s = feedback_lease_s
        self.clock = clock
        self.valid_until = None
        self.lock = threading.RLock()

    def accept(self, obs, info, after=None):
        with self.lock:
            self.valid_until = None
            accepted = self.observation_guard(obs, info, after)
            if accepted is True:
                self.valid_until = self.clock() + self.feedback_lease_s
            return accepted
    def __call__(self):
        with self.lock:
            return self.valid_until is not None and self.clock() <= self.valid_until

def create_motion_env(config, coordinator, *, cli_allow_motion, factories=None):
    if not (cli_allow_motion and config.motion_permitted):
        raise PermissionError('commissioned motion permission is required')
    factories = factories or MotionFactories.production()
    clock_client = factories.clock_client(config.commissioning.clock_socket,
                                          config.commissioning.expected_master)
    reader = factories.reader(adapter_root=config.motion.adapter_root,
                              timeout_s=config.motion.reader_timeout_s,
                              allow_motion=True)
    source = OwnedObservationSource(reader, clock_client)
    observation_guard = ObservationFreshnessGuard(clock_client, config.freshness)
    lease = FreshnessLeaseGuard(observation_guard,
                                feedback_lease_s=config.motion.command_timeout_s)
    port = factories.command_port(source.controller,
                     expected_mode=config.motion.control_mode,
                     allow_motion=True, freshness_guard=lease,
                     life_time_s=config.motion.command_lifetime_s)
    backend = MotionBackend(source, port, config=config.motion.limits,
                            observation_guard=lease.accept,
                            outcome=coordinator.outcome,
                            command_timeout=config.motion.command_timeout_s,
                            send_timeout=config.motion.send_timeout_s,
                            stop_timeout=config.motion.stop_timeout_s,
                            send_rate_hz=config.motion.send_rate_hz,
                            step_period=1.0/config.task.control_hz,
                            allow_motion=True)
    return G2LocalEnv(backend, max_steps=config.task.max_episode_steps,
                      image_size=config.observation.image_size,
                      camera_rois=config.observation.camera_rois,
                      intervention=coordinator.intervention)
```

Define `MotionFactories` as a frozen dataclass of three callables: `clock_client`, `reader`, and `command_port`. `MotionFactories.production()` imports concrete implementations lazily only after the permission check. Wrap the reader and clock client in `OwnedObservationSource`: it delegates `observe`, `last_info`, and `controller`, then closes the GDK reader before the clock client exactly once. On partial construction failure, close resources in reverse order and never create the command port before clock-client construction, GDK-reader construction, initial mode validation, and commissioning-evidence verification have succeeded. `G2LocalEnv` validates each configured ROI against the acquired frame, crops first, then resizes; an invalid/out-of-frame ROI aborts instead of silently falling back to the full image.

- [ ] **Step 5: Run motion/freshness regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_motion_env.py tests/test_g2_motion_backend.py tests/test_g2_command_stream.py tests/test_g2_freshness.py tests/test_g2_gdk_reader.py -q`

Expected: PASS without real GDK import/send.

- [ ] **Step 6: Commit**

```bash
git add -- g2_local/motion_env.py g2_local/gdk_backend.py g2_local/freshness.py g2_local/motion_backend.py g2_local/env.py tests/test_g2_motion_env.py
git commit --only -m "feat: assemble commissioned G2 motion environment" -- g2_local/motion_env.py g2_local/gdk_backend.py g2_local/freshness.py g2_local/motion_backend.py g2_local/env.py tests/test_g2_motion_env.py
```

### Task 5: Real Actor Runtime and Reliable Transition Uplink

**Files:**
- Create: `g2_local/real_actor.py`
- Create: `tests/test_g2_real_actor.py`
- Modify: `g2_local/contract.py`

**Interfaces:**
- Consumes: `LoadedTrainingConfig`, `RealEpisodeCoordinator`, `create_motion_env`, `create_policy`, `make_policy_obs`, and loopback gRPC.
- Produces: `ParameterEnvelope`; `TransitionIdentity`; `GrpcActorTransport.receive_latest_parameters() -> ParameterEnvelope | None`; `GrpcActorTransport.send_transition_batch(rows) -> None`; `GrpcActorTransport.assert_alive() -> None`; `RealActorRuntime.run() -> ActorRunSummary`; `RealActorRuntime.stop(reason: str) -> None`.

- [ ] **Step 1: Write failing identity, action-provenance, and backpressure tests**

```python
def test_actor_uses_real_factory_and_uploads_confirmed_action():
    rig = actor_rig()
    rig.runtime.run(max_completed_steps=1)
    assert rig.env_factory.calls == 1
    assert rig.transport.sent[0]['action'].tolist() == [0., .25, 0., 0., 0., 0.]
    info = rig.transport.sent[0]['complementary_info']
    assert info['policy_action'] != info['executed_action']
    assert info['transition_id'] == 'run-1/episode-1/0'
    assert info['synthetic'] is False

def test_actor_rejects_wrong_or_rolled_back_parameters_before_loading():
    rig = actor_rig(parameter_versions=(3, 2), parameter_run_id='run-1')
    rig.runtime.accept_latest_parameters()
    with pytest.raises(RuntimeError, match='rolled-back parameter version'):
        rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]

def test_same_version_heartbeat_keeps_transport_live_without_reloading():
    rig = actor_rig(parameter_messages=((3, 8), (3, 9)))
    rig.runtime.accept_latest_parameters()
    rig.runtime.accept_latest_parameters()
    assert rig.policy.loaded_versions == [3]
    assert rig.runtime.last_message_sequence == 9

def test_full_transition_queue_stops_motion_instead_of_dropping_data():
    rig = actor_rig(transport_blocked=True, transition_queue_capacity=1)
    with pytest.raises(TimeoutError, match='transition uplink backpressure'):
        rig.runtime.run(max_completed_steps=2)
    assert rig.env.backend.stop_calls == 1
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_actor.py -q`

Expected: collection fails because `g2_local.real_actor` does not exist.

- [ ] **Step 3: Add strict wire identities and version envelopes**

```python
@dataclass(frozen=True)
class TransitionIdentity:
    run_id: str
    episode_id: str
    step_id: int

    @property
    def value(self) -> str:
        return f'{self.run_id}/{self.episode_id}/{self.step_id}'

@dataclass(frozen=True)
class ParameterEnvelope:
    run_id: str
    config_hash: str
    version: int
    message_sequence: int
    actor_state: Mapping[str, torch.Tensor]

@dataclass(frozen=True)
class ActorRunSummary:
    transitions_sent: int
    episodes_completed: int
    interventions: int
    final_parameter_version: int
    stop_reason: str

def validate_parameter_envelope(envelope, *, run_id, config_hash,
                                current_version, current_sequence):
    if envelope.run_id != run_id or envelope.config_hash != config_hash:
        raise ValueError('parameter run/config identity mismatch')
    if envelope.message_sequence <= current_sequence:
        raise RuntimeError('non-increasing parameter message sequence')
    if envelope.version < current_version:
        raise RuntimeError('rolled-back parameter version')
    return envelope.version > current_version
```

Every real transition must contain `run_id`, `config_hash`, `transition_id`, `episode_id`, `step_id`, `actor_version`, context payload, gate summary, policy/human/selected/executed actions, reward source, success label, and `synthetic=false`. Extend `contract.transition()` to retain `selected_action` separately from the driver-confirmed `executed_action`. Reject unknown identity fields, nonfinite action/reward data, and tensors outside the established dual-RGB/state/action contract.

- [ ] **Step 4: Implement the bounded Real Actor loop**

```python
class RealActorRuntime:
    def run(self, *, max_completed_steps=None):
        env = self.env_factory(self.config, self.coordinator)
        completed = 0
        try:
            while not self.stop_event.is_set():
                self._require_transport_alive()
                self._accept_latest_parameters_at_step_boundary()
                if not self.coordinator.running:
                    self._wait_for_context_and_start_chord(env)
                    continue
                before = self.current_observation
                policy_action = self._infer(before)
                token = self.coordinator.begin_step()
                try:
                    after, reward, terminated, truncated, info = env.step(policy_action)
                except BaseException:
                    self.coordinator.abort_step(token)
                    raise
                row = self._build_confirmed_transition(before, after, reward,
                                                       terminated, truncated, info)
                self.uplink.put(row, timeout=self.config.runtime.queue_put_timeout_s)
                completed += 1
                self.current_observation = after
                if terminated or truncated:
                    self.coordinator.seal_episode()
                if max_completed_steps is not None and completed >= max_completed_steps:
                    return self.summary()
        except BaseException:
            self.stop('actor_failure')
            raise
        finally:
            env.close()
```

Implement `GrpcActorTransport` with bounded transition/parameter queues, RPC deadlines, one sender thread, and channel-failure propagation. A timed-out transition batch may be retried only with identical transition IDs; Learner deduplication supplies exactly-once replay mutation. Learner heartbeat envelopes may repeat the current policy version but must increase `message_sequence`; they refresh liveness without reloading weights. The Actor requires one valid policy before accepting a start chord and stops if no valid envelope arrives within `learner_silence_timeout_s`. A lost learner, expired transport deadline, evidence-write failure, conflicting Y/F, input fault, or full queue calls `stop()` before propagating the error. Never retry a failed `MotionBackend` instance and never load parameters during an in-flight Gym step.

- [ ] **Step 5: Run Actor and contract regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_actor.py tests/test_g2_operator_control.py tests/test_g2_real_episode.py tests/test_g2_contract.py tests/test_g2_gym.py tests/test_g2_motion_backend.py -q`

Expected: PASS; all tests use injected fakes and record zero hardware sends.

- [ ] **Step 6: Commit**

```bash
git add -- g2_local/real_actor.py g2_local/contract.py tests/test_g2_real_actor.py
git commit --only -m "feat: add formal G2 real actor runtime" -- g2_local/real_actor.py g2_local/contract.py tests/test_g2_real_actor.py
```

### Task 6: Real Learner, Dual Replay, and Atomic Resume

**Files:**
- Create: `g2_local/real_learner.py`
- Create: `tests/test_g2_real_learner.py`
- Modify: `g2_local/runtime.py`

**Interfaces:**
- Consumes: validated real transition rows from Task 5, `create_policy`, `ReplayBuffer`, and the existing loopback gRPC serialization helpers.
- Produces: `GrpcLearnerService` with transition RPC and parameter/heartbeat stream; `RealLearnerRuntime.ingest(rows) -> IngestResult`; `RealLearnerRuntime.update_once() -> Mapping[str, float] | None`; `save_checkpoint(path: Path) -> Path`; `load_checkpoint(path: Path, *, expected_run_id: str, expected_config_hash: str) -> LearnerSnapshot`.

- [ ] **Step 1: Write failing deduplication, dual-replay, optimizer, and resume tests**

```python
def test_duplicate_transition_does_not_mutate_replay_or_updates():
    learner, row = learner_and_transition(intervention=True)
    assert learner.ingest([row]).accepted == 1
    before = learner.snapshot_counts()
    assert learner.ingest([row]).duplicates == 1
    assert learner.snapshot_counts() == before

def test_critic_trains_executed_action_and_human_pool_is_separate():
    learner, row = learner_and_transition(intervention=True,
                                          policy_action=[1,0,0,0,0,0],
                                          executed_action=[0,0,0,0,0,0])
    learner.ingest([row])
    online, human = learner.sample_for_test()
    assert online['action'].eq(0).all()
    assert human['action'].eq(0).all()

def test_empty_human_pool_skips_expert_and_extra_bc_step():
    learner = ready_learner(human_rows=0, online_rows=8)
    before = optimizer_steps(learner, ('expert', 'actor'))
    learner.update_once()
    after = optimizer_steps(learner, ('expert', 'actor'))
    assert after['expert'] == before['expert']
    assert after['actor'] == before['actor'] + 1

def test_resume_validates_identity_and_returns_to_waiting_for_reset(tmp_path):
    learner = ready_learner(human_rows=8, online_rows=8)
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    restored = load_checkpoint(checkpoint, expected_run_id='run-1',
                               expected_config_hash=learner.config_hash)
    assert restored.software_state_restored is True
    assert restored.physical_episode_state == 'WAITING_FOR_RESET'
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_learner.py -q`

Expected: collection fails because `g2_local.real_learner` does not exist.

- [ ] **Step 3: Implement validated ingestion and configurable dual replay**

```python
@dataclass(frozen=True)
class IngestResult:
    accepted: int
    duplicates: int

@dataclass(frozen=True)
class LearnerSnapshot:
    run_id: str
    config_hash: str
    version: int
    message_sequence: int
    software_state_restored: bool
    physical_episode_state: str = 'WAITING_FOR_RESET'

def ingest(self, rows):
    accepted = duplicates = 0
    for row in rows:
        identity = validate_real_transition(row, self.run_id, self.config_hash)
        if identity.value in self.seen_transition_ids:
            duplicates += 1
            continue
        training_row = _buffer_transition(row)
        self.online_replay.add(**training_row)
        if row['complementary_info']['is_intervention']:
            self.human_replay.add(**training_row)
        self.seen_transition_ids.add(identity.value)
        self.records.append(row)
        accepted += 1
    return IngestResult(accepted, duplicates)
```

Create replay capacities, minimum replay size, human sampling ratio, batch size, UTD, learning rates, target-update interval, publish interval, and checkpoint interval only from `OptimizationConfig`; reject values that make a batch impossible. Do not mutate replay before the entire row validates.

- [ ] **Step 4: Implement SiLRI update order and monotonic publication**

```python
def update_once(self):
    if len(self.online_replay) < self.config.min_online_transitions:
        return None
    online = self.online_replay.sample(self.config.online_batch_size)
    has_human = len(self.human_replay) >= self.config.human_batch_size > 0
    if has_human:
        human = self.human_replay.sample(self.config.human_batch_size)
        data = concatenate_batch_transitions(online, human)
    else:
        data = online
    names = ['critic', 'actor', 'lagrange']
    if has_human:
        names.extend(['expert', 'actor_bc'])
    metrics = train_batch(self.policy, self.optimizers, data, tuple(names))
    if (self.version + 1) % self.config.target_update_interval == 0:
        self.policy.update_target_networks()
    self.version += 1
    if self.version % self.config.publish_interval == 0:
        self.message_sequence += 1
        self.publish(ParameterEnvelope(self.run_id, self.config_hash, self.version,
                                       self.message_sequence,
                                       cpu_actor_state(self.policy.actor)))
    return metrics
```

The configured UTD loop calls `update_once()` exactly `utd_ratio` times per accepted interaction budget. `GrpcLearnerService` validates complete batches before calling `ingest()` and emits a heartbeat at `parameter_heartbeat_s` when no new policy is due; heartbeat `message_sequence` increases while `version` and Actor state remain unchanged. New-policy versions are strictly increasing; a failed stream or publish stops the learner rather than pretending the Actor received the version.

- [ ] **Step 5: Implement atomic, identity-bound checkpoint/resume**

```python
def save_checkpoint(self, path):
    payload = self._snapshot_payload()
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                          dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
        return path
    finally:
        temporary.unlink(missing_ok=True)

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
```

The payload includes schema, run/config identity, immutable manifest digest, policy and target networks, every optimizer, version, both replay contents and insertion indices, transition-ID set, CPU/CUDA/NumPy/Python RNG state, counters, and provenance records. Resume accepts only an owned regular local file, validates schema/hash/action/camera contract before mutation, and never restores an active physical episode.

- [ ] **Step 6: Run Learner and existing SiLRI regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_learner.py tests/test_silri_runtime.py tests/test_g2_processes.py -q`

Expected: PASS; Critic/Actor/lagrange and expert/BC behavior is covered, including empty-human-pool gating.

- [ ] **Step 7: Commit**

```bash
git add -- g2_local/real_learner.py g2_local/runtime.py tests/test_g2_real_learner.py
git commit --only -m "feat: add resumable SiLRI real learner" -- g2_local/real_learner.py g2_local/runtime.py tests/test_g2_real_learner.py
```

### Task 7: Train/Eval CLI, Lifecycle Evidence, and Operator Runbook

**Files:**
- Create: `g2_local/real_train.py`
- Create: `tests/test_g2_real_train.py`
- Create: `docs/g2-real-training.md`
- Modify: `run_g2_python.sh`

**Interfaces:**
- Consumes: Tasks 1–6 and existing `clock_monitor` socket; CLI roles `learner`, `actor`, and `eval`.
- Produces: `python -m g2_local.real_train ROLE --run-id ID --config PATH --output PATH [--allow-motion]`; `RunEvidenceWriter`; deterministic process exit codes.

- [ ] **Step 1: Write failing preflight, role, eval, and shutdown tests**

```python
def test_preflight_writes_manifest_before_importing_gdk(tmp_path, monkeypatch):
    imported = []
    monkeypatch.setattr(real_train, 'import_gdk_runtime', lambda: imported.append(True))
    code = real_train.main(['actor', '--run-id', 'run-1',
                            '--config', str(readonly_config_path()),
                            '--output', str(tmp_path / 'run')])
    assert code == 2
    assert imported == []
    assert (tmp_path / 'run/run_manifest.json').is_file()

def test_eval_loads_frozen_checkpoint_without_optimizer_or_uplink(tmp_path):
    rig = eval_cli_rig(tmp_path)
    rig.run_one_episode()
    assert rig.optimizer_factory.calls == 0
    assert rig.transition_sender.calls == 0
    assert rig.summary['intervention_free_success_rate_denominator'] == 0

def test_actor_disconnect_stops_before_transport_close():
    rig = lifecycle_rig(disconnect='learner')
    with pytest.raises(ConnectionError): rig.run()
    assert rig.events[:3] == ['command_stop', 'env_close', 'transport_close']
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_train.py -q`

Expected: collection fails because `g2_local.real_train` does not exist.

- [ ] **Step 3: Implement strict role-specific CLI and preflight**

```python
def main(argv=None):
    args = parser().parse_args(argv)
    loaded = load_training_config(args.config,
                                  cli_allow_motion=bool(args.allow_motion))
    manifest = loaded.write_manifest(args.output, run_id=args.run_id, role=args.role)
    evidence = RunEvidenceWriter(args.output, manifest, role=args.role)
    if args.role in ('actor', 'eval') and not loaded.motion_permitted:
        evidence.finish('motion_not_permitted')
        return 2
    try:
        return run_role(args, loaded, evidence)
    except KeyboardInterrupt:
        evidence.finish('operator_interrupt')
        return 130
    except BaseException as error:
        evidence.finish('failed', error=bounded_error(error))
        return 1
```

Require a new output directory, a bounded ASCII `--run-id` shared by Actor and Learner, owned regular config/checkpoint/evidence files, loopback learner address, explicit clock socket/master, and CUDA availability for the checked-in training profile. The two manifest copies must have the same run ID and config hash; their role and output path may differ. Resume/eval must use the checkpoint's run ID and config hash, while each process still gets a fresh output directory. `--allow-motion` is legal only for `actor`/`eval`; `learner` cannot import GDK or open HID.

For Actor/eval, `run_role` binds `env_factory = functools.partial(create_motion_env, cli_allow_motion=args.allow_motion)` only after preflight has validated `loaded.motion_permitted`; neither the config flag nor CLI flag alone reaches `GdkCommandPort`. For Learner, `run_role` constructs only `GrpcLearnerService` and `RealLearnerRuntime`.

- [ ] **Step 4: Implement train/eval behavior and ordered shutdown**

```python
def stop_actor(runtime, transport, evidence, reason):
    stop_error = None
    try:
        runtime.stop(reason)
    except BaseException as error:
        stop_error = error
    finally:
        runtime.close_environment()
        transport.close()
        evidence.finish('stop_unconfirmed' if stop_error else reason)
    if stop_error is not None:
        raise stop_error
```

`train` uploads transitions and consumes new parameters. `eval` loads one exact checkpoint, creates no optimizer/uplink, keeps intervention safety available, and excludes any assisted episode from the no-intervention success denominator. Both modes emit bounded JSONL events and per-episode summaries for success, unassisted success, intervention ratio, episode length, action clipping, freshness rejects, stop reason, Actor latency, control period, policy version, and context offsets. Raw RGB logging remains off unless the config explicitly enables it.

- [ ] **Step 5: Write the exact operator runbook**

Document, in order:

1. Start/verify hardware E-stop and establish an exclusion zone.
2. Start `clock_monitor` in its own foreground terminal and inspect healthy snapshots.
3. Start Learner with read-only configuration and wait for its ready event.
4. Start Actor first without `--allow-motion` to prove fail-closed preflight.
5. After all field gates have separately passed, copy the read-only template, fill evidence hashes/workspace/scales, and add CLI `--allow-motion`.
6. For each episode, upstream vision resets and atomically writes a new context; operator presses/releases both SpaceMouse buttons; movement intervenes; fresh neutral returns control; Y/F ends the episode.
7. Stop Actor first, verify stop evidence, then stop Learner and clock monitor.

Include explicit warnings that software completion does not approve motion, Y/F does not move to a reset pose, output directories are never reused, and hardware emergency stop remains independent.

- [ ] **Step 6: Run CLI/lifecycle regressions**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_train.py tests/test_g2_real_actor.py tests/test_g2_real_learner.py tests/test_g2_processes.py -q`

Expected: PASS with no GDK import or command send in tests.

- [ ] **Step 7: Commit**

```bash
git add -- g2_local/real_train.py tests/test_g2_real_train.py docs/g2-real-training.md run_g2_python.sh
git commit --only -m "feat: add real SiLRI train and eval entrypoints" -- g2_local/real_train.py tests/test_g2_real_train.py docs/g2-real-training.md run_g2_python.sh
```

### Task 8: End-to-End Isolation Proof, Full Verification, and Progress Handoff

**Files:**
- Create: `tests/test_g2_real_training_integration.py`
- Modify: `docs/g2-adaptation-status.md`
- Modify: `/home/flyfuture/桌面/hil-rRL/项目进展_CN.md`

**Interfaces:**
- Consumes: formal Actor, Learner, configuration, state machine, motion assembly, and CLI from Tasks 1–7.
- Produces: a bounded two-process integration proof, repository-wide verification evidence, updated project ledger, and the exact remaining field-gate sequence.

- [ ] **Step 1: Write the failing formal-path integration test**

```python
def test_formal_actor_learner_path_updates_and_resumes_without_synthetic_backend(tmp_path):
    rig = formal_runtime_rig(tmp_path, episodes=2, intervention_steps={1, 3})
    summary = rig.run()
    assert summary.actor.transitions_sent == 4
    assert summary.learner.online_replay == 4
    assert summary.learner.human_replay == 2
    assert summary.learner.version > 0
    assert summary.command_port.executed_actions == rig.confirmed_actions
    assert all(row['complementary_info']['synthetic'] is False
               for row in summary.records)
    assert SyntheticBackend not in rig.constructed_types
    resumed = rig.resume()
    assert resumed.actor_state == 'WAITING_FOR_RESET'

def test_fault_matrix_is_fail_closed_and_never_reuses_backend(tmp_path):
    for fault in ('clock_expired', 'hid_unplug', 'learner_disconnect',
                  'command_timeout', 'successor_stale', 'evidence_write'):
        rig = formal_runtime_rig(tmp_path / fault, fault=fault)
        with pytest.raises((RuntimeError, TimeoutError, ConnectionError, OSError)):
            rig.run()
        assert rig.stop_attempted is True
        assert rig.backend_constructions == 1
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_training_integration.py -q`

Expected: FAIL until Tasks 1–7 expose the complete formal runtime fixture and lifecycle.

- [ ] **Step 3: Complete the isolated two-process harness**

Use real serialization, queues, replay, policy update code, state machine, Gym adapter, and CLI lifecycle. Inject only the GDK reader, command port, clock snapshots, HID, keyboard, and upstream context source. The fake command port must acknowledge a clipped action and count stop calls; it must never import `agibot_gdk` or access `/dev/hidraw*`.

Run: `PYTHONPATH=lerobot/src .venv/bin/python -m pytest tests/test_g2_real_training_integration.py -q`

Expected: PASS for normal train/resume and every fail-closed fault row.

- [ ] **Step 4: Run focused safety and algorithm suites**

Run:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest \
  tests/test_g2_training_config.py tests/test_g2_operator_control.py \
  tests/test_g2_real_episode.py tests/test_g2_motion_env.py \
  tests/test_g2_real_actor.py tests/test_g2_real_learner.py \
  tests/test_g2_real_train.py tests/test_g2_real_training_integration.py \
  tests/test_g2_command_stream.py tests/test_g2_motion_backend.py \
  tests/test_g2_freshness.py tests/test_silri_runtime.py -q
```

Expected: PASS; no test requires root, GDK, a SpaceMouse, or robot motion.

- [ ] **Step 5: Run repository-wide verification and static checks**

Run:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m pytest -q
.venv/bin/python -m py_compile g2_local/*.py
git diff --check
rg -n "SyntheticBackend|allow_motion=True|agibot_gdk" \
  g2_local/motion_env.py g2_local/real_*.py
```

Expected: full suite PASS with only documented pre-existing warnings; `SyntheticBackend` absent from formal runtime modules; `allow_motion=True` appears only in the commissioned Actor assembly reached after configuration and CLI gates; `agibot_gdk` is imported only inside the Actor-side GDK factory.

- [ ] **Step 6: Update status and the mandatory top-level ledger**

Record each task commit, focused/full test counts, review findings, and unchanged motion authorization. State explicitly that the software runtime is complete but continuous real training remains blocked on:

1. three qualified real-Actor/freshness sessions and six-threshold approval;
2. empty-gripper XYZ/RPY direction/scale/workspace commissioning;
3. software stop, lease-expiry, and hardware E-stop timing/distance evidence;
4. one low-speed reset/chord/Y/F/intervention handoff episode;
5. small-batch checkpoint/resume and fixed-checkpoint eval.

- [ ] **Step 7: Request independent final review and fix all Critical/Important findings**

Review the complete diff against `docs/superpowers/specs/2026-09-22-real-silri-training-runtime-design.md`, with explicit attention to the five items in **Review Focus**. Re-run the focused and full suites after every safety-significant correction. Do not provide a motion command until the final review reports zero open Critical and Important findings.

- [ ] **Step 8: Commit integration and documentation**

```bash
git add -- tests/test_g2_real_training_integration.py docs/g2-adaptation-status.md
git commit --only -m "test: verify formal G2 SiLRI training runtime" -- tests/test_g2_real_training_integration.py docs/g2-adaptation-status.md
```

`/home/flyfuture/桌面/hil-rRL/项目进展_CN.md` is outside the repository: update it after the commit so it can record the final commit hash, but do not copy it into the repository or include it in `git add`.
