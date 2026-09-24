"""Isolated formal Actor/Learner proof. All physical interfaces are test doubles."""

from concurrent import futures
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from g2_local.config import HingeInsertTaskConfig
from g2_local.clock_mapping import PTP_LEASE_NS
from g2_local.contract import EpisodeContext
from g2_local.env import SyntheticBackend
from g2_local.freshness import FreshnessLimits
from g2_local.live_clock import ClockSnapshot
from g2_local.motion_env import MotionFactories, create_motion_env
from g2_local.real_actor import GrpcActorTransport, RealActorRuntime
from g2_local.real_episode import RealEpisodeCoordinator
from g2_local.real_learner import GrpcLearnerService, RealLearnerRuntime, load_checkpoint
from g2_local.spacemouse import AutomaticIntervention
from g2_local.training_config import InterventionConfig, OptimizationConfig, RuntimeConfig
from lerobot.transport import services_pb2_grpc


RUN_ID = 'formal-isolated'
CONFIG_HASH = 'formal-config-hash'
ORIGIN = 1_700_000_000_000_000_000
MASTER = 'isolated-master'


def _config():
    task = HingeInsertTaskConfig(control_hz=20., max_episode_steps=2)
    motion = SimpleNamespace(
        limits=task.motion_config(workspace_low=(.399, .2, .7),
                                  workspace_high=(.4, .5, 1.)),
        # CPU optimization is synchronous at the gRPC boundary in this fixture.
        # A long fake lease keeps the functional proof independent of CPU speed.
        control_mode=1, command_timeout_s=10., send_timeout_s=.5,
        stop_timeout_s=.5, reader_timeout_s=.1, command_lifetime_s=.1,
        send_rate_hz=25., adapter_root=Path('/isolated/fake-adapter'))
    runtime = RuntimeConfig(17, 'cpu', '127.0.0.1', 1, 4, 2., 5., 5., .02, 10., .001)
    optimization = OptimizationConfig(8, 4, 1, 1, 1, 1, 1e-4, 1e-4,
                                      1e-4, 1e-4, 2, 1, 1)
    return SimpleNamespace(task=task, motion=motion,
                           observation=SimpleNamespace(image_size=128, camera_rois={}),
                           intervention=InterventionConfig((-2, -1, -3), 0, 1,
                               .12, .08, .0001, .25),
                           runtime=runtime, optimization=optimization,
                           freshness=FreshnessLimits(.1, .1, .1, .01, .05, .01),
                           commissioning=FakeCommissioning(),
                           requested_motion=True, motion_permitted=True,
                           config_hash=CONFIG_HASH)


class FakeCommissioning:
    clock_socket = Path('/isolated/clock.sock')
    expected_master = MASTER

    def verify_files_and_hashes(self, freshness):
        return True


class FakeClock:
    def __init__(self, fault):
        self.sequence = 0
        self.fault = fault
        self.expired_reads = 0
        self.closed = False

    def read(self):
        now = time.monotonic_ns()
        self.sequence += 1
        self.expired_reads += int(self.fault == 'clock_expired')
        sample = now - 3_000_000_000 if self.fault == 'clock_expired' else now
        return ClockSnapshot(
            schema=1, sequence=self.sequence, healthy=True, reason='ok',
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            session_id='isolated-session', expected_master=MASTER, actual_master=MASTER,
            scale='raw_ptp', utc_offset_s=37, utc_offset_valid=0, leap61=0,
            leap59=0, ptp_timescale=1, reference_mono_ns=sample,
            offset_at_reference_ns=0., drift_ppm=0., residual_ns=0.,
            path_delay_ns=0, empirical_error_ns=100_000.,
            wall_minus_mono_ns=ORIGIN, created_mono_ns=sample,
            last_sample_mono_ns=sample, valid_until_ns=sample + PTP_LEASE_NS)

    def close(self):
        self.closed = True


class FakeReader:
    def __init__(self, fault):
        self.fault = fault
        self.state = np.array([.3995, .35, .85, 0., 0., 0., 1.], dtype=np.float64)
        self.last_info = {}
        self.observations = 0
        self.sent = False
        self.stale_successor_served = False
        self.closed = False
        self.controller = SimpleNamespace(
            checked_arm_state=lambda: None,
            motion_status_summary=lambda: {'control_mode': 1, 'error_code': 0})

    def observe(self):
        self.observations += 1
        end = time.monotonic_ns()
        start = end - 1_000_000
        source = ORIGIN + end - 2_000_000
        if self.fault == 'successor_stale' and self.sent:
            self.stale_successor_served = True
            source -= 1_000_000_000
        stamps = dict.fromkeys(('left_wrist', 'right_aux', 'joint', 'tf'), source)
        pose = self.state.astype(float).tolist()
        self.last_info = dict(
            source_timestamp_ns=stamps,
            camera_timestamp_ns={k: stamps[k] for k in ('left_wrist', 'right_aux')},
            tf_queries=[dict(target='base_link', source='arm_l_end_link',
                             timestamp_ns=source, pose=pose.copy()),
                        dict(target='arm_l_end_link', source='base_link',
                             timestamp_ns=source, pose=pose.copy())],
            tf_position_error_m=0., tf_rotation_error_rad=0., motion_pose=pose,
            read_start_monotonic_ns=start, read_end_monotonic_ns=end,
            read_start_wall_ns=ORIGIN + start, read_end_wall_ns=ORIGIN + end,
            read_start_sdk_clock_ns=source - 1_000_000,
            read_end_sdk_clock_ns=source, sdk_clock_ns=source,
            received_monotonic_ns=end, state_received_monotonic_ns=start,
            read_duration_s=.001)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :, 0] = self.observations % 255
        return {'state': self.state.copy(), 'left_wrist': image,
                'right_aux': image.copy()}

    def close(self):
        self.closed = True


class FakeCommandPort:
    def __init__(self, reader, guard, scale, fault):
        self.reader = reader
        self.guard = guard
        self.scale = np.array(scale)
        self.fault = fault
        self.executed_actions = []
        self.stop_calls = 0
        self._failed = False
        self.timeout_triggered = False
        self._last_sequence = 0
        self.stream = None

    def send(self, target):
        if self.guard() is not True:
            raise RuntimeError('feedback lease expired')
        if self.fault == 'command_timeout' and not self._failed:
            self._failed = True
            self.timeout_triggered = True
            time.sleep(.6)
            raise TimeoutError('isolated command acknowledgement timeout')
        target_pose = np.array((*target.position_m, *target.orientation_xyzw))
        sequence = self.stream.sequence
        if self._last_sequence == sequence:
            return
        previous = self.reader.state.astype(float)
        rotation = (Rotation.from_quat(target_pose[3:]) *
                    Rotation.from_quat(previous[3:]).inv()).as_rotvec()
        action = tuple(np.concatenate(((target_pose[:3] - previous[:3]) / self.scale[:3],
                                       rotation / self.scale[3:])))
        self.executed_actions.append(action)
        self._last_sequence = sequence
        self.reader.state = target_pose
        self.reader.sent = True

    def stop(self):
        self.stop_calls += 1


class FakeHID:
    """Emit timed frames only; AutomaticIntervention and chord logic stay real."""

    def __init__(self, intervention_steps, fault):
        self.intervention_steps = intervention_steps
        self.fault_kind = fault
        self._start_phase = 0
        self._step = 0
        self.coordinator = None
        self.unplug_triggered = False

    def poll(self):
        now = time.monotonic()
        if self.coordinator.running:
            if self.fault_kind == 'hid_unplug':
                self.unplug_triggered = True
                raise OSError('isolated HID unplug')
            moving = self._step in self.intervention_steps
            self._step += 1
            axes = (0., -1. if moving else 0., 0., 0., 0., 0.)
            buttons, pressed = (False, False), ()
        else:
            buttons = (True, True) if self._start_phase % 2 == 0 else (False, False)
            pressed = (0, 1) if buttons == (True, True) else ()
            axes = (0.,) * 6
            self._start_phase += 1
        return SimpleNamespace(axes=axes, buttons=buttons, pressed=pressed,
                               ready=True, axis_times=(now - .0001, now - .0001))


class FakeKeys:
    def read_available(self, *, limit):
        return []


def _learner_process(connection, checkpoint):
    torch.set_num_threads(2)
    config = _config()
    learner = RealLearnerRuntime(config=config, run_id=RUN_ID,
                                 checkpoint_path=checkpoint)
    service = GrpcLearnerService(learner)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    service.publish(learner.publish_parameters())
    connection.send(port)
    try:
        if connection.poll(30):
            connection.recv()
        learner.save_checkpoint(checkpoint)
        connection.send((learner.snapshot_counts(), learner.version,
                         len(learner.records), _hardware_imports()))
    finally:
        server.stop(grace=.2).wait()
        connection.close()


@dataclass
class FormalRig:
    path: Path
    episodes: int
    intervention_steps: set[int]
    fault: str | None

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.path / 'checkpoint.pt'
        self.constructed_types = []
        self.backend_constructions = 0
        self.stop_attempted = False
        self.command_port = None
        self.ports = []
        self.readers = []
        self.clocks = []
        self.confirmed_actions = []
        self.hardware_imports = []
        self.actor_runtime = None
        self.transport = None
        self.hid = None
        self.freshness_guard = None
        self.evidence_write_triggered = False
        self.disconnect_triggered = False

    def run(self):
        torch.set_num_threads(2)
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=_learner_process, args=(child, self.checkpoint))
        process.start()
        child.close()
        runtime = None
        try:
            assert parent.poll(20), 'Learner failed to start'
            port = parent.recv()
            config = _config()
            hid = FakeHID(self.intervention_steps, self.fault)
            self.hid = hid
            intervention = AutomaticIntervention(hid, config.intervention)
            coordinator = RealEpisodeCoordinator(
                intervention, FakeKeys(), config.task, context_max_age_s=5.)
            hid.coordinator = coordinator
            contexts = iter(EpisodeContext(
                f'episode-{number}', (0., 0., 0.), 'isolated-upstream', 'isolated-grasp',
                visual_reset_monotonic_ns=time.monotonic_ns(),
                upstream_frame_id=f'frame-{number}') for number in range(self.episodes))

            def env_factory(config, coordinator):
                self.backend_constructions += 1
                reader = FakeReader(self.fault)
                clock = FakeClock(self.fault)
                self.readers.append(reader)
                self.clocks.append(clock)

                def make_port(controller, *, expected_mode, allow_motion,
                              freshness_guard, life_time_s):
                    assert controller is reader.controller
                    assert expected_mode == 1 and allow_motion is True
                    self.freshness_guard = freshness_guard
                    self.command_port = FakeCommandPort(reader, freshness_guard,
                                                         config.motion.limits.action_scale,
                                                         self.fault)
                    self.ports.append(self.command_port)
                    return self.command_port

                def make_clock(socket, master):
                    assert (socket, master) == (FakeCommissioning.clock_socket, MASTER)
                    return clock

                def make_reader(**kwargs):
                    assert kwargs == {'adapter_root': Path('/isolated/fake-adapter'),
                                      'timeout_s': .1, 'allow_motion': True}
                    return reader

                factories = MotionFactories(make_clock, make_reader, make_port)
                env = create_motion_env(config, coordinator, cli_allow_motion=True,
                                        factories=factories)
                self.command_port.stream = env.backend.stream
                self.constructed_types.append(type(env.backend))
                return env

            transport = GrpcActorTransport(f'127.0.0.1:{port}', queue_capacity=4,
                                           timeout_s=5.)
            self.transport = transport
            def telemetry(kind, **fields):
                if self.fault == 'evidence_write' and kind == 'step':
                    self.evidence_write_triggered = True
                    raise OSError('isolated evidence write failure')

            runtime = RealActorRuntime(
                config=config, run_id=RUN_ID, config_hash=CONFIG_HASH,
                coordinator=coordinator,
                context_source=SimpleNamespace(read_new=lambda: next(contexts, None)),
                transport=transport, env_factory=env_factory, telemetry=telemetry)
            self.actor_runtime = runtime
            if self.fault == 'learner_disconnect':
                send_transition_batch = transport.send_transition_batch
                def disconnect(rows):
                    self.disconnect_triggered = True
                    transport.channel.close()
                    # Keep the production queue, serialization, sender thread,
                    # and gRPC error propagation in this fault path.
                    return send_transition_batch(rows)
                transport.send_transition_batch = disconnect
            actor = runtime.run(max_completed_steps=self.episodes * 2)
        finally:
            if runtime is not None:
                self.actor_runtime = runtime
                self.stop_attempted = runtime.stop_event.is_set() or any(
                    port.stop_calls for port in self.ports)
            if process.is_alive():
                parent.send('finish')
                if parent.poll(20):
                    counts, version, record_count, self.hardware_imports = parent.recv()
                    self.learner_result = SimpleNamespace(
                        online_replay=counts['online'], human_replay=counts['human'],
                        version=version, record_count=record_count)
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            parent.close()
            self.hardware_imports.extend(_hardware_imports())
        restored = self.resume().learner
        self.confirmed_actions = [tuple(row['complementary_info']['executed_action'])
                                  for row in restored.records]
        self.command_port = SimpleNamespace(executed_actions=[action
            for port in self.ports for action in port.executed_actions])
        return SimpleNamespace(actor=actor, learner=self.learner_result,
                               records=restored.records)

    def resume(self):
        restored = load_checkpoint(self.checkpoint, expected_run_id=RUN_ID,
                                   expected_config_hash=CONFIG_HASH)
        return SimpleNamespace(actor_state=restored.physical_episode_state,
                               learner=restored)


def formal_runtime_rig(path, *, episodes=1, intervention_steps=(), fault=None):
    return FormalRig(path, episodes, set(intervention_steps), fault)


def _hardware_imports():
    return sorted(name for name in sys.modules if name == 'agibot_gdk' or
                  name.startswith('agibot_gdk.') or name == 'hid' or
                  name.startswith('hid.'))


def test_formal_actor_learner_path_updates_and_resumes_without_synthetic_backend(tmp_path):
    rig = formal_runtime_rig(tmp_path, episodes=2, intervention_steps={1, 3})
    summary = rig.run()
    assert summary.actor.transitions_sent == 4
    assert summary.actor.episodes_completed == 2
    assert summary.actor.interventions == 2
    assert summary.learner.online_replay == 4
    assert summary.learner.human_replay == 2
    assert summary.learner.version > 0
    assert summary.records and len(rig.command_port.executed_actions) == 4
    assert np.allclose(rig.command_port.executed_actions, rig.confirmed_actions,
                       atol=1e-4)
    assert all(row['complementary_info']['synthetic'] is False
               for row in summary.records)
    assert [row['complementary_info']['transition_id'] for row in summary.records] == [
        'formal-isolated/episode-0/0', 'formal-isolated/episode-0/1',
        'formal-isolated/episode-1/0', 'formal-isolated/episode-1/1']
    assert any(tuple(row['complementary_info']['selected_action']) !=
               tuple(row['complementary_info']['executed_action'])
               for row in summary.records if row['complementary_info']['is_intervention'])
    assert SyntheticBackend not in rig.constructed_types
    assert rig.hardware_imports == []
    assert all(reader.closed for reader in rig.readers)
    assert all(clock.closed for clock in rig.clocks)
    resumed = rig.resume()
    assert resumed.actor_state == 'WAITING_FOR_RESET'
    assert len(resumed.learner.records) == 4


@pytest.mark.parametrize('fault', ('clock_expired', 'hid_unplug', 'learner_disconnect',
                                   'command_timeout', 'successor_stale', 'evidence_write'))
def test_fault_matrix_is_fail_closed_and_never_reuses_backend(tmp_path, fault):
    rig = formal_runtime_rig(tmp_path / fault, fault=fault)
    with pytest.raises((RuntimeError, TimeoutError, ConnectionError, OSError, ValueError)) as raised:
        rig.run()
    assert rig.stop_attempted is True
    assert rig.backend_constructions == 1
    if fault not in ('clock_expired', 'hid_unplug'):
        assert rig.ports[0].stop_calls > 0
    assert rig.hardware_imports == []
    assert all(reader.closed for reader in rig.readers)
    assert all(clock.closed for clock in rig.clocks)
    assert rig.actor_runtime is not None
    assert rig.actor_runtime.transitions_sent == 0
    assert rig.learner_result.online_replay == 0
    assert rig.learner_result.human_replay == 0

    error_chain = []
    error = raised.value
    while error is not None:
        error_chain.append(f'{type(error).__name__}: {error}')
        error = error.__cause__ or error.__context__
    error_text = '\n'.join(error_chain).lower()
    if fault == 'clock_expired':
        assert sum(clock.expired_reads for clock in rig.clocks) > 0
        assert rig.freshness_guard.last_decision.code == 'mapping_expired'
        expected_commands = 0
    elif fault == 'hid_unplug':
        assert rig.hid.unplug_triggered is True
        assert 'isolated hid unplug' in error_text
        expected_commands = 0
    elif fault == 'learner_disconnect':
        assert rig.disconnect_triggered is True
        assert rig.transport.error is not None
        assert 'closed' in error_text or 'unavailable' in error_text
        expected_commands = 1
    elif fault == 'command_timeout':
        assert rig.ports[0].timeout_triggered is True
        assert 'timeout' in error_text or 'deadline' in error_text
        expected_commands = 0
    elif fault == 'successor_stale':
        assert rig.readers[0].stale_successor_served is True
        assert rig.freshness_guard.last_decision.code == 'camera_stale:left_wrist'
        expected_commands = 1
    else:
        assert rig.evidence_write_triggered is True
        assert 'isolated evidence write failure' in error_text
        expected_commands = 1
    assert sum(len(port.executed_actions) for port in rig.ports) == expected_commands
