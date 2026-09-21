from types import SimpleNamespace as NS
import json
import sys
import time
import pytest


def test_ptp_command_cannot_adjust_clocks_and_is_bounded():
    from g2_local.clock_probe import ptp_command
    cmd = ptp_command(45, '/var/run/g2-clock-test')
    assert '--free_running=1' in cmd and '-S' in cmd and '-s' in cmd
    assert '--utc_offset=37' in cmd
    assert cmd[:3] == ['sudo', '-n', '/usr/bin/timeout']
    assert '45s' in cmd and '--kill-after=3s' in cmd
    assert 'phc2sys' not in ' '.join(cmd)
    with pytest.raises(ValueError): ptp_command(99999, '/var/run/g2-clock-test')


def test_log_parser_latches_master_change_and_rejects_late_pipe_delivery():
    from g2_local.clock_probe import PtpEvidence
    p = PtpEvidence('044052.fffe.000010')
    p.feed('ptp4l[10.000]: selected best master clock 044052.fffe.000010', 10_000_000_000)
    p.feed('ptp4l[11.000]: master offset 55000000000 s0 freq +0 path delay 40000', 11_010_000_000)
    assert len(p.samples) == 1
    p.feed('ptp4l[12.000]: selected best master clock other', 12_000_000_000)
    p.feed('ptp4l[13.000]: selected best master clock 044052.fffe.000010', 13_000_000_000)
    assert p.errors
    q = PtpEvidence('044052.fffe.000010')
    q.feed('ptp4l[10.000]: selected best master clock 044052.fffe.000010', 10_000_000_000)
    q.feed('ptp4l[11.000]: master offset 55000000000 s0 freq +0 path delay 40000', 16_000_000_000)
    assert q.errors and not q.samples


def test_time_properties_must_confirm_scale_and_no_leap():
    from g2_local.clock_probe import check_properties
    raw = ('TIME_PROPERTIES_DATA_SET\n currentUtcOffset 37\n '
           'currentUtcOffsetValid 0\n leap61 0\n leap59 0\n ptpTimescale 1\n')
    assert check_properties(raw)['currentUtcOffsetValid'] == 0
    for bad in (raw.replace('Offset 37', 'Offset 36'), raw.replace('leap61 0', 'leap61 1'),
                raw.replace('ptpTimescale 1', 'ptpTimescale 0'), ''):
        with pytest.raises(ValueError): check_properties(bad)


def transform(x, q=(0, 0, 0, 1)):
    return NS(translation=NS(x=x, y=.2, z=.3), rotation=NS(**dict(zip('xyzw', q))))


def test_gdk_sample_retains_both_tf_directions_without_changing_robot():
    from g2_local.clock_probe import sample_gdk
    class TF:
        def lookup_transform_latest(self, target, source, return_timestamp):
            assert return_timestamp is True
            return (transform(.1 if target == 'arm_l_end_link' else -.7), 1234)
    reader = NS(streams={'left_wrist': 1, 'right_aux': 2},
                camera=NS(get_latest_image=lambda stream, timeout: NS(timestamp_ns=1000+stream)),
                robot=NS(get_joint_states=lambda: {'timestamp': 1200}),
                controller=NS(read_end_effector_pose=lambda name: NS(
                    position_m=(.1, .2, .3), orientation_xyzw=(0, 0, 0, -1))),
                gdk=NS(Clock=NS(now_ns=lambda: 1300)))
    r = sample_gdk(reader, TF())
    assert r['timestamps'] == dict(left_wrist=1001, right_aux=1002, joint=1200, tf=1234)
    assert r['tf_position_error_m'] == 0
    assert r['tf_rotation_error_rad'] == 0  # q and -q equivalent
    assert r['tf_queries'][0]['pose'][0] == -.7
    assert r['tf_queries'][1]['pose'][0] == .1
    assert r['mono_ns'] >= r['start_mono_ns']
    json.dumps(r, allow_nan=False)


def evidence():
    from test_g2_clock_mapping import gdk_rows
    events = [dict(kind='ptp', mono_ns=100_000_000_000,
                   raw='ptp4l[100.000]: selected best master clock 044052.fffe.000010')]
    for i in range(10):
        events.append(dict(kind='ptp', mono_ns=(100+2*i)*10**9,
                           raw=f'ptp4l[{100+2*i}.000]: master offset {55_000_000_000+i*20_000} '
                               's0 freq +10000 path delay 40000'))
    events.extend(gdk_rows())
    for t in (102, 108, 114, 118):
        events.append(dict(kind='properties', mono_ns=t*10**9, returncode=0,
                           raw='TIME_PROPERTIES_DATA_SET\ncurrentUtcOffset 37\n'
                               'currentUtcOffsetValid 0\nleap61 0\nleap59 0\nptpTimescale 1\n'))
    return events


def test_complete_evidence_produces_retrospective_mapping_not_live_permission():
    from g2_local.clock_probe import summarize
    r = summarize(evidence(), master='044052.fffe.000010', session='test')
    assert r['scale'] == 'raw_ptp'
    assert r['properties']['currentUtcOffsetValid'] == 0
    assert not r['valid_for_live_use'] and not r['motion_authorized']
    assert r['mapping']['expires_ns'] == 120_500_000_000


@pytest.mark.parametrize('bad', ['missing_properties', 'scale_change', 'collection_error',
                                  'master_change', 'log_corruption'])
def test_complete_evidence_fails_closed(bad):
    from g2_local.clock_probe import summarize
    events = evidence()
    if bad == 'missing_properties':
        events = [e for e in events if e['kind'] != 'properties']
    if bad == 'scale_change': events[-1]['raw'] = events[-1]['raw'].replace('Offset 37', 'Offset 36')
    if bad == 'collection_error': events.append(dict(kind='error', error='lost camera'))
    if bad == 'master_change':
        events.append(dict(kind='ptp', mono_ns=119_000_000_000,
                           raw='ptp4l[119.000]: selected best master clock other'))
    if bad == 'log_corruption':
        events.append(dict(kind='ptp', mono_ns=119_000_000_000,
                           raw='ptp4l[119.000]: master offset nan s0 freq +0 path delay 10'))
    with pytest.raises(ValueError):
        summarize(events, master='044052.fffe.000010', session='test')


def test_supervisor_ends_blocked_reader_process_and_records_rejection(tmp_path):
    from g2_local.clock_probe import run_supervised
    output = tmp_path/'run'
    started = time.monotonic()
    code = run_supervised([sys.executable, '-c', 'import time; time.sleep(10)'],
                          timeout_s=.1, output=output)
    assert code == 124 and time.monotonic()-started < 3
    report = json.loads((output/'supervisor.json').read_text())
    assert report['status'] == 'rejected'
    assert report['motion_authorized'] is False


def test_tf_preflight_waits_for_cache_before_returning():
    from g2_local.clock_probe import wait_for_tf
    class TF:
        calls = 0
        def lookup_transform_latest(self, target, source, return_timestamp):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError('Failed to lookup latest transform')
            return transform(.1), 1234
    tf = TF()
    result = wait_for_tf(tf, timeout_s=.1, retry_s=.001)
    assert tf.calls == 4  # two failures, then both required directions
    assert [item['timestamp_ns'] for item in result] == [1234, 1234]


def test_tf_preflight_times_out_with_specific_error():
    from g2_local.clock_probe import wait_for_tf
    class TF:
        def lookup_transform_latest(self, *_args):
            raise RuntimeError('Failed to lookup latest transform')
    with pytest.raises(TimeoutError, match='TF cache'):
        wait_for_tf(TF(), timeout_s=.005, retry_s=.001)


def test_supervisor_records_nonzero_worker_as_rejected(tmp_path):
    from g2_local.clock_probe import run_supervised
    output = tmp_path/'run'
    code = run_supervised([sys.executable, '-c', 'raise SystemExit(7)'],
                          timeout_s=1, output=output)
    assert code == 7
    report = json.loads((output/'supervisor.json').read_text())
    assert report['status'] == 'rejected'
    assert 'code 7' in report['reason']


def test_supervisor_rejects_existing_evidence_directory_before_worker_starts(tmp_path):
    from g2_local.clock_probe import run_supervised
    marker = tmp_path/'worker-started'
    with pytest.raises(FileExistsError, match='output'):
        run_supervised([sys.executable, '-c',
                        f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                       timeout_s=1, output=tmp_path)
    assert not marker.exists()
