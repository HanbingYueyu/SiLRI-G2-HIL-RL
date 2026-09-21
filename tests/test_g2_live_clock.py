from dataclasses import asdict, FrozenInstanceError, replace
import json

import pytest


MASTER = '044052.fffe.000010'
ORIGIN_NS = 1_000_000_000_000_000


def wall(mono_ns):
    return ORIGIN_NS + mono_ns


def properties(*, utc_offset=37, valid=0, leap61=0, leap59=0, timescale=1):
    return (
        'TIME_PROPERTIES_DATA_SET\n'
        f' currentUtcOffset {utc_offset}\n'
        f' currentUtcOffsetValid {valid}\n'
        f' leap61 {leap61}\n'
        f' leap59 {leap59}\n'
        f' ptpTimescale {timescale}\n'
    )


def offset_line(second, offset_ns, delay_ns=40_000):
    return (f'ptp4l[{second:.3f}]: master offset {offset_ns} '
            f's0 freq +10000 path delay {delay_ns}')


def feed_master_properties_and_8_samples(window):
    window.feed_ptp(
        f'ptp4l[1.000]: selected best master clock {MASTER}',
        1_010_000_000,
        wall(1_010_000_000),
    )
    window.feed_properties(properties(), 1_100_000_000)
    for index, second in enumerate(range(2, 18, 2)):
        mono_ns = second * 1_000_000_000
        window.feed_ptp(
            offset_line(second, 55_000_000_000 + index * 20_000),
            mono_ns + 10_000_000,
            wall(mono_ns + 10_000_000),
        )


def healthy_window():
    from g2_local.live_clock import ClockWindow

    window = ClockWindow(MASTER, 'boot', 'run')
    feed_master_properties_and_8_samples(window)
    return window


def test_window_warms_then_publishes_short_raw_ptp_lease():
    from g2_local.live_clock import ClockWindow

    window = ClockWindow(MASTER, 'boot', 'run')
    early = window.snapshot(100_000_000, wall(100_000_000))
    assert early.healthy is False
    assert early.reason == 'warming_up'

    feed_master_properties_and_8_samples(window)
    snap = window.snapshot(16_100_000_000, wall(16_100_000_000))

    assert snap.healthy is True
    assert snap.reason == 'ok'
    assert snap.scale == 'raw_ptp'
    assert snap.schema == 1
    assert snap.sequence == 2
    assert snap.boot_id == 'boot'
    assert snap.session_id == 'run'
    assert snap.expected_master == MASTER
    assert snap.actual_master == MASTER
    assert snap.utc_offset_s == 37
    assert snap.utc_offset_valid == 0
    assert snap.leap61 == 0
    assert snap.leap59 == 0
    assert snap.ptp_timescale == 1
    assert snap.reference_mono_ns == 16_000_000_000
    assert snap.offset_at_reference_ns == pytest.approx(18_000_140_000, abs=1)
    assert snap.drift_ppm == pytest.approx(10)
    assert snap.residual_ns == pytest.approx(0, abs=1)
    assert snap.path_delay_ns == 40_000
    assert snap.empirical_error_ns >= 2_040_000
    assert snap.wall_minus_mono_ns == ORIGIN_NS
    assert snap.created_mono_ns == 16_100_000_000
    assert snap.last_sample_mono_ns == 16_000_000_000
    assert snap.valid_until_ns == snap.last_sample_mono_ns + 2_500_000_000


def test_repeating_snapshot_does_not_extend_old_lease():
    window = healthy_window()
    first = window.snapshot(16_100_000_000, wall(16_100_000_000))
    later = window.snapshot(17_000_000_000, wall(17_000_000_000))

    assert later.sequence == first.sequence + 1
    assert later.created_mono_ns == 17_000_000_000
    assert later.valid_until_ns == first.valid_until_ns


def test_new_sample_refits_and_moves_only_the_ptp_derived_lease():
    window = healthy_window()
    first = window.snapshot(16_100_000_000, wall(16_100_000_000))
    window.feed_ptp(offset_line(18, 55_000_160_000),
                    18_010_000_000, wall(18_010_000_000))
    second = window.snapshot(18_100_000_000, wall(18_100_000_000))

    assert second.reference_mono_ns == 18_000_000_000
    assert second.valid_until_ns == 20_500_000_000
    assert second.valid_until_ns > first.valid_until_ns
    assert second.offset_at_reference_ns == pytest.approx(18_000_160_000, abs=1)


def inject_fault(window, fault):
    now_ns = 16_200_000_000
    wall_ns = wall(now_ns)
    if fault == 'master':
        window.feed_ptp('ptp4l[16.200]: selected best master clock other',
                        now_ns, wall_ns)
    elif fault == 'gap':
        now_ns = 21_000_000_000
        wall_ns = wall(now_ns)
        window.feed_ptp(offset_line(21, 55_000_190_000), now_ns, wall_ns)
    elif fault == 'drift':
        now_ns = 18_000_000_000
        wall_ns = wall(now_ns)
        window.feed_ptp(offset_line(18, 55_010_000_000), now_ns, wall_ns)
    elif fault == 'residual':
        now_ns = 18_000_000_000
        wall_ns = wall(now_ns)
        window.feed_ptp(offset_line(18, 55_002_160_000), now_ns, wall_ns)
    elif fault == 'delay':
        now_ns = 18_000_000_000
        wall_ns = wall(now_ns)
        window.feed_ptp(offset_line(18, 55_000_160_000, -1), now_ns, wall_ns)
    elif fault == 'properties':
        window.feed_properties(properties(utc_offset=36), now_ns)
    elif fault == 'wall_jump':
        wall_ns += 1_000_001
    elif fault == 'malformed':
        window.feed_ptp('ptp4l[16.200]: master offset nan s0 freq +0 path delay 10',
                        now_ns, wall_ns)
    else:
        raise AssertionError(f'unknown fault {fault}')
    return now_ns, wall_ns


@pytest.mark.parametrize('fault', ['master', 'gap', 'drift', 'residual',
                                    'delay', 'properties', 'wall_jump', 'malformed'])
def test_any_clock_fault_latches_unhealthy(fault):
    window = healthy_window()
    now_ns, wall_ns = inject_fault(window, fault)
    failed = window.snapshot(now_ns, wall_ns)
    assert failed.healthy is False
    assert failed.reason != 'warming_up'

    # Valid evidence cannot clear a structural fault; recovery reconstructs a window.
    window.feed_properties(properties(), now_ns + 1)
    assert window.snapshot(now_ns + 2, wall_ns + 2).healthy is False
    assert window.snapshot(now_ns + 2, wall_ns + 2).reason == failed.reason


def test_late_delivery_and_explicit_ptp_fault_are_latched():
    late = healthy_window()
    late.feed_ptp(offset_line(18, 55_000_160_000),
                  18_500_000_001, wall(18_500_000_001))
    assert late.snapshot(18_500_000_001, wall(18_500_000_001)).reason == 'ptp_delivery_delay'

    faulty = healthy_window()
    faulty.feed_ptp('ptp4l[16.200]: port 1: SLAVE to LISTENING on FAULT_DETECTED',
                    16_200_000_000, wall(16_200_000_000))
    assert faulty.snapshot(16_200_000_000, wall(16_200_000_000)).reason == 'ptp_fault'


@pytest.mark.parametrize('raw', [
    properties().replace('currentUtcOffset 37', 'currentUtcOffset 37.5'),
    properties().replace('ptpTimescale 1', 'ptpTimescale 1 trailing'),
    properties() + 'leap61 0\n',
    properties() + 'leap61 1\n',
])
def test_malformed_or_duplicate_properties_latch_and_cannot_recover(raw):
    window = healthy_window()
    window.feed_properties(raw, 16_200_000_000)

    failed = window.snapshot(16_200_000_001, wall(16_200_000_001))
    assert failed.healthy is False
    assert failed.reason == 'properties_invalid'

    window.feed_properties(properties(), 16_200_000_002)
    recovered = window.snapshot(16_200_000_003, wall(16_200_000_003))
    assert recovered.healthy is False
    assert recovered.reason == 'properties_invalid'


@pytest.mark.parametrize('raw', [
    offset_line(18, '9' * 400),
    'ptp4l[' + '9' * 400 + '.000]: master offset 55000160000 '
    's0 freq +10000 path delay 40000',
    'ptp4l[18.000]: master offset 55000160000 s0 freq +'
    + '9' * 400 + ' path delay 40000',
    offset_line(18, 55_000_160_000, '9' * 400),
    'ptp4l[18.0000000001]: master offset 55000160000 '
    's0 freq +10000 path delay 40000',
])
def test_out_of_range_or_inexact_report_numbers_latch_fail_closed(raw):
    window = healthy_window()
    window.feed_ptp(raw, 18_000_000_000, wall(18_000_000_000))

    failed = window.snapshot(18_000_000_001, wall(18_000_000_001))
    assert failed.healthy is False
    assert failed.reason == 'ptp_report_invalid'
    json.dumps(asdict(failed), allow_nan=False)

    window.feed_ptp(offset_line(18, 55_000_160_000),
                    18_000_000_002, wall(18_000_000_002))
    recovered = window.snapshot(18_000_000_003, wall(18_000_000_003))
    assert recovered.healthy is False
    assert recovered.reason == 'ptp_report_invalid'


@pytest.mark.parametrize('failure', [
    'exception', 'nonfinite_float', 'nonfinite_time', 'out_of_range',
])
def test_fit_failures_and_nonfinite_outputs_latch_fail_closed(monkeypatch, failure):
    import g2_local.live_clock as live_clock
    from g2_local.clock_mapping import MAX_REPORT_INTEGER

    window = healthy_window()
    original_fit = live_clock.fit_mapping
    if failure == 'exception':
        def broken_fit(*_args, **_kwargs):
            raise OverflowError('numeric conversion overflow')
    elif failure == 'nonfinite_float':
        def broken_fit(*args, **kwargs):
            return replace(original_fit(*args, **kwargs),
                           offset_at_last_ns=float('nan'))
    elif failure == 'nonfinite_time':
        def broken_fit(*args, **kwargs):
            return replace(original_fit(*args, **kwargs),
                           last_ns=float('inf'))
    else:
        def broken_fit(*args, **kwargs):
            return replace(original_fit(*args, **kwargs),
                           offset_at_last_ns=float(MAX_REPORT_INTEGER) * 2)
    monkeypatch.setattr(live_clock, 'fit_mapping', broken_fit)

    window.feed_ptp(offset_line(18, 55_000_160_000),
                    18_000_000_000, wall(18_000_000_000))
    failed = window.snapshot(18_000_000_001, wall(18_000_000_001))
    assert failed.healthy is False
    assert failed.reason == 'mapping_invalid'
    json.dumps(asdict(failed), allow_nan=False)


def test_startup_residual_spike_restarts_warmup_and_allows_a_fresh_mapping():
    """Catch a one-off startup fit failure being latched permanently."""
    from g2_local.live_clock import ClockWindow

    window = ClockWindow(MASTER, 'boot', 'run')
    window.feed_ptp(
        f'ptp4l[1.000]: selected best master clock {MASTER}',
        1_010_000_000,
        wall(1_010_000_000),
    )
    window.feed_properties(properties(), 1_100_000_000)
    for index, second in enumerate(range(2, 16, 2)):
        mono_ns = second * 1_000_000_000
        window.feed_ptp(
            offset_line(second, 55_000_000_000 + index * 20_000),
            mono_ns + 10_000_000,
            wall(mono_ns + 10_000_000),
        )

    # The eighth candidate is valid PTP syntax/delivery evidence, but its
    # offset makes the first fit exceed the fixed residual limit.
    window.feed_ptp(
        offset_line(16, 55_002_140_000),
        16_010_000_000,
        wall(16_010_000_000),
    )
    warming = window.snapshot(16_100_000_000, wall(16_100_000_000))
    assert warming.healthy is False
    assert warming.reason == 'warming_up'

    # Eight clean reports after the rejected startup candidate form a new
    # window; none of the earlier candidate samples may contaminate it.
    for index, second in enumerate(range(18, 34, 2)):
        mono_ns = second * 1_000_000_000
        window.feed_ptp(
            offset_line(second, 55_100_000_000 + index * 20_000),
            mono_ns + 10_000_000,
            wall(mono_ns + 10_000_000),
        )
    recovered = window.snapshot(32_100_000_000, wall(32_100_000_000))
    assert recovered.healthy is True
    assert recovered.reason == 'ok'
    assert recovered.reference_mono_ns == 32_000_000_000


def test_mapping_expiry_cannot_exceed_signed_64_bit_report_range():
    from g2_local.clock_mapping import MAX_REPORT_INTEGER
    from g2_local.live_clock import ClockWindow

    window = ClockWindow(MASTER, 'boot', 'run')
    last_ns = MAX_REPORT_INTEGER - 1_000_000_000
    first_ns = last_ns - 14_000_000_000
    window.feed_ptp(
        f'ptp4l[1.000]: selected best master clock {MASTER}',
        first_ns - 1,
        first_ns - 1,
    )
    window.feed_properties(properties(), first_ns)
    for index in range(8):
        mono_ns = first_ns + index * 2_000_000_000
        seconds, fraction = divmod(mono_ns, 1_000_000_000)
        raw = (f'ptp4l[{seconds}.{fraction:09d}]: master offset '
               f'{55_000_000_000 + index * 20_000} '
               's0 freq +10000 path delay 40000')
        window.feed_ptp(raw, mono_ns, mono_ns)

    snap = window.snapshot(last_ns, last_ns)
    assert snap.healthy is False
    assert snap.reason == 'mapping_invalid'
    json.dumps(asdict(snap), allow_nan=False)


def test_missing_or_expired_evidence_is_unhealthy_without_extending_the_lease():
    from g2_local.live_clock import ClockWindow

    no_properties = ClockWindow(MASTER, 'boot', 'run')
    no_properties.feed_ptp(
        f'ptp4l[1.000]: selected best master clock {MASTER}',
        1_000_000_000,
        wall(1_000_000_000),
    )
    for index, second in enumerate(range(2, 18, 2)):
        mono_ns = second * 1_000_000_000
        no_properties.feed_ptp(offset_line(second, 55_000_000_000 + index * 20_000),
                               mono_ns, wall(mono_ns))
    assert no_properties.snapshot(16_100_000_000, wall(16_100_000_000)).healthy is False

    window = healthy_window()
    expired = window.snapshot(18_500_000_001, wall(18_500_000_001))
    assert expired.healthy is False
    assert expired.reason == 'lease_expired'
    assert expired.valid_until_ns == 18_500_000_000


def test_snapshot_is_frozen():
    snap = healthy_window().snapshot(16_100_000_000, wall(16_100_000_000))
    with pytest.raises(FrozenInstanceError):
        snap.healthy = False


@pytest.mark.parametrize('method,args', [
    ('feed_ptp', ('unrelated log', True, 1)),
    ('feed_ptp', ('unrelated log', 1, 1.0)),
    ('feed_properties', (properties(), False)),
    ('snapshot', (1.0, 1)),
    ('snapshot', (1, False)),
])
def test_nanosecond_arguments_require_exact_ints_before_state_changes(method, args):
    window = healthy_window()
    first = window.snapshot(16_100_000_000, wall(16_100_000_000))
    with pytest.raises(ValueError, match='Integer nanosecond evidence required'):
        getattr(window, method)(*args)
    later = window.snapshot(16_200_000_000, wall(16_200_000_000))
    assert later.healthy is True
    assert later.sequence == first.sequence + 1


def test_constructor_requires_nonempty_string_identities():
    from g2_local.live_clock import ClockWindow

    for args in (('', 'boot', 'run'), (MASTER, '', 'run'), (MASTER, 'boot', '')):
        with pytest.raises(ValueError):
            ClockWindow(*args)
