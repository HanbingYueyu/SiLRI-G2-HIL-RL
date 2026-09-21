"""Diagnostic clock mapping only. Never a motion/freshness authorization.

PTP software timestamp offsets include linuxptp's UTC correction. Compare
both raw-PTP and UTC sensor hypotheses; never estimate offset from sensor age.
Error allowances below are empirical diagnostics, not certified bounds.
"""
from dataclasses import dataclass
from decimal import Decimal
import math
import re
import numpy as np

PTP_LINE = re.compile(r'ptp4l\[(\d+\.\d+)\]: master offset\s+(-?\d+) '
                      r's\d+ freq\s+[+-]?\d+ path delay\s+(-?\d+)')
SOURCES = ('left_wrist', 'right_aux', 'joint', 'tf')


def parse_ptp(line, master):
    match = PTP_LINE.search(line)
    if not match:
        return None
    return dict(mono_ns=int(Decimal(match[1])*10**9),
                offset_ns=int(match[2]), delay_ns=int(match[3]), master=master)


def integer(value):
    if type(value) is not int:
        raise ValueError('Integer nanosecond evidence required')
    return value


@dataclass(frozen=True)
class DiagnosticMapping:
    session: str
    master: str
    utc_offset_s: int
    start_ns: int
    last_ns: int
    expires_ns: int
    offset_at_last_ns: float
    drift_ppm: float
    residual_ns: float
    empirical_error_ns: float

    def offset_at(self, mono_ns, *, session, scale):
        integer(mono_ns)
        if session != self.session or not self.start_ns <= mono_ns <= self.expires_ns:
            raise ValueError('Mapping expired, predates evidence, or belongs to another session')
        if scale not in ('raw_ptp', 'utc'):
            raise ValueError('Unknown time scale')
        dt = mono_ns - self.last_ns
        offset = self.offset_at_last_ns + dt*self.drift_ppm/1e6
        if scale == 'raw_ptp':
            offset -= self.utc_offset_s*10**9
        # An explicit 100 ppm allowance grows only for extrapolation.
        error = self.empirical_error_ns + max(0, dt)*100/1e6
        return offset, error


def fit_mapping(samples, *, master, utc_offset_s, session):
    if not session or not master or type(utc_offset_s) is not int or not 0 <= utc_offset_s <= 100:
        raise ValueError('Explicit session, master and UTC correction required')
    if len(samples) < 8:
        raise ValueError('Need at least eight PTP measurements')
    times, offsets, delays = [], [], []
    for s in samples:
        if s['master'] != master:
            raise ValueError('PTP master changed or missing')
        times.append(integer(s['mono_ns']))
        offsets.append(integer(s['offset_ns']))
        delays.append(integer(s['delay_ns']))
    if any(not 0 < b-a <= 4_000_000_000 for a, b in zip(times, times[1:])):
        raise ValueError('PTP gap, duplicate, or reversed time')
    if times[-1]-times[0] < 10_000_000_000 or times[0] <= 0:
        raise ValueError('Insufficient PTP time span')
    if min(delays) < 0 or max(delays) > 1_000_000:
        raise ValueError('Invalid or excessive path delay')
    x = np.asarray([(t-times[-1])/1e9 for t in times])
    y = np.asarray([v-offsets[-1] for v in offsets], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.max(np.abs(y-(slope*x+intercept))))
    if not math.isfinite(slope) or abs(slope/1000) > 100 or residual > 1_000_000:
        raise ValueError('PTP jump, excessive drift or residual')
    return DiagnosticMapping(session, master, utc_offset_s, times[0], times[-1],
                             times[-1]+2_500_000_000, float(offsets[-1]+intercept),
                             float(slope/1000), residual,
                             2_000_000 + residual + max(delays))


def associate(mapping, rows):
    """Test clock-domain consistency, not prove sensor acquisition semantics.

Diagnostic limits: <=500 ms age, <=100 ms pair skew, <=5 mm/0.02 rad
TF-vs-motion agreement. No production safety thresholds are chosen here.
"""
    overlap = [r for r in rows if mapping.start_ns <= r['mono_ns'] <= mapping.last_ns]
    if len(overlap) < 8:
        raise ValueError('Need eight simultaneous GDK/PTP samples')
    baseline = None
    previous = None
    max_skew = 0
    for row in rows:
        for key in ('mono_ns', 'wall_ns', 'start_mono_ns', 'start_wall_ns'):
            integer(row[key])
        if not 0 <= row['mono_ns']-row['start_mono_ns'] <= 500_000_000:
            raise ValueError('Excessively slow GDK sample')
        for wall, mono in ((row['wall_ns'], row['mono_ns']),
                           (row['start_wall_ns'], row['start_mono_ns'])):
            origin = wall-mono
            if baseline is None:
                baseline = origin
            if abs(origin-baseline) > 1_000_000:
                raise ValueError('Local wall clock jump')
        stamps = row['timestamps']
        if set(stamps) != set(SOURCES) or any(integer(v) <= 0 for v in stamps.values()):
            raise ValueError('Missing/invalid source timestamp')
        if previous and row['mono_ns']-previous['mono_ns'] > 5_000_000_000:
            raise ValueError('GDK acquisition gap')
        if previous and (row['mono_ns'] <= previous['mono_ns'] or any(
                stamps[k] <= previous['timestamps'][k] for k in SOURCES)):
            raise ValueError('Frozen or reversed source timestamp')
        previous = row
        skew = abs(stamps['left_wrist']-stamps['right_aux'])
        max_skew = max(max_skew, skew)
        if skew > 100_000_000:
            raise ValueError('Excessive camera skew')
        for key, limit in (('tf_position_error_m', .005), ('tf_rotation_error_rad', .02)):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= limit:
                raise ValueError('TF direction/pose not consistent with motion status')
    candidates = []
    for scale in ('raw_ptp', 'utc'):
        ages = {k: [] for k in SOURCES}
        errors = []
        for row in overlap:
            offset, error = mapping.offset_at(row['mono_ns'], session=mapping.session, scale=scale)
            errors.append(error)
            for key in SOURCES:
                # Subtract integers before floats to retain nanosecond precision.
                ages[key].append(row['wall_ns']-row['timestamps'][key]-offset)
        if all(age >= -err and age+err <= 500_000_000
               for values in ages.values() for age, err in zip(values, errors)):
            candidates.append(dict(scale=scale, ages_ms={key: dict(
                min=min(values)/1e6, max=max(values)/1e6,
                lower_min=min(a-e for a, e in zip(values, errors))/1e6,
                upper_max=max(a+e for a, e in zip(values, errors))/1e6)
                for key, values in ages.items()}))
    if len(candidates) != 1:
        raise ValueError('No unique source time-scale association')
    return dict(candidates[0], status='diagnostic_consistent', motion_authorized=False,
                source_clock_identity_proven=False, sample_count=len(overlap),
                max_camera_skew_ms=max_skew/1e6,
                uncertainty_kind='empirical allowance; not a guaranteed physical bound')
