"""Pure rolling PTP mapping state and immutable short-lived snapshots.

This module observes clock evidence only.  It does not start processes, adjust
clocks, contact GDK, or authorize motion.
"""
from collections import deque
from dataclasses import dataclass
import math
import re
import threading

from .clock_mapping import (
    MAX_DRIFT_PPM,
    MAX_PATH_DELAY_NS,
    MAX_PTP_DELIVERY_DELAY_NS,
    MAX_PTP_DELIVERY_LEAD_NS,
    MAX_PTP_GAP_NS,
    MAX_REPORT_INTEGER,
    MAX_RESIDUAL_NS,
    MAX_WALL_JUMP_NS,
    MIN_PTP_SAMPLES,
    MIN_PTP_SPAN_NS,
    PTP_LEASE_NS,
    UTC_OFFSET_S,
    TransientMappingFitError,
    fit_mapping,
    integer,
    parse_ptp,
    parse_time_properties,
)


_MASTER_LINE = re.compile(r'\bselected best master clock (\S+)')
_FAULT_MARKERS = ('FAULTY', 'UNCALIBRATED to LISTENING',
                  'SLAVE to LISTENING', 'clockcheck:', 'timed out')
_WINDOW_CAPACITY = 64


@dataclass(frozen=True)
class ClockSnapshot:
    schema: int
    sequence: int
    healthy: bool
    reason: str
    boot_id: str
    session_id: str
    expected_master: str
    actual_master: str
    scale: str
    utc_offset_s: int
    utc_offset_valid: int
    leap61: int
    leap59: int
    ptp_timescale: int
    reference_mono_ns: int
    offset_at_reference_ns: float
    drift_ppm: float
    residual_ns: float
    path_delay_ns: int
    empirical_error_ns: float
    wall_minus_mono_ns: int
    created_mono_ns: int
    last_sample_mono_ns: int
    valid_until_ns: int


class ClockWindow:
    """Accumulate one fail-closed PTP session into rolling snapshots."""

    def __init__(self, expected_master, boot_id, session_id):
        for value in (expected_master, boot_id, session_id):
            if type(value) is not str or not value:
                raise ValueError('Non-empty clock identities required')
        self.expected_master = expected_master
        self.boot_id = boot_id
        self.session_id = session_id
        self._lock = threading.RLock()
        self._samples = deque(maxlen=_WINDOW_CAPACITY)
        self._actual_master = ''
        self._properties = None
        self._properties_mono_ns = 0
        self._mapping = None
        self._wall_origin_ns = None
        self._fault = None
        self._sequence = 0
        self._last_snapshot_mono_ns = None
        self._last_snapshot = None
        self._last_ptp_mono_ns = None

    @staticmethod
    def _validate_ns(*values):
        for value in values:
            integer(value)
            if value < 0:
                raise ValueError('Nanosecond evidence cannot be negative')

    def _latch(self, reason):
        if self._fault is None:
            self._fault = reason

    def _check_wall(self, mono_ns, wall_ns):
        origin = wall_ns - mono_ns
        if self._wall_origin_ns is None:
            self._wall_origin_ns = origin
        elif abs(origin - self._wall_origin_ns) > MAX_WALL_JUMP_NS:
            self._latch('wall_clock_jump')

    def _mapping_is_valid(self, mapping):
        integer_fields = (mapping.start_ns, mapping.last_ns, mapping.expires_ns)
        return (
            all(type(value) is int for value in integer_fields) and
            0 < mapping.start_ns <= mapping.last_ns and
            mapping.expires_ns == mapping.last_ns + PTP_LEASE_NS and
            mapping.expires_ns <= MAX_REPORT_INTEGER and
            mapping.master == self.expected_master and
            mapping.session == self.session_id and
            mapping.utc_offset_s == UTC_OFFSET_S and
            all(math.isfinite(value) for value in (
                mapping.offset_at_last_ns,
                mapping.drift_ppm,
                mapping.residual_ns,
                mapping.empirical_error_ns,
            )) and
            abs(mapping.offset_at_last_ns) <= MAX_REPORT_INTEGER and
            abs(mapping.drift_ppm) <= MAX_DRIFT_PPM and
            0 <= mapping.residual_ns <= MAX_RESIDUAL_NS and
            0 <= mapping.empirical_error_ns <= MAX_REPORT_INTEGER
        )

    def feed_ptp(self, line, received_mono_ns, wall_ns):
        if type(line) is not str:
            raise ValueError('PTP line must be text')
        self._validate_ns(received_mono_ns, wall_ns)
        with self._lock:
            self._check_wall(received_mono_ns, wall_ns)
            if self._fault is not None:
                return

            selected = _MASTER_LINE.search(line)
            if selected:
                master = selected.group(1)
                if master != self.expected_master:
                    self._latch('master_mismatch')
                    return
                if self._actual_master and master != self._actual_master:
                    self._latch('master_changed')
                    return
                self._actual_master = master

            if any(marker in line for marker in _FAULT_MARKERS):
                self._latch('ptp_fault')
                return

            try:
                sample = parse_ptp(line, self._actual_master or None)
            except Exception:
                self._latch('ptp_report_invalid')
                return
            if 'master offset' in line and sample is None:
                self._latch('ptp_report_invalid')
                return
            if sample is None:
                return
            if sample['master'] != self.expected_master:
                self._latch('master_missing')
                return

            delivery_ns = received_mono_ns - sample['mono_ns']
            if not -MAX_PTP_DELIVERY_LEAD_NS <= delivery_ns <= MAX_PTP_DELIVERY_DELAY_NS:
                self._latch('ptp_delivery_delay')
                return
            if not 0 <= sample['delay_ns'] <= MAX_PATH_DELAY_NS:
                self._latch('path_delay')
                return
            if self._last_ptp_mono_ns is not None:
                interval_ns = sample['mono_ns'] - self._last_ptp_mono_ns
                if not 0 < interval_ns <= MAX_PTP_GAP_NS:
                    self._latch('ptp_sample_interval')
                    return
            self._last_ptp_mono_ns = sample['mono_ns']

            candidate = list(self._samples)
            candidate.append(sample)
            mapping = None
            if (len(candidate) >= MIN_PTP_SAMPLES and
                    candidate[-1]['mono_ns'] - candidate[0]['mono_ns'] >= MIN_PTP_SPAN_NS):
                try:
                    mapping = fit_mapping(candidate, master=self.expected_master,
                                          utc_offset_s=UTC_OFFSET_S,
                                          session=self.session_id)
                    if not self._mapping_is_valid(mapping):
                        raise ValueError('Invalid PTP mapping')
                except TransientMappingFitError:
                    # Before the first mapping is published, one otherwise
                    # well-formed report can be a transient startup residual
                    # outlier.  It cannot seed the next fit, but there is no
                    # lease yet to revoke: begin a wholly fresh candidate
                    # window.  After publication, quarantine only this
                    # transient outlier and retain the existing mapping and
                    # its unchanged lease.  A later clean report may refit;
                    # consumers reject snapshots during any intervening lease
                    # expiry.  Repeated anomalies therefore remain unhealthy
                    # through the normal lease-expiry path.  Do not renew a
                    # lease from the rejected sample.
                    if self._mapping is None:
                        self._samples.clear()
                        return
                    return
                except Exception:
                    self._latch('mapping_invalid')
                    return
            self._samples.append(sample)
            if mapping is not None:
                self._mapping = mapping

    def feed_properties(self, raw, mono_ns):
        if type(raw) is not str:
            raise ValueError('PTP time properties must be text')
        self._validate_ns(mono_ns)
        with self._lock:
            if self._fault is not None:
                return
            try:
                properties = parse_time_properties(raw)
            except Exception:
                self._latch('properties_invalid')
                return
            if self._properties is not None and properties != self._properties:
                self._latch('properties_changed')
                return
            if mono_ns < self._properties_mono_ns:
                self._latch('properties_reversed')
                return
            self._properties = properties
            self._properties_mono_ns = mono_ns

    def snapshot(self, now_mono_ns, wall_ns):
        self._validate_ns(now_mono_ns, wall_ns)
        with self._lock:
            self._check_wall(now_mono_ns, wall_ns)
            if (self._last_snapshot_mono_ns is not None and
                    now_mono_ns < self._last_snapshot_mono_ns):
                self._latch('monotonic_reversed')

            mapping = self._mapping
            properties = self._properties
            reason = self._fault
            if reason is None and (mapping is None or properties is None or
                                   self._actual_master != self.expected_master):
                reason = 'warming_up'
            if reason is None and now_mono_ns < mapping.last_ns:
                self._latch('monotonic_reversed')
                reason = self._fault
            if reason is None and now_mono_ns > mapping.expires_ns:
                reason = 'lease_expired'
            healthy = reason is None
            if healthy:
                reason = 'ok'

            self._sequence += 1
            if properties is None:
                utc_offset_valid = leap61 = leap59 = ptp_timescale = 0
            else:
                utc_offset_valid = properties['currentUtcOffsetValid']
                leap61 = properties['leap61']
                leap59 = properties['leap59']
                ptp_timescale = properties['ptpTimescale']
            if mapping is None:
                reference_mono_ns = last_sample_mono_ns = valid_until_ns = 0
                offset_ns = drift_ppm = residual_ns = empirical_error_ns = 0.0
                path_delay_ns = self._samples[-1]['delay_ns'] if self._samples else 0
            else:
                reference_mono_ns = last_sample_mono_ns = mapping.last_ns
                valid_until_ns = mapping.expires_ns
                offset_ns = mapping.offset_at_last_ns - UTC_OFFSET_S * 1_000_000_000
                drift_ppm = mapping.drift_ppm
                residual_ns = mapping.residual_ns
                empirical_error_ns = mapping.empirical_error_ns
                path_delay_ns = self._samples[-1]['delay_ns']

            result = ClockSnapshot(
                schema=1,
                sequence=self._sequence,
                healthy=healthy,
                reason=reason,
                boot_id=self.boot_id,
                session_id=self.session_id,
                expected_master=self.expected_master,
                actual_master=self._actual_master,
                scale='raw_ptp',
                utc_offset_s=UTC_OFFSET_S,
                utc_offset_valid=utc_offset_valid,
                leap61=leap61,
                leap59=leap59,
                ptp_timescale=ptp_timescale,
                reference_mono_ns=reference_mono_ns,
                offset_at_reference_ns=float(offset_ns),
                drift_ppm=float(drift_ppm),
                residual_ns=float(residual_ns),
                path_delay_ns=path_delay_ns,
                empirical_error_ns=float(empirical_error_ns),
                wall_minus_mono_ns=wall_ns - now_mono_ns,
                created_mono_ns=now_mono_ns,
                last_sample_mono_ns=last_sample_mono_ns,
                valid_until_ns=valid_until_ns,
            )
            self._last_snapshot_mono_ns = now_mono_ns
            self._last_snapshot = result
            return result
