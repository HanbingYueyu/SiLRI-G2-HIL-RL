import pytest


def samples():
    # Local wall clock leads raw GM by 18 s, gaining 10 us per second.
    return [dict(mono_ns=(100 + 2*i)*10**9,
                 offset_ns=55_000_000_000 + i*20_000,
                 delay_ns=40_000, master='044052.fffe.000010')
            for i in range(10)]


def model():
    from g2_local.clock_mapping import fit_mapping
    return fit_mapping(samples(), master='044052.fffe.000010',
                       utc_offset_s=37, session='test')


def test_parser_uses_measurement_time_not_pipe_delivery():
    from g2_local.clock_mapping import parse_ptp
    r = parse_ptp('ptp4l[12500.216]: master offset 55044921865 s0 '
                  'freq  +16486 path delay     37716', '044052.fffe.000010')
    assert r['mono_ns'] == 12_500_216_000_000
    assert r['offset_ns'] == 55_044_921_865
    assert r['delay_ns'] == 37_716
    assert parse_ptp('unrelated log', 'x') is None


def test_mapping_sign_scale_drift_and_error_growth():
    m = model()
    raw, error = m.offset_at(118_000_000_000, session='test', scale='raw_ptp')
    assert raw == pytest.approx(18_000_180_000, abs=1)
    utc, _ = m.offset_at(118_000_000_000, session='test', scale='utc')
    assert utc == pytest.approx(55_000_180_000, abs=1)
    assert m.drift_ppm == pytest.approx(10)
    assert error >= 2_000_000
    _, later_error = m.offset_at(119_000_000_000, session='test', scale='raw_ptp')
    assert later_error > error


@pytest.mark.parametrize('when,session,scale', [
    (121_000_000_000, 'test', 'raw_ptp'),
    (99_000_000_000, 'test', 'raw_ptp'),
    (118_000_000_000, 'other-boot-or-run', 'raw_ptp'),
    (118_000_000_000, 'test', 'unknown'),
])
def test_expired_wrong_session_or_scale_rejected(when, session, scale):
    with pytest.raises(ValueError):
        model().offset_at(when, session=session, scale=scale)


@pytest.mark.parametrize('mutation', ['master', 'gap', 'jump', 'negative_delay', 'nan'])
def test_bad_ptp_evidence_cannot_form_mapping(mutation):
    from g2_local.clock_mapping import fit_mapping
    s = samples()
    if mutation == 'master': s[5]['master'] = 'other'
    if mutation == 'gap': s[5]['mono_ns'] += 8_000_000_000
    if mutation == 'jump': s[5]['offset_ns'] += 1_000_000_000
    if mutation == 'negative_delay': s[5]['delay_ns'] = -1
    if mutation == 'nan': s[5]['offset_ns'] = float('nan')
    with pytest.raises(ValueError):
        fit_mapping(s, master='044052.fffe.000010', utc_offset_s=37, session='test')


@pytest.mark.parametrize('mutation', ['drift', 'residual'])
def test_only_drift_or_residual_limit_failures_are_marked_transient(mutation):
    """Catch startup recovery being enabled for structural mapping errors."""
    from g2_local.clock_mapping import TransientMappingFitError, fit_mapping

    evidence = samples()
    if mutation == 'drift':
        for index, sample in enumerate(evidence):
            sample['offset_ns'] = 55_000_000_000 + index * 1_000_000
    else:
        evidence[5]['offset_ns'] += 3_000_000

    with pytest.raises(TransientMappingFitError):
        fit_mapping(evidence, master='044052.fffe.000010',
                    utc_offset_s=37, session='test')


@pytest.mark.parametrize('polyfit_result', [
    (float('nan'), 0.0),
    (0.0, float(1 << 63)),
])
def test_nonfinite_or_out_of_range_fit_result_is_not_transient(
        monkeypatch, polyfit_result):
    """Catch structural mapping outputs being mislabeled recoverable."""
    import g2_local.clock_mapping as clock_mapping

    monkeypatch.setattr(clock_mapping.np, 'polyfit',
                        lambda *_args, **_kwargs: polyfit_result)
    with pytest.raises(ValueError) as caught:
        clock_mapping.fit_mapping(
            samples(), master='044052.fffe.000010', utc_offset_s=37,
            session='test')
    assert not isinstance(caught.value, clock_mapping.TransientMappingFitError)


def gdk_rows():
    rows = []
    for i in range(10):
        mono = (100+2*i)*10**9
        wall = 1_700_000_000_000_000_000 + mono
        stamp = wall - 18_000_000_000 - i*20_000 - 30_000_000
        rows.append(dict(kind='gdk', mono_ns=mono, wall_ns=wall,
                         start_mono_ns=mono-10_000_000,
                         start_wall_ns=wall-10_000_000,
                         timestamps={k: stamp for k in
                                     ('left_wrist', 'right_aux', 'joint', 'tf')},
                         tf_position_error_m=0., tf_rotation_error_rad=0.))
    return rows


def test_association_selects_only_raw_ptp_and_reports_age_intervals():
    from g2_local.clock_mapping import associate
    result = associate(model(), gdk_rows())
    assert result['scale'] == 'raw_ptp'
    assert result['motion_authorized'] is False
    assert result['ages_ms']['joint']['min'] == pytest.approx(30)
    assert result['ages_ms']['joint']['upper_max'] > 32


@pytest.mark.parametrize('mutation', ['stale', 'future', 'wall_jump', 'frozen', 'tf_direction', 'skew'])
def test_association_rejects_invalid_source_evidence(mutation):
    from g2_local.clock_mapping import associate
    rows = gdk_rows()
    if mutation == 'stale': rows[5]['timestamps']['joint'] -= 2_000_000_000
    if mutation == 'future': rows[5]['timestamps']['joint'] += 1_000_000_000
    if mutation == 'wall_jump': rows[5]['wall_ns'] += 1_000_000_000
    if mutation == 'frozen': rows[5]['timestamps']['tf'] = rows[4]['timestamps']['tf']
    if mutation == 'tf_direction': rows[5]['tf_position_error_m'] = .5
    if mutation == 'skew': rows[5]['timestamps']['right_aux'] -= 200_000_000
    with pytest.raises(ValueError):
        associate(model(), rows)


def test_gdk_collection_gap_cannot_be_hidden_by_remaining_samples():
    from g2_local.clock_mapping import associate
    rows = gdk_rows()
    del rows[4:6]
    with pytest.raises(ValueError, match='gap'):
        associate(model(), rows)
