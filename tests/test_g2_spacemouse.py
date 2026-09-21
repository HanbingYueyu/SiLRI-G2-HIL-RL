import pytest
import json
import subprocess
import sys
from g2_local import spacemouse


def test_translation_rotation_and_neutral_switch():
    mapper = spacemouse.ProposalMapper(axis_map=(2, -1, 3), deadzone=0.1)
    def feed(axes=(0, 0, 0, 1, 1, 1), left=False, valid=True):
        return mapper.update(axes, left_pressed=left, valid=valid)
    assert feed((.5, .5, .5, 0, 0, 0)).action == (0,) * 6
    assert feed().blocked is False
    assert feed((-.55, .55, -.55, 1, 1, 1)).action == pytest.approx((.5, .5, -.5, 0, 0, 0))
    assert feed((-.55, .55, -.55, 0, 0, 0), True).blocked
    assert feed(left=True).action == (0,) * 6
    assert feed((-.55, .55, -.55, 0, 0, 0), True).action == pytest.approx((0, 0, 0, .5, .5, -.5))
    assert feed((0, .55, 0, 0, 0, 0), False).blocked


def test_invalid_input_requires_neutral_rearm():
    mapper = spacemouse.ProposalMapper(axis_map=(1, 2, 3))
    mapper.update((0,) * 6, left_pressed=False, valid=True)
    assert mapper.update((1,) * 6, left_pressed=False, valid=False).blocked
    assert mapper.update((1,) * 6, left_pressed=False, valid=True).blocked
    assert not mapper.update((0,) * 6, left_pressed=False, valid=True).blocked
    with pytest.raises(ValueError):
        mapper.update((float('nan'),) * 6, left_pressed=False, valid=True)
    assert mapper.update((1,) * 6, left_pressed=False, valid=True).blocked


@pytest.mark.parametrize('mapping', [(1, 1, 3), (1, 2, 4), (0, 2, 3)])
def test_reject_bad_axis_mapping(mapping):
    with pytest.raises(ValueError):
        spacemouse.ProposalMapper(axis_map=mapping)


def test_readonly_json_preview():
    rows = [dict(axes=[0]*6, left_pressed=False, valid=True),
            dict(axes=[.55, 0, 0, 0, 0, 0], left_pressed=False, valid=True)]
    result = subprocess.run([sys.executable, '-m', 'g2_local.spacemouse',
                             '--axis-map', '1,2,3'],
                            input='\n'.join(map(json.dumps, rows))+'\n',
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(output) == 2
    assert output[-1]['action'] == pytest.approx([.5, 0, 0, 0, 0, 0])
    assert output[-1]['preview_only'] is True
