import json
from types import SimpleNamespace

import pytest

from g2_local.operator_control import EpisodeContextInbox, StartChord, TerminalKeyReader


def frame(*, buttons, pressed=()):
    return SimpleNamespace(buttons=buttons, pressed=pressed, ready=True)


def write_context(path, *, episode_id):
    path.write_text(json.dumps(dict(episode_id=episode_id, target_offset_m=[0, 0, 0],
                                    approach_source='visual', grasp_description='fixed')))


def test_chord_requires_both_buttons_then_full_release():
    chord = StartChord(left_button=0, right_button=1)
    assert chord.update(frame(buttons=(True,True), pressed=(0,1))) is False
    assert chord.update(frame(buttons=(False,False))) is True
    assert chord.update(frame(buttons=(False,False))) is False


def test_duplicate_context_is_rejected(tmp_path):
    inbox = EpisodeContextInbox(tmp_path / 'context.json')
    write_context(inbox.path, episode_id='episode-1')
    assert inbox.read_new().episode_id == 'episode-1'
    with pytest.raises(ValueError, match='duplicate'): inbox.read_new()


def test_chord_needs_fresh_joint_press_and_release():
    chord = StartChord(left_button=0, right_button=1)
    assert chord.update(frame(buttons=(True, False), pressed=(0,))) is False
    assert chord.update(frame(buttons=(True, True), pressed=(1,))) is False
    assert chord.update(frame(buttons=(False, False))) is False
    assert chord.update(frame(buttons=(True, True), pressed=(0, 1))) is False
    assert chord.update(frame(buttons=(True, False))) is False
    assert chord.update(frame(buttons=(False, False))) is True


def test_terminal_reader_bounds_and_conflicting_labels():
    class Keys:
        def __init__(self): self.keys = ['Y', 'x', 'y']
        def read_available(self, *, limit):
            assert limit == 32
            return self.keys
    keys = Keys()
    reader = TerminalKeyReader(keys)
    assert reader.poll() == 'success'
    keys.keys = ['f']
    assert reader.poll() == 'failure'
    keys.keys = ['y', 'f']
    with pytest.raises(RuntimeError, match='Conflicting'): reader.poll()


def test_context_inbox_rejects_symlink_and_oversize(tmp_path):
    owned = tmp_path / 'owned.json'
    write_context(owned, episode_id='episode-1')
    link = tmp_path / 'link.json'
    link.symlink_to(owned)
    with pytest.raises(ValueError): EpisodeContextInbox(link).read_new()
    owned.write_text(' ' * 16_385)
    with pytest.raises(ValueError): EpisodeContextInbox(owned).read_new()
