"""Session-wide safety net: the test suite must never touch the real robot.

Why this exists
---------------
``scripts/start_training_actor.py`` runs the real per-episode loop. One loop test
forgot to replace one seam (``run_pre_reset``), so the *real*
``g2_local.pre_reset`` entry point was spawned as a child process and executed a
+Z 5 cm / +Y 10 cm retraction on the physical arm while a test suite was running.
A monkeypatch inside the test process cannot stop a child process, so the guard
has to sit on the spawn itself.

Two autouse fixtures:

* ``_no_real_motion_subprocesses`` refuses to spawn any argv that names a
  real-motion entry point (pre-reset, actor/learner, demonstration collection,
  jog, upstream Section 6, GDK probes). Child processes that only exercise the
  simulator (``--g2-software``) or the checkpoint/identity helpers still work.
* ``_no_real_gdk_init`` makes the real ``agibot_gdk.gdk_init()`` raise. Tests
  that inject a fake ``agibot_gdk`` into ``sys.modules`` are unaffected, because
  they never reach the real module.

Both fixtures preserve the existing seams: a test that monkeypatches
``launcher.run_child`` / ``launcher.run_pre_reset`` / ``manual_demo``'s
``command_runner`` is still granted its fake, since the guard only forbids the
process that would really move the arm.
"""
import subprocess
import sys

import pytest

# Any argv containing one of these would command (or take exclusive ownership of
# the hardware of) the real robot when run.
FORBIDDEN_ARGV = (
    '-m g2_local.pre_reset',
    '-m g2_local.real_train',
    '-m g2_local.manual_demo',
    '-m g2_local.commissioning_jog',
    'run_bilateral_flow_gdk.py',
    'gdk_reinit_probe.py',
    'gdk_clock_probe.py',
)


def _argv_text(argv):
    if isinstance(argv, (list, tuple)):
        return ' '.join(str(item) for item in argv)
    return str(argv)


def _refuse(command):
    text = _argv_text(command)
    for marker in FORBIDDEN_ARGV:
        if marker in text:
            raise RuntimeError(
                'Refusing to spawn a real-motion process from a test: '
                f'argv contains {marker!r}.\n'
                'Replace the seam (e.g. launcher.run_pre_reset / run_child, or '
                'manual_demo command_runner) with a fake instead of letting the '
                'test drive the hardware.')


@pytest.fixture(autouse=True)
def _no_real_motion_subprocesses(monkeypatch):
    real_popen = subprocess.Popen
    real_run = subprocess.run

    def popen(command, *args, **kwargs):
        _refuse(command)
        return real_popen(command, *args, **kwargs)

    def run(command, *args, **kwargs):
        _refuse(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setattr(subprocess, 'run', run)


@pytest.fixture(scope='session', autouse=True)
def _no_real_gdk_import():
    """Refuse to import the real GDK at all, without polluting ``sys.modules``.

    Importing ``agibot_gdk`` merely to patch it would make
    ``test_g2_real_training_integration`` fail: that rig asserts no hardware
    module was ever imported (``rig.hardware_imports == []``). A meta-path
    finder keeps the module out of ``sys.modules`` entirely, while tests that
    inject a fake (``monkeypatch.setitem(sys.modules, 'agibot_gdk', fake)``)
    still work, because the import system consults ``sys.modules`` first.
    """
    class _BlockRealHardware:
        BLOCKED = ('agibot_gdk',)

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in self.BLOCKED:
                raise ImportError(
                    f'Refusing to import the real hardware module {fullname!r} during '
                    'tests: it would initialise the robot. Inject a fake module into '
                    'sys.modules instead — see tests/test_g2_gdk_reader.py.')
            return None

    finder = _BlockRealHardware()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def test_the_motion_guard_refuses_every_real_entry_point():
    """Self-test of the safety net, without ever spawning a process."""
    for marker in FORBIDDEN_ARGV:
        with pytest.raises(RuntimeError, match='Refusing to spawn'):
            _refuse(['bash', 'run_g2_python.sh', marker, '--allow-motion'])
    # The simulator and the read-only helper paths must stay allowed.
    _refuse([sys.executable, 'learner.py', '--g2-software', '--port', '1'])
    _refuse(['git', 'rev-parse', 'HEAD'])
    _refuse([sys.executable, '-m', 'pytest', 'tests/test_g2_console.py'])
