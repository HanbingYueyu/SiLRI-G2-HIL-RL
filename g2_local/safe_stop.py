"""Prepare a left-arm-only stop request without sending it or changing mode."""
import hashlib
import json
from pathlib import Path


def prepare_safe_stop(robot, expected_mode):
    from . import _gdk_safe_stop as native
    binary = Path(native.__file__)
    record = json.loads(binary.with_suffix(binary.suffix+'.json').read_text())
    source = Path(__file__).parent/'native/safe_stop.cpp'
    if not native.robot_type_registered():
        raise RuntimeError('GDK Robot binding ABI is not registered')
    if (record['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or
            record['binary_sha256'] != hashlib.sha256(binary.read_bytes()).hexdigest()):
        raise RuntimeError('Rebuild GDK stop binding: source/binary identity changed')
    if native.request_fields(expected_mode) != (6, 1, expected_mode, 2, 10):
        raise RuntimeError('Unexpected GDK SAFE_STOP field layout')
    # Exact pybind type validation; no pointer arithmetic or new Robot instance.
    if native.check_robot(robot._robot) is not True:
        raise RuntimeError('GDK Robot ABI mismatch')

    def request():
        with robot._sdk_call_lock:
            return native.request_left_safe_stop(robot._robot, expected_mode)
    return request
