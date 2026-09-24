"""Event-driven HID-only rotation check with an exclusive evidence journal."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
from .spacemouse import RotationCheck


def run_check(reader, session, emit, *, clock=time.monotonic, sleep=time.sleep):
    try:
        while session.stage not in ('passed', 'failed'):
            frame = reader.poll()
            now = clock()
            result = session.update(frame, now=now)
            emit(dict(result, raw_axes=frame.axes, buttons=frame.buttons,
                      axis_times=frame.axis_times, ready=frame.ready,
                      received_monotonic=now))
            if session.stage not in ('passed', 'failed'):
                sleep(.02)
    except (Exception, KeyboardInterrupt):
        emit(session.fail('reader_or_recording_error'))
        raise
    finally:
        session.gate.invalidate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', choices=('roll', 'pitch', 'yaw'), required=True)
    parser.add_argument('--axis-map', required=True)
    parser.add_argument('--left-button', type=int, required=True, choices=(0, 1))
    parser.add_argument('--stage-timeout', type=float, default=60.)
    parser.add_argument('--device')
    parser.add_argument('--adapter-root', type=Path, default=Path('/home/flyfuture/g2_hinge_assembly'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    session = RotationCheck(axis=args.axis, started_at=time.monotonic(),
                            timeout=args.stage_timeout,
                            axis_map=tuple(int(i) for i in args.axis_map.split(',')),
                            left_button=args.left_button)
    output = args.output or Path('runtime/calibration') / f'{uuid.uuid4().hex}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    positive, negative = {'roll': ('向绕 X 正方向倾斜', '向绕 X 反方向倾斜'),
                          'pitch': ('向绕 Y 正方向倾斜', '向绕 Y 反方向倾斜'),
                          'yaw': ('向绕 Z 正方向扭转', '向绕 Z 反方向扭转')}[args.axis]
    prompts = {'press': '按住左键，轻拨后回中，等待收到轴报告',
               'neutral': '保持左键，轻拨后松旋帽回中',
               'positive': f'保持左键，{positive}；若门控锁住，先轻拨回中再做',
               'return_positive': '保持左键，松旋帽回中',
               'negative': f'保持左键，{negative}；若门控锁住，先轻拨回中再做',
               'return_negative': '保持左键，松旋帽回中',
               'release': '松开左键，旋帽保持回中；若报告过期，轻拨后回中',
               'passed': '本轴只读流程通过；不是机器人运动验收',
               'failed': '本次未通过，请查看原因；不会自动重试'}
    previous = None
    with output.open('x', encoding='utf-8') as log:
        log.write(json.dumps(dict(event='config', axis=args.axis, axis_map=args.axis_map,
                                  left_button=args.left_button, preview_only=True,
                                  stage_timeout=args.stage_timeout,
                                  max_age=.25, deadzone=.1, target_min=.2, cross_max=.15))+'\n')
        def emit(row):
            nonlocal previous
            log.write(json.dumps(row)+'\n')
            log.flush()
            key = (row['stage'], row['reason'])
            if key != previous:
                print(f"[{row['stage']}] {prompts[row['stage']]} | {row['reason']}", flush=True)
                previous = key
        emit(session.status())
        print(f'只读证据文件：{output}', flush=True)
        sys.path.insert(0, str(args.adapter_root.resolve()))
        try:
            from g2_adapter.spacemouse_input import CompactHID
            with CompactHID(args.device) as reader:
                run_check(reader, session, emit)
        except (Exception, KeyboardInterrupt):
            if session.stage != 'failed':
                emit(session.fail('reader_open_or_import_error'))
            raise
    if session.stage != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
