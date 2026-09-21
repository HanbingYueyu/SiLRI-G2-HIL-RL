"""Bounded read-only check: python -m g2_local.probe --frames 3."""
import argparse
import json
from pathlib import Path
from .gdk_backend import GdkReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('runtime/gdk_probe'))
    args = parser.parse_args()
    if not 1 <= args.frames <= 100:
        parser.error('--frames must be in [1,100]')
    args.output.mkdir(parents=True, exist_ok=True)
    reader = GdkReader()
    try:
        import cv2
        with (args.output / 'observations.jsonl').open('a') as log:
            for index in range(args.frames):
                obs = reader.observe()
                info = dict(reader.last_info, state=obs['state'].tolist(),
                            shapes={key: list(value.shape) for key, value in obs.items()})
                log.write(json.dumps(info) + '\n')
                log.flush()
                print(json.dumps(info), flush=True)
                for key in reader.streams:
                    path = args.output / f'{info["received_monotonic_ns"]}_{key}.png'
                    if not cv2.imwrite(str(path), cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR)):
                        raise OSError(f'Cannot save {path}')
    finally:
        reader.close()


if __name__ == '__main__':
    main()
