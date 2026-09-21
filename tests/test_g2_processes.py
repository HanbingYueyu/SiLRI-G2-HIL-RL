"""Real entrypoint/gRPC regression: checkpoint errors must fail the run."""
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch


def test_actor_learner_exchange_saves_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(root / 'lerobot/src'))
    args = ['--g2-software', '--port', str(port), '--updates', '2',
            '--timeout', '60', '--output', str(tmp_path)]
    processes = []
    try:
        for name in ('learner', 'actor'):
            processes.append(subprocess.Popen(
                [sys.executable, f'{name}.py', *args], cwd=root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        outputs = [process.communicate(timeout=90)[0] for process in processes]
        for process, output in zip(processes, outputs):
            assert process.returncode == 0, output
        assert '"loaded", "version": 2' in outputs[1]
        snapshot = torch.load(tmp_path / 'checkpoint.pt', weights_only=False)
        assert snapshot['version'] == 2
        assert len(snapshot['records']) == 8
        assert len(snapshot['optimizers']) >= 4
        for row in snapshot['records']:
            assert row['complementary_info']['synthetic']
            if row['complementary_info']['is_intervention']:
                assert torch.equal(row['action'], torch.zeros(6))
        resume_args = ['--g2-software', '--port', str(port), '--updates', '3',
                       '--timeout', '60', '--output', str(tmp_path / 'resumed'),
                       '--resume', str(tmp_path / 'checkpoint.pt')]
        resumed = []
        for name in ('learner', 'actor'):
            process = subprocess.Popen(
                [sys.executable, f'{name}.py', *resume_args], cwd=root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(process)
            resumed.append(process)
        outputs = [process.communicate(timeout=90)[0] for process in resumed]
        for process, output in zip(resumed, outputs):
            assert process.returncode == 0, output
        assert '"loaded", "version": 2' in outputs[1]
        assert '"loaded", "version": 3' in outputs[1]
        restored = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=False)
        assert restored['version'] == 3
        assert len(restored['records']) == 12
        before = next(iter(snapshot['optimizers']['critic']['state'].values()))['step']
        after = next(iter(restored['optimizers']['critic']['state'].values()))['step']
        assert after == before + 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
