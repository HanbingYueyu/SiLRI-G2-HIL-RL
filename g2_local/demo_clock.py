"""Reuse a read-only monitor across manually launched demo episodes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .clock_ipc import SnapshotClient
from .real_actor_audit import _stop_monitor, _wait_for_healthy_monitor


def _background_group():
    # Keep the controlling terminal/session for sudo's existing tty ticket,
    # but do not receive the demo foreground group's Ctrl+C or shell SIGHUP.
    os.setpgrp()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def ensure_clock(root, master):
    directory = Path(root) / 'runtime' / 'shared-demo-clock'
    directory.mkdir(mode=0o700, exist_ok=True)
    stat = directory.lstat()
    if directory.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        raise PermissionError('Shared clock directory must be owned and private')
    fd = os.open(directory / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = directory / 'current.json'
        if registry.exists():
            if registry.is_symlink():
                raise PermissionError('Symlink clock registry')
            saved = json.loads(registry.read_text())
            path = Path(saved['socket'])
            if path.exists():
                client = SnapshotClient(path, timeout_s=.5, expected_master=master)
                try:
                    # Reuse only currently healthy evidence, never cached approval.
                    client.read()
                except Exception as error:
                    raise RuntimeError(f'后台时钟尚未健康：{error}；日志 {path.parent}。'
                                       '未启动采集，也未另起监控覆盖旧会话。') from error
                finally:
                    client.close()
                print(f'复用后台时钟：{path}（本回合退出不关闭）', flush=True)
                return path
        print('首次启动后台只读时钟（最长运行 12 小时）；如提示请输入 sudo 密码。', flush=True)
        subprocess.run(['/usr/bin/sudo', '-v'], check=True)
        output = Path('/tmp') / ('g2-shared-' + uuid.uuid4().hex[:12])
        path = output / 'clock.sock'
        log_path = directory / (output.name + '.log')
        process = None
        ready = False
        try:
            with log_path.open('x') as log:
                process = subprocess.Popen([
                    sys.executable, '-m', 'g2_local.clock_monitor', '--master', master,
                    '--max-seconds', '43200', '--output', str(output)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    preexec_fn=_background_group)
            _wait_for_healthy_monitor(
                process, path, master=master, timeout_s=120.,
                client_factory=lambda **kw: SnapshotClient(
                    kw['path'], timeout_s=kw['timeout_s'], expected_master=kw['expected_master']),
                now_fn=time.monotonic, sleep_fn=time.sleep)
            temporary = directory / (output.name + '.json')
            with temporary.open('x') as stream:
                json.dump({'socket': str(path), 'log': str(log_path)}, stream)
            temporary.replace(registry)
            ready = True
            print(f'后台时钟已就绪：{path}；日志：{log_path}', flush=True)
            return path
        except Exception as error:
            raise RuntimeError(f'后台时钟启动失败：{error}；日志：{log_path}') from error
        finally:
            if not ready:
                _stop_monitor(process)
