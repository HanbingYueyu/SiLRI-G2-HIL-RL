"""Small local reproducibility record including uncommitted source contents."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess


def algorithm_identity(device):
    import draccus
    from .policy import create_policy_config
    root = Path(__file__).resolve().parents[1]
    def revision(path):
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise ValueError(f'Cannot identify algorithm repository: {path.name}')
        return result.stdout.strip()
    digest = hashlib.sha256()
    for directory in (root/'g2_local', root/'lerobot/src/lerobot'):
        for path in sorted(directory.rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0')
            digest.update(path.read_bytes())
            digest.update(b'\0')
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'lerobot'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed-as-distribution'
    return dict(schema=1, main_sha=revision(root),
                submodules={name: revision(root/name)
                            for name in ('lerobot', 'rl_envs', 'rl_envs_sim')},
                source_sha256=digest.hexdigest(), python=platform.python_version(),
                packages=versions, policy_config=draccus.encode(create_policy_config(device)))
