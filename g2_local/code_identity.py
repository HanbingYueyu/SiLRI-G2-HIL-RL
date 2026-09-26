"""Small local reproducibility record including uncommitted source contents."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess


def source_digest(root):
    """Source identity, including dirty env code; not a dataset snapshot."""
    digest = hashlib.sha256()
    paths = set(root.glob('*.sh')) | set(root.glob('*.py'))
    for name in ('g2_local', 'lerobot/src/lerobot', 'rl_envs', 'rl_envs_sim'):
        paths.update(path for path in (root/name).rglob('*.py')
                     if not {'.git', '.venv', '__pycache__'}.intersection(path.parts))
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode()+b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


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
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'lerobot'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed-as-distribution'
    return dict(schema=3, main_sha=revision(root),
                vendored_sources={'lerobot': 'lerobot/src/lerobot'},
                submodules={name: revision(root/name)
                            for name in ('rl_envs', 'rl_envs_sim')},
                source_sha256=source_digest(root), python=platform.python_version(),
                packages=versions, policy_config=draccus.encode(create_policy_config(device)))
