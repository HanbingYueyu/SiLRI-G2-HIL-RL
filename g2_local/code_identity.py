"""Small local reproducibility record including uncommitted source contents.

Two identities, on purpose
--------------------------
``contract_digest`` is what a seed/checkpoint actually *depends on*: the
demonstration contract (task/observation/intervention/motion keys), the modules
that define and validate that contract, and the policy configuration
(observation shape, action size, architecture). It is the only identity that is
enforced when a seed is loaded.

``source_digest`` is a full digest of every algorithm source file. It is recorded
in every manifest and checkpoint for audit, but it is **not** enforced: none of
the runtime/executor code is read back out of a checkpoint, so editing the Actor
loop, the GDK backend, a console string or a docstring must not invalidate a 5 GB
seed and a running Learner. Enforcing it (the old behaviour) turned every honest
bug fix into "rebuild the seed", which is a bug in its own right.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys


# Modules whose bytes participate in the *contract*: they define the task/action
# contract, the workspace/observation schema, and the demonstration import and
# validation format. A change here can reinterpret stored data or stored replay
# tensors, so it invalidates a seed on purpose.
SEED_CONTRACT_SOURCES = (
    'g2_local/contract.py',
    'g2_local/config.py',
    'g2_local/demonstrations.py',
)

# Keys inside the encoded policy configuration that say *where* and *how fast* to
# compute, not *what* is computed. They are excluded from the contract digest so
# the enforced identity is machine-independent: a seed built on a CUDA box must be
# verifiable and loadable on a CPU-only one, and an operator must be able to
# reproduce the digest by hand. Both values live in the configuration JSON, whose
# own hash (``config_hash``) is still enforced.
POLICY_PLACEMENT_KEYS = ('device', 'storage_device', 'use_amp')

# Training-signal keys that are *baked into stored transitions*: every saved row
# carries the reward and the done/truncated labels computed with these values, so
# a checkpoint trained under different values would silently mix two reward
# definitions. They therefore belong to the enforced contract, while tunables that
# are not baked in (timeouts, batch sizes, bc_weight, learning rates, checkpoint
# cadence, capacities -- the latter is checked by the replay contract anyway) do
# not, and can be changed without rebuilding anything.
CHECKPOINT_SIGNAL_TASK_KEYS = ('success_reward', 'failure_reward', 'step_reward',
                              'max_episode_steps')

# Training-semantics implementation: the Actor/Critic/Expert/Target mathematics and
# the optimizer step itself. A checkpoint carries optimizer state and target
# networks that only mean what they meant when this code wrote them, so changing
# any of these files must NOT be able to resume silently. Everything else in the
# source tree stays audit-only (see `source_sha256`).
TRAINING_SEMANTICS_SOURCES = (
    'lerobot/src/lerobot/policies/silri/modeling_silri.py',
    'lerobot/src/lerobot/policies/silri/configuration_silri.py',
    'g2_local/runtime.py',
)


def source_digest(root):
    """Full source identity, including dirty env code; not a dataset snapshot."""
    root = Path(root)
    digest = hashlib.sha256()
    paths = set(root.glob('*.sh')) | set(root.glob('*.py'))
    for name in ('g2_local', 'lerobot/src/lerobot', 'rl_envs', 'rl_envs_sim'):
        paths.update(path for path in (root/name).rglob('*.py')
                     if not {'.git', '.venv', '__pycache__'}.intersection(path.parts))
    paths.update((root/'g2_local/native').glob('*.cpp'))
    paths.update((root/'g2_local').glob('_gdk_safe_stop*.so'))
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode()+b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def algorithm_digest():
    """The enforced training-semantics digest (losses, targets, optimizer step)."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in TRAINING_SEMANTICS_SOURCES:
        digest.update(name.encode()+b'\0')
        digest.update((root/name).read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def current_source_digest():
    """Full source digest of the running checkout (audit/notice only)."""
    return source_digest(Path(__file__).resolve().parents[1])


def contract_digest(config):
    """The enforced seed/checkpoint identity: contract + policy configuration."""
    import draccus
    from .demonstrations import demonstration_contract
    from .policy import create_policy_config
    from .training_config import canonical_json
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(b'demonstration-contract\0')
    digest.update(demonstration_contract(config).encode())
    digest.update(b'checkpoint-signal\0')
    task = dict(config.canonical_payload)['task']
    digest.update(canonical_json({key: task[key] for key in CHECKPOINT_SIGNAL_TASK_KEYS}))
    for name in SEED_CONTRACT_SOURCES:
        digest.update(name.encode()+b'\0')
        digest.update((root/name).read_bytes())
        digest.update(b'\0')
    digest.update(b'policy-config\0')
    policy = {key: value for key, value in
              draccus.encode(create_policy_config(config.runtime.device)).items()
              if key not in POLICY_PLACEMENT_KEYS}
    digest.update(canonical_json(policy))
    return digest.hexdigest()


def algorithm_identity(config):
    """Full recorded identity for a loaded config (or a partial test config).

    ``contract_sha256`` is present whenever ``config`` is a fully loaded training
    configuration. Enforcement always compares that field, so a recorded identity
    without it (a partial stub, or a checkpoint written before this change) is
    refused rather than silently accepted.
    """
    import draccus
    from .policy import create_policy_config
    if not hasattr(config, 'runtime'):
        raise ValueError('A training config with runtime settings is required')
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
    identity = dict(schema=3, main_sha=revision(root),
                    vendored_sources={'lerobot': 'lerobot/src/lerobot'},
                    submodules={name: revision(root/name)
                                for name in ('rl_envs', 'rl_envs_sim')},
                    source_sha256=source_digest(root), python=platform.python_version(),
                    packages=versions,
                    policy_config=draccus.encode(create_policy_config(config.runtime.device)))
    if hasattr(config, 'canonical_payload'):
        identity['contract_sha256'] = contract_digest(config)
    identity['algorithm_sha256'] = algorithm_digest()
    return identity


def _field(identity, name):
    return identity.get(name) if isinstance(identity, dict) else None


def seed_identity_check(path, stored, current):
    """(ok, detail) for a stored vs current algorithm identity.

    Two things decide usability: the data/task contract and the training-semantics
    implementation. Everything else (runtime code, logs, git revision, packages)
    is reported by ``*_drift_notice`` instead of refusing the checkpoint.
    """
    if not isinstance(stored, dict):
        return False, f'checkpoint={Path(path).resolve()} has no algorithm identity'
    for field in ('contract_sha256', 'algorithm_sha256'):
        if stored.get(field) is None or stored.get(field) != current.get(field):
            return False, (f'checkpoint={Path(path).resolve()} {field}: '
                           f'stored={stored.get(field)} current={current.get(field)}')
    return True, None


def config_drift_notice(path, stored_hash, current_hash):
    """One-line notice when only the (unenforced) config hash differs."""
    if stored_hash is None or stored_hash == current_hash:
        return None
    return (f'note: {Path(path).resolve()} was written under config_hash '
            f'{stored_hash} but this run uses {current_hash}; the task/reward/action '
            'contract is unchanged, so this checkpoint stays usable. The mismatch is '
            'recorded for audit only.')


def announce_config_drift(path, stored_hash, current_hash):
    notice = config_drift_notice(path, stored_hash, current_hash)
    if notice is not None:
        print(notice, file=sys.stderr, flush=True)
    return notice


def source_drift_notice(path, stored, current):
    """One-line audit notice when only the (unenforced) source digest differs."""
    stored_source = _field(stored, 'source_sha256')
    current_source = _field(current, 'source_sha256')
    if stored_source is None or stored_source == current_source:
        return None
    return (f'note: {Path(path).resolve()} was written by different algorithm '
            f'sources (stored source_sha256={stored_source} '
            f'current source_sha256={current_source}); the task/action contract and '
            'policy configuration are unchanged, so this checkpoint stays usable. '
            'The mismatch is recorded for audit only.')


def announce_source_drift(path, stored, current):
    """Print the audit notice, if any, without changing behaviour."""
    notice = source_drift_notice(path, stored, current)
    if notice is not None:
        print(notice, file=sys.stderr, flush=True)
    return notice


def identity_mismatch_detail(path, stored, current):
    """Operator-facing evidence for a refused checkpoint.

    Kept for callers that still want the raw digests; the enforced comparison is
    ``seed_identity_check``.
    """
    return seed_identity_check(path, stored, current)[1] or (
        f'checkpoint={Path(path).resolve()} identities match')
