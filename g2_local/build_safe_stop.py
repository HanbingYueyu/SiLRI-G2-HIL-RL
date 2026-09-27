"""Build the narrow GDK 4.1.5 stop binding; never import or initialize GDK."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile


def main():
    root = Path(__file__).resolve().parent
    app = Path(os.environ.get('G2_GDK_RUNTIME', '/home/flyfuture/.cache/agibot/app')).resolve()
    if sys.version_info[:2] != (3, 10) or platform.machine() != 'x86_64':
        raise RuntimeError('This verified binding targets CPython 3.10 / x86_64 only')
    version = (app/'gdk/version').read_text()
    if not version.startswith('Version: 4.1.5\n'):
        raise RuntimeError('GDK version changed; review ABI before building')
    sdk = app/'gdk/build_dep/cpp/x86_64'
    torch = importlib.util.find_spec('torch')
    if torch is None:
        raise RuntimeError('Torch-supplied pybind11 headers are required')
    headers = Path(torch.origin).parent/'include'
    source = root/'native/safe_stop.cpp'
    target = root/('_gdk_safe_stop'+sysconfig.get_config_var('EXT_SUFFIX'))
    with tempfile.TemporaryDirectory(prefix='g2-stop-build-') as directory:
        binary = Path(directory)/target.name
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-fabi-version=16',
                        '-DPYBIND11_INTERNALS_VERSION=4',
                        '-I'+str(headers), '-I'+sysconfig.get_path('include'),
                        '-I'+str(sdk/'include'), str(source),
                        '-L'+str(app/'lib'), '-lgdk_adapter',
                        '-Wl,-rpath,'+str(app/'lib'), '-o', str(binary)], check=True)
        target.write_bytes(binary.read_bytes())
    record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  sdk_version=version)
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(record, indent=2))
    print('Built stop binding (no GDK initialization or motion):', target)


if __name__ == '__main__':
    main()
