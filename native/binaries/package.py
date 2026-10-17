#!/usr/bin/env python3
"""Build portable native tools from the pinned llama.cpp checkout."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(*args):
    subprocess.run([str(a) for a in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'metal', 'vulkan', 'cuda', 'opencl'], default='cpu')
    parser.add_argument('--product', choices=['mimir', 'llama'], default='mimir')
    parser.add_argument('--version', default='0.1.5')
    args = parser.parse_args()
    system = {'Darwin': 'macos', 'Linux': 'linux', 'Windows': 'windows'}[platform.system()]
    arch = 'arm64' if platform.machine().lower() in ('arm64', 'aarch64') else 'x64'
    is_llama = args.product == 'llama'
    prefix = 'llama-mimir' if is_llama else 'dfm-mimir'
    name = f'{prefix}-{args.version}-{system}-{arch}-{args.backend}' + ('' if is_llama else '-cli')
    targets = ['llama-cli', 'llama-server', 'llama-bench', 'llama-quantize'] if is_llama else ['mimir-chat', 'mimir-backend-policy-tests', 'mimir-compaction-tests']
    build = ROOT / 'logs' / 'binary-builds' / name
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DBUILD_SHARED_LIBS=ON',
             '-DGGML_BACKEND_DL=ON', '-DGGML_OPENMP=OFF', '-DLLAMA_OPENSSL=OFF',
             '-DMIMIR_BUILD_TESTS=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
             '-DCMAKE_INSTALL_RPATH=' + ('@loader_path' if system == 'macos' else '$ORIGIN')]
    for backend in ['metal', 'vulkan', 'cuda', 'opencl']:
        flags.append(f'-DGGML_{backend.upper()}={"ON" if backend == args.backend else "OFF"}')
    if system == 'windows' and arch == 'arm64':
        flags += ['-T', 'ClangCL', '-A', 'ARM64']
    if arch == 'x64':
        flags += ['-DGGML_CPU_ALL_VARIANTS=ON']
    if system == 'macos':
        flags += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0', '-DGGML_METAL_EMBED_LIBRARY=ON']
    if is_llama:
        flags += ['-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                  '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_TOOLS=ON',
                  '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_COMMON=ON']
    _run('cmake', '-S', ROOT / ('llama.cpp' if is_llama else 'native/mimir'), '-B', build, *flags)
    _run('cmake', '--build', build, '--config', 'Release', '--target',
         *targets, '--parallel', '3')
    if not is_llama:
        _run('ctest', '--test-dir', build, '-C', 'Release', '-R', 'backend-policy|compaction-planner', '--output-on-failure')
    stage = ROOT / 'logs' / 'binary-packages' / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for path in (build / 'bin').rglob('*'):
        if path.is_file() and (path.name in [t + ('.exe' if system == 'windows' else '') for t in targets] or
                              '.so' in path.name or path.suffix in ('.dll', '.dylib', '.metallib')):
            shutil.copy2(path, stage / path.name)
    exe = stage / (targets[0] + ('.exe' if system == 'windows' else ''))
    for target in (targets if is_llama else targets[:1]):
        result = subprocess.run([str(stage / (target + ('.exe' if system == 'windows' else ''))), '--help'],
                                cwd=stage, text=True, capture_output=True)
        print(result.stdout, result.stderr)
        expected = 1 if target == 'llama-quantize' else 0
        if result.returncode != expected or 'usage' not in (result.stdout + result.stderr).lower():
            raise RuntimeError(f'{target} help failed: exit {result.returncode}')
    devices = subprocess.check_output([str(exe), '--list-devices'], text=True, cwd=stage)
    (stage / 'devices.txt').write_text(devices)
    shutil.copy2(ROOT / 'LICENSE', stage / 'LICENSE')
    shutil.copy2(ROOT / 'llama.cpp/LICENSE', stage / 'LICENSE-llama.cpp')
    shutil.copytree(ROOT / 'llama.cpp/licenses', stage / 'licenses')
    shutil.copy2(ROOT / ('native/binaries/LLAMA.md' if is_llama else 'native/binaries/README.md'), stage / 'README.md')
    (stage / 'manifest.json').write_text(json.dumps({
        'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'llama_cpp': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'llama.cpp', text=True).strip(),
        'product': args.product, 'platform': system, 'architecture': arch, 'backend': args.backend,
        'validation': 'Build, packaged help and device enumeration only; model inference not qualified by this job.',
    }, indent=2))
    output = ROOT / 'logs' / 'binary-assets'
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(shutil.make_archive(str(output / name), 'zip' if system == 'windows' else 'gztar', stage.parent, name))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_text(f'{digest}  {archive.name}\n')
    print(archive)


if __name__ == '__main__':
    main()
