"""Training input integrity is mandatory; executable hashes are provenance."""
import hashlib
import json
from pathlib import Path
import sys
import time


def check_training_pins(pins, *, audit_path=None):
    changes = []
    for name, expected in sorted(pins.items()):
        path = Path(name)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            actual = None
        if actual == expected:
            continue
        change = {'path': name, 'expected': expected, 'actual': actual}
        if path.suffix not in {'.py', '.sh'}:
            raise ValueError(f'Training input/configuration changed: {name}')
        changes.append(change)
    if changes:
        # Do not use warnings.warn: PYTHONWARNINGS=error must not turn
        # provenance into an accidental production stop again.
        print('TRAINING CODE PROVENANCE WARNING: continuing with changed code: '
              + ', '.join(c['path'] for c in changes), file=sys.stderr, flush=True)
        if audit_path is not None:
            try:
                path = Path(audit_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps({'time': time.time(), 'changes': changes}) + '\n')
            except OSError as error:
                print(f'Cannot record code provenance: {error}', file=sys.stderr, flush=True)
    return changes
