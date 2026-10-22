"""Fan stdin out to N worker processes in large newline-aligned blocks (a faster `split -n r/N --filter`).

`split -n r/N` writes line by line (one syscall per line), which caps a 1 TB N-Triples stream at about
500 MB/s. This reads BLOCK bytes at a time, cuts at the last newline and writes whole blocks round-robin.

    lbzip2 -dc dump.bz2 | python fanout.py 48 -- bash worker.sh OUT {worker} 256
Each "{worker}" argument is replaced by the worker index (w00, w01, ...).
"""
from __future__ import annotations

import subprocess
import sys

BLOCK = 8 << 20


def main():
    n = int(sys.argv[1])
    cmd = sys.argv[sys.argv.index("--") + 1:]
    procs = [subprocess.Popen([a.replace("{worker}", f"w{i:02d}") for a in cmd], stdin=subprocess.PIPE,
                              bufsize=0) for i in range(n)]
    inp = sys.stdin.buffer.raw
    carry, i = b"", 0
    while True:
        buf = inp.read(BLOCK)
        if not buf:
            break
        buf = carry + buf
        cut = buf.rfind(b"\n") + 1
        if cut == 0:
            carry = buf
            continue
        carry = buf[cut:]
        procs[i].stdin.write(memoryview(buf)[:cut])
        i = (i + 1) % n
    if carry:
        procs[i].stdin.write(carry)
    for p in procs:
        p.stdin.close()
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker exit codes: {codes}")


if __name__ == "__main__":
    main()
