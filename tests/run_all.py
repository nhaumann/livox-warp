"""Run every check_*.py and smoke_*.py in this directory and summarise.

Each script runs as a subprocess with this interpreter and environment, its output shown as it comes. A
script that exits 0 after printing a line starting with "SKIP:" (its data is absent) counts as SKIP, any
other exit 0 as OK, and anything else as FAIL. Exit status 1 if anything failed.

usage: python tests/run_all.py [name ...]      e.g. python tests/run_all.py check_odometry smoke_pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: Path) -> tuple[str, float]:
    """Run one script; (OK | FAIL | SKIP, seconds)."""
    t0 = time.perf_counter()
    skipped = False
    proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=HERE.parent, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        skipped |= line.startswith("SKIP:")
        print(f"    {line}", end="")
    code = proc.wait()
    status = "FAIL" if code else ("SKIP" if skipped else "OK")
    return status, time.perf_counter() - t0


def main(names):
    scripts = sorted(list(HERE.glob("check_*.py")) + list(HERE.glob("smoke_*.py")))
    if names:
        wanted = {n.removesuffix(".py") for n in names}
        unknown = wanted - {s.stem for s in scripts}
        if unknown:
            sys.exit(f"no such script: {', '.join(sorted(unknown))}")
        scripts = [s for s in scripts if s.stem in wanted]
    results = []
    for script in scripts:
        print(f"== {script.name}")
        status, secs = run(script)
        results.append((script.name, status, secs))
        print(f"== {script.name}: {status} ({secs:.0f} s)\n")
    width = max(len(name) for name, _, _ in results)
    for name, status, secs in results:
        print(f"{name:<{width}}  {status:<4}  {secs:7.1f} s")
    counts = {s: sum(1 for _, status, _ in results if status == s) for s in ("OK", "FAIL", "SKIP")}
    print(f"\n{counts['OK']} OK, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP "
          f"in {sum(secs for _, _, secs in results):.0f} s")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
