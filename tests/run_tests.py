"""Dependency-free test runner for the frozen experiment environment."""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


MODULES = ("test_metrics", "test_methods", "test_data_trace", "test_queue", "test_r2c")


def main() -> int:
    failures = []
    count = 0
    started = time.perf_counter()
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, value in sorted(vars(module).items()):
            if name.startswith("test_") and inspect.isfunction(value):
                count += 1
                try:
                    value()
                    print(f"PASS {module_name}.{name}")
                except Exception as exc:
                    failures.append((module_name, name, exc))
                    print(f"FAIL {module_name}.{name}: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    print(f"tests={count} failures={len(failures)} elapsed_s={elapsed:.3f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
