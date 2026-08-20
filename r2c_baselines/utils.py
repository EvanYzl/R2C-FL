from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path], root: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p).lower()):
        rel = path.relative_to(root) if root is not None else path
        digest.update(str(rel).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def config_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame.copy()
    if "schema_version" not in value.columns:
        value.insert(0, "schema_version", SCHEMA_VERSION)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value.to_parquet(temp, index=False, compression="snappy")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temp, path)


def git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        source_files = sorted(path.rglob("*.py")) + sorted(path.rglob("*.yaml"))
        return f"tree-sha256:{hash_files(source_files, path)}" if source_files else "UNVERSIONED"


def environment_snapshot() -> dict[str, Any]:
    import numpy
    import pandas
    import pyarrow
    import sklearn
    import torch
    import torchvision

    value: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "sklearn": sklearn.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        value.update(
            {
                "gpu_model": props.name,
                "gpu_total_mib": int(props.total_memory / 2**20),
                "gpu_uuid": gpu_query().get("uuid"),
                "driver_version": gpu_query().get("driver_version"),
            }
        )
    return value


def gpu_query() -> dict[str, Any]:
    fields = "uuid,name,memory.total,memory.used,utilization.gpu,power.draw,driver_version"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [part.strip() for part in result.stdout.strip().split(",")]
        if len(parts) >= 7:
            return {
                "uuid": parts[0],
                "name": parts[1],
                "memory_total_mib": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "utilization_gpu_pct": float(parts[4]),
                "power_draw_w": None if parts[5] in {"[N/A]", "N/A"} else float(parts[5]),
                "driver_version": parts[6],
            }
    except Exception:
        pass
    return {}


def cpu_model() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return platform.processor() or "unknown"

