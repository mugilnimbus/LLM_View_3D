from __future__ import annotations

import json
import platform
import shutil
import subprocess

import psutil

from llm_view.core.schemas import GpuInfo, HardwareInfo


def detect_hardware() -> HardwareInfo:
    memory = psutil.virtual_memory()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(memory.total / 1024**3, 2),
        ram_available_gb=round(memory.available / 1024**3, 2),
        gpus=_detect_nvidia_gpus(),
    )


def _detect_nvidia_gpus() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        return []

    query = ",".join(
        [
            "name",
            "memory.total",
            "memory.used",
            "memory.free",
            "utilization.gpu",
            "temperature.gpu",
        ]
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=6)
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        name, total, used, free, utilization, temperature = parts
        gpus.append(
            GpuInfo(
                name=name,
                memory_total_mb=_int_or_zero(total),
                memory_used_mb=_int_or_zero(used),
                memory_free_mb=_int_or_zero(free),
                utilization_percent=_int_or_zero(utilization),
                temperature_c=_int_or_zero(temperature),
            )
        )
    return gpus


def _int_or_zero(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def hardware_as_json() -> str:
    return json.dumps(detect_hardware().model_dump(), indent=2)
