"""Collect a local training-environment report without network access.

Python 3.8+. No third-party dependency is required for the basic report.
Use --probe-torch to inspect PyTorch/CUDA in a separate, timed process.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run_command(argv, timeout=20):
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        options = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False, **options
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}


def package_versions():
    names = (
        "torch", "torchvision", "diffusers", "transformers", "accelerate",
        "peft", "safetensors", "huggingface-hub", "numpy", "Pillow",
        "einops", "webdataset", "prodigyopt", "flash-attn", "xformers",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


TORCH_PROBE = r'''
import json
import torch
info = {
    "torch_version": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cudnn_version": torch.backends.cudnn.version(),
    "devices": [],
}
if info["cuda_available"]:
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        info["devices"].append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        })
try:
    from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2, Flux2Transformer2DModel
    info["klein_classes_importable"] = True
except Exception as error:
    info["klein_classes_importable"] = False
    info["klein_import_error"] = str(error)
print(json.dumps(info))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="machine_environment.json")
    parser.add_argument("--disk-path", default=".", help="Check free space at this path")
    parser.add_argument("--probe-torch", action="store_true")
    args = parser.parse_args()
    disk_path = Path(args.disk_path).resolve()
    usage = shutil.disk_usage(str(disk_path))
    report = {
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_logical_count": os.cpu_count(),
        "disk": {
            "path": str(disk_path),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
        "packages": package_versions(),
        "nvidia_smi": run_command([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "gpu_topology": run_command(["nvidia-smi", "topo", "-m"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "docker_client": run_command(["docker", "--version"]),
    }
    os_release = Path("/etc/os-release")
    if platform.system() == "Linux" and os_release.is_file():
        report["linux_os_release"] = os_release.read_text(encoding="utf-8")
    meminfo = Path("/proc/meminfo")
    if platform.system() == "Linux" and meminfo.is_file():
        report["host_memory"] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in meminfo.read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))
        }
    if args.probe_torch:
        probe = run_command([sys.executable, "-c", TORCH_PROBE], timeout=45)
        if probe.get("returncode") == 0:
            try:
                probe["details"] = json.loads(probe["stdout"].splitlines()[-1])
            except (ValueError, IndexError):
                pass
        report["torch_probe"] = probe
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved environment report: {}".format(output))
    print("This report does not test model loading, GPU training, or distributed communication.")


if __name__ == "__main__":
    main()
