"""
benchmark_core.py — System environment and git provenance.
Every value is measured at runtime; nothing is hardcoded.
"""
import platform, subprocess, datetime, json, os, sys
import torch

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def run():
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    commit = get_git_commit()

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_available else None

    try:
        cuda_version = torch.version.cuda
    except Exception:
        cuda_version = None

    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
        cpu_model = platform.processor() or "unknown"
        logical_cpus = psutil.cpu_count(logical=True)
    except ImportError:
        ram_gb = None
        cpu_model = platform.processor() or "unknown"
        logical_cpus = os.cpu_count()

    result = {
        "timestamp": ts,
        "git_commit": commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "cpu_model": cpu_model,
        "logical_cpus": logical_cpus,
        "ram_gb": ram_gb,
        "os": platform.platform(),
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
