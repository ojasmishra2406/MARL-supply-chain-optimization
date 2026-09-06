import platform
import psutil
import cpuinfo

def get_hardware_info() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": cpuinfo.get_cpu_info().get("brand_raw", platform.processor()),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
        
    return info
