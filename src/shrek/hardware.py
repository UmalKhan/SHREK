"""Hardware diagnostics without requiring a GPU inference installation."""

import importlib.util
import platform
import shutil
import subprocess
import sys


def hardware_report() -> dict:
    report = {"python": sys.version, "platform": platform.platform(),
              "packages": {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "accelerate")},
              "nvidia_smi": None, "cuda_available": None}
    command = shutil.which("nvidia-smi")
    if command:
        try:
            result = subprocess.run([command, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                report["nvidia_smi"] = result.stdout.strip()
            else:
                report["gpu_diagnostic_error"] = "nvidia-smi exited unsuccessfully"
        except (OSError, subprocess.TimeoutExpired):
            report["gpu_diagnostic_error"] = "nvidia-smi could not complete"
    if report["packages"]["torch"]:
        try:
            import torch
            report["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                report["gpu_name"] = torch.cuda.get_device_name(0)
                report["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        except (ImportError, OSError, RuntimeError):
            report["torch_diagnostic_error"] = "Torch is installed but could not initialize"
    report["note"] = "A missing NVIDIA diagnostic does not establish that no GPU exists. Select the model after reviewing usable VRAM."
    return report
