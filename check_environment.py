from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "single_phase"
REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("pymatgen", "pymatgen"),
    ("mp_api", "mp-api"),
    ("tqdm", "tqdm"),
]
REQUIRED_SPLITS = ["train", "val", "test_normal", "test_hard"]
REQUIRED_NPZ_FIELDS = [
    "X",
    "y",
    "labels",
    "two_theta",
    "lattice_params",
    "peak_hkls",
    "peak_2theta",
    "peak_mask",
    "crystal_system_y",
    "crystal_system_labels",
]


def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def import_checks() -> tuple[bool, dict[str, Any]]:
    results: dict[str, Any] = {}
    all_ok = True
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            results[package_name] = {
                "ok": True,
                "module": module_name,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as exc:  # pragma: no cover - diagnostic script
            all_ok = False
            results[package_name] = {
                "ok": False,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return all_ok, results


def pip_check() -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout.strip()


def torch_checks() -> tuple[bool, dict[str, Any]]:
    import torch
    from torch import nn

    cuda_available = torch.cuda.is_available()
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )

    forward_backward_ok = True
    forward_backward_error = None
    try:
        device = torch.device("cuda" if cuda_available else "cpu")
        x = torch.randn(4, 1, 3501, device=device)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, 207),
        ).to(device)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
    except Exception as exc:  # pragma: no cover - diagnostic script
        forward_backward_ok = False
        forward_backward_error = f"{type(exc).__name__}: {exc}"

    return forward_backward_ok, {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "forward_backward_ok": forward_backward_ok,
        "forward_backward_error": forward_backward_error,
    }


def dataset_checks() -> tuple[bool, dict[str, Any]]:
    import numpy as np

    all_ok = True
    results: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        split_path = DATASET_DIR / f"{split}.npz"
        split_result: dict[str, Any] = {"path": str(split_path), "exists": split_path.exists()}
        if not split_path.exists():
            all_ok = False
            results[split] = split_result
            continue

        try:
            data = np.load(split_path, allow_pickle=True)
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in data.files]
            split_ok = not missing
            x = data["X"] if "X" in data.files else None
            y = data["y"] if "y" in data.files else None
            crystal_y = data["crystal_system_y"] if "crystal_system_y" in data.files else None
            labels = data["labels"] if "labels" in data.files else None
            crystal_labels = data["crystal_system_labels"] if "crystal_system_labels" in data.files else None
            split_result.update(
                {
                    "ok": split_ok,
                    "missing_fields": missing,
                    "X_shape": list(x.shape) if x is not None else None,
                    "X_dtype": str(x.dtype) if x is not None else None,
                    "y_shape": list(y.shape) if y is not None else None,
                    "y_minmax": [int(y.min()), int(y.max())] if y is not None and y.size else None,
                    "label_count": int(len(labels)) if labels is not None else None,
                    "crystal_system_y_minmax": (
                        [int(crystal_y.min()), int(crystal_y.max())]
                        if crystal_y is not None and crystal_y.size
                        else None
                    ),
                    "crystal_system_label_count": int(len(crystal_labels)) if crystal_labels is not None else None,
                    "lattice_params_shape": (
                        list(data["lattice_params"].shape) if "lattice_params" in data.files else None
                    ),
                    "peak_hkls_shape": list(data["peak_hkls"].shape) if "peak_hkls" in data.files else None,
                }
            )
            all_ok = all_ok and split_ok
        except Exception as exc:  # pragma: no cover - diagnostic script
            all_ok = False
            split_result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        results[split] = split_result
    return all_ok, results


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    print_section("Python")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.replace(chr(10), ' ')}")
    print(f"platform:   {platform.platform()}")
    print(f"project:    {PROJECT_ROOT}")

    print_section("Imports")
    imports_ok, imports = import_checks()
    for package_name, info in imports.items():
        if info["ok"]:
            print(f"{status(True):4} {package_name:14} version={info['version']}")
        else:
            print(f"{status(False):4} {package_name:14} {info['error']}")

    print_section("Pip Dependency Check")
    pip_ok, pip_output = pip_check()
    print(f"{status(pip_ok):4} {pip_output or 'No broken requirements found.'}")

    print_section("PyTorch / CUDA")
    torch_ok, torch_info = torch_checks()
    print(f"torch version:       {torch_info['torch_version']}")
    print(f"cuda available:      {torch_info['cuda_available']}")
    print(f"torch cuda version:  {torch_info['torch_cuda_version']}")
    print(f"cuda device count:   {torch_info['device_count']}")
    if torch_info["devices"]:
        for device in torch_info["devices"]:
            print(f"device {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
    else:
        print("devices:             none visible to PyTorch")
    print(f"{status(torch_info['forward_backward_ok']):4} torch forward/backward smoke test")
    if torch_info["forward_backward_error"]:
        print(f"     {torch_info['forward_backward_error']}")

    print_section("Dataset")
    dataset_ok, dataset = dataset_checks()
    for split, info in dataset.items():
        print(
            f"{status(bool(info.get('ok', False))):4} {split:11} "
            f"exists={info['exists']} X={info.get('X_shape')} y={info.get('y_minmax')} "
            f"crystal_y={info.get('crystal_system_y_minmax')}"
        )
        if info.get("missing_fields"):
            print(f"     missing_fields={info['missing_fields']}")
        if info.get("error"):
            print(f"     {info['error']}")

    summary = {
        "imports_ok": imports_ok,
        "pip_check_ok": pip_ok,
        "torch_smoke_test_ok": torch_ok,
        "cuda_available": torch_info["cuda_available"],
        "dataset_ok": dataset_ok,
        "can_run_training": imports_ok and pip_ok and torch_ok and dataset_ok,
    }
    print_section("Summary")
    print(json.dumps(summary, indent=2))

    if not summary["can_run_training"]:
        return 1
    if not summary["cuda_available"]:
        print("\nNote: training can run, but PyTorch is CPU-only or CUDA is not visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
