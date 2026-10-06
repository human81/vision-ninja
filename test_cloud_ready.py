"""Cloud-readiness — offline regression for what changes on a Linux/Cloud Run host.

Simulates "no MPS" (and "no MPS, has CUDA") without leaving the Mac: the device helper
must fall back instead of crashing, YOLO must drop FP16 on CPU, and STUDIO_LOCAL_MODELS=off
must hide + refuse the ~8GB local vision models (an OOM on a CPU instance).

    .venv/bin/python test_cloud_ready.py
"""

from __future__ import annotations

import os

import torch

from occ import device


def collect() -> list[tuple[str, bool, str]]:
    R: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        R.append((name, bool(cond), detail))

    real_mps, real_cuda = torch.backends.mps.is_available, torch.cuda.is_available
    saved_env = os.environ.get("STUDIO_LOCAL_MODELS")
    try:
        # --- device fallback ---
        torch.backends.mps.is_available, torch.cuda.is_available = (lambda: False), (lambda: False)
        check("no GPU: 'mps' → cpu", device.pick_device("mps") == "cpu")
        check("no GPU: 'cuda' → cpu", device.pick_device("cuda") == "cpu")
        check("no GPU: 'auto' → cpu", device.pick_device("auto") == "cpu")
        torch.cuda.is_available = lambda: True
        check("L4 host: 'mps' → cuda", device.pick_device("mps") == "cuda")
        check("L4 host: 'auto' → cuda", device.pick_device(None) == "cuda")
        check("explicit 'cpu' honoured", device.pick_device("cpu") == "cpu")
        torch.cuda.is_available = lambda: False

        from occ.config import Config
        from occ.detectors.yolo import YoloDetector
        y = YoloDetector(Config.load(overrides=["detector.device=mps", "detector.half=true"]))
        check("YOLO on a CPU host: device cpu, FP16 off", (y.device, y.half) == ("cpu", False),
              f"{y.device}, half={y.half}")
        torch.backends.mps.is_available, torch.cuda.is_available = real_mps, real_cuda

        # --- local GPU models switch ---
        from occ.studio import gemma_vision, medgemma
        from occ.studio.settings import LOCAL_MODELS, StudioSettings, model_options
        os.environ["STUDIO_LOCAL_MODELS"] = "off"
        opts = model_options()
        check("off: local models hidden from every dropdown",
              not any(m in LOCAL_MODELS for v in opts.values() for m in v), str(opts))
        s = StudioSettings(models={"vision": "medgemma-4b-it", "agent": "gemma-4-12b-it"})
        check("off: saved local choice → cloud default",
              (s.model_for("vision"), s.model_for("agent")) == ("gemini-2.5-flash",) * 2)
        saved = (gemma_vision._STATE, medgemma._STATE)
        gemma_vision._STATE = medgemma._STATE = None
        check("off: vision loaders refuse (no 8GB load)",
              gemma_vision._load() is None and medgemma._load() is None
              and gemma_vision._STATE is None and medgemma._STATE is None)
        # --- L4 VRAM: the two ~8.6GB vision models swap instead of stacking ---
        torch.cuda.is_available = lambda: True
        real_empty, torch.cuda.empty_cache = torch.cuda.empty_cache, (lambda: None)
        gemma_vision._STATE, medgemma._STATE = None, ("resident",)
        gemma_vision._load()                 # (local models are off → returns before loading)
        swap1 = medgemma._STATE is None
        gemma_vision._STATE, medgemma._STATE = ("resident",), None
        medgemma._load()
        swap2 = gemma_vision._STATE is None
        check("CUDA: loading one Gemma vision model unloads the other", swap1 and swap2)
        torch.cuda.is_available = lambda: False
        gemma_vision._STATE, medgemma._STATE = None, ("resident",)
        gemma_vision._load()
        check("no CUDA (Mac): both may stay resident", medgemma._STATE == ("resident",))
        torch.cuda.is_available, torch.cuda.empty_cache = real_cuda, real_empty
        gemma_vision._STATE, medgemma._STATE = saved
        os.environ.pop("STUDIO_LOCAL_MODELS")
        check("default (local): options include local models",
              "gemma-3-4b-it" in model_options()["vision"])
        check("default (local): saved local choice kept",
              StudioSettings(models={"vision": "gemma-3-4b-it"}).model_for("vision")
              == "gemma-3-4b-it")
    finally:
        torch.backends.mps.is_available, torch.cuda.is_available = real_mps, real_cuda
        if saved_env is None:
            os.environ.pop("STUDIO_LOCAL_MODELS", None)
        else:
            os.environ["STUDIO_LOCAL_MODELS"] = saved_env
    return R


def main():
    results = collect()
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail and not ok else ""))
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    assert not bad, bad


if __name__ == "__main__":
    main()
