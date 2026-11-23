"""Count supported FP32 forward operations independently of pilot timings."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp

from .baselines import PIXEL, ROOT


SCOPES = {
    "vits": "Dense correspondence, depth, and uncertainty prior network",
    "vitb": "Dense correspondence, depth, and uncertainty prior network",
    "deca": "Coarse encoder and posed/neutral FLAME tensor operations",
    "mica": "ArcFace, shape generator, and neutral FLAME tensor operations",
    "uv_map": "UV network, original and mirrored crop",
    "normals": "Normal network, original and mirrored crop",
}


@contextmanager
def count_forward(rows, stage):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        yield
    operations = {str(op): count for op, count in counter.get_flop_counts()["Global"].items()}
    total = counter.get_total_flops()
    if total <= 0 or sum(operations.values()) != total:
        raise RuntimeError(f"Invalid operation counts for {stage}")
    rows.append({"stage": stage, "counted_flops": total, "by_operator": operations})


def realdenseface(model_id, source, rows):
    import numpy as np
    import torch

    upstream = ROOT / "third_party/RealDenseFace"
    sys.path.insert(0, str(upstream))
    from common.config import load_model_config
    from models.realdenseface import RealDenseFace

    config = load_model_config(upstream / f"configs/model/{model_id}.yaml")
    config["dino_pretrained"] = False
    model = RealDenseFace(**config).eval()
    checkpoint = upstream / f"weights/realdenseface/{model_id}.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True, mmap=True), assign=True)
    with np.load(source, allow_pickle=False) as data:
        image = torch.from_numpy(data["network_input"]).float()
    with torch.inference_mode():
        model.build_inference_cache()
        with count_forward(rows, model_id):
            outputs = model(image)
        if not all(torch.isfinite(value).all() for value in outputs):
            raise ValueError("Nonfinite RealDenseFace prior")
    return list(image.shape)


def comparator(model_id, image, scratch, rows):
    from . import baseline_worker

    timing_context = baseline_worker.timed_forward

    @contextmanager
    def counted_forward(device, metadata):
        with timing_context(device, metadata), count_forward(rows, model_id):
            yield

    os.environ["TRUEFACE3D_COMPARATOR_DEVICE"] = "cpu"
    baseline_worker.timed_forward = counted_forward
    (scratch / "native").mkdir(parents=True)
    metadata = {}
    try:
        fit = {"deca": baseline_worker.fit_deca, "mica": baseline_worker.fit_mica}[model_id]
        fit(image, scratch, metadata)
    finally:
        baseline_worker.timed_forward = timing_context
    return [1, 3, *metadata["input_size"]]


def pixel_prior(stage, source, scratch, rows):
    import torch
    from torch.nn import Module

    sys.path.insert(0, str(PIXEL / "src"))
    from .baseline_pixel3dmm import CASE, TORCH_CACHE, _one, _run_upstream

    prepared = scratch / "preprocessed"
    for folder in ("cropped", "seg_og"):
        shutil.copytree(source / folder, prepared / CASE / folder)
    os.environ.update(
        PIXEL3DMM_CODE_BASE=str(PIXEL),
        PIXEL3DMM_PREPROCESSED_DATA=str(prepared),
        PIXEL3DMM_TRACKING_OUTPUT=str(scratch / "tracking"),
        TORCH_HOME=str(TORCH_CACHE),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    from pixel3dmm.lightning.p3dmm_system import system

    original_load_state = system.load_state_dict

    def load_state_with_count(self, *args, **kwargs):
        loaded = original_load_state(self, *args, **kwargs)
        native_forward = self.net.forward

        def counted_forward(*args, **kwargs):
            with count_forward(rows, stage):
                return native_forward(*args, **kwargs)

        self.net.forward = counted_forward
        return loaded

    # The upstream demo hardcodes .cuda() for its module and input tensors.
    # Redirect only inside this isolated profiling process.
    module_cuda, tensor_cuda, available = Module.cuda, torch.Tensor.cuda, torch.cuda.is_available
    try:
        system.load_state_dict = load_state_with_count
        Module.cuda = torch.Tensor.cuda = lambda self, *args, **kwargs: self
        torch.cuda.is_available = lambda: False
        _run_upstream(stage, scratch)
    finally:
        system.load_state_dict = original_load_state
        Module.cuda, torch.Tensor.cuda, torch.cuda.is_available = module_cuda, tensor_cuda, available
    _one(prepared / CASE / "p3dmm" / stage, "00000.png")
    if len(rows) != 2:
        raise RuntimeError(f"Expected two Pixel3DMM forwards, counted {len(rows)}")
    return [1, 1, 512, 512, 3]


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=SCOPES)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    scratch = Path(mkdtemp(prefix=output.stem + "-work-", dir=output.parent))
    rows = []
    if args.model in {"vits", "vitb"}:
        input_shape = realdenseface(args.model, source, rows)
    elif args.model in {"deca", "mica"}:
        input_shape = comparator(args.model, source, scratch, rows)
    else:
        input_shape = pixel_prior(args.model, source, scratch, rows)
    result = {
        "model_stage": args.model, "scope": SCOPES[args.model], "input_shape": input_shape,
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "backend": "CPU FP32 (math SDPA)", "torch": str(torch.__version__),
        "convention": "2 FLOPs per multiply-accumulate; PyTorch FlopCounterMode operator subset",
        "counted_flops": sum(row["counted_flops"] for row in rows), "forwards": rows,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"{args.model}  {result['counted_flops'] / 1e9:.3f} counted GFLOPs  {output}", flush=True)


if __name__ == "__main__":
    main()
