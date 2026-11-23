"""Inspect persisted arrays and create the Chapter 1 notebook figures."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .benchmark import _active_artifact, _score_values, pair_variants


def inspect_arrays(bundle, keys=None):
    with np.load(bundle, allow_pickle=False) as arrays:
        if keys is None:
            keys = arrays.files
        rows = []
        for name in keys:
            value = arrays[name]
            numeric = value.dtype.kind in "biuf"
            sample = value.reshape(-1)[:6].tolist()
            finite = value[np.isfinite(value)] if numeric else []
            rows.append({
                "array": name, "shape": " × ".join(map(str, value.shape)) or "scalar",
                "dtype": str(value.dtype), "min": float(np.min(finite)) if len(finite) else None,
                "max": float(np.max(finite)) if len(finite) else None,
                "finite": f"{len(finite)}/{value.size}" if numeric else None, "first values": sample,
            })
        return rows


def table_markdown(rows):
    rows = list(rows)
    if not rows:
        return "No rows available."
    keys = list(rows[0])

    def label(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "&#124;").replace("\n", " ")

    return "\n".join([
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
        *["| " + " | ".join(label(row.get(key)) for key in keys) + " |" for row in rows],
    ])


def plot_cases(cases, records, title):
    import matplotlib.pyplot as plt
    from PIL import Image
    import trimesh

    selected = records[:5]
    fig = plt.figure(figsize=(4 * max(len(selected), 1), 4), layout="constrained")
    fig.suptitle(title)
    for index, row in enumerate(selected, 1):
        folder = Path(row.case_dir)
        render = next((path for path in (folder / "fit/render.png", folder / "render.png")
                       if path.is_file()), None)
        if render is not None:
            ax = fig.add_subplot(1, len(selected), index)
            with Image.open(render) as image:
                ax.imshow(image)
        else:
            mesh_path = next((path for path in (folder / "neutral.obj", folder / "fit/neutral.obj")
                              if path.is_file()), None)
            if mesh_path is not None:
                ax = fig.add_subplot(1, len(selected), index, projection="3d")
                mesh = trimesh.load_mesh(mesh_path, process=False)
                xyz = mesh.vertices
                ax.plot_trisurf(*xyz.T, triangles=mesh.faces, color="#b0bcc4", linewidth=.1)
                ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1e-6))
                ax.view_init(12, -75)
            else:
                ax = fig.add_subplot(1, len(selected), index)
                ax.text(.5, .5, row.error_stage or row.fit_status, ha="center", va="center", wrap=True)
        ax.set_title(row.model_id)
        ax.set_axis_off()
    if not selected:
        fig.text(.5, .45, "No recorded model attempts", ha="center")
    return fig


def plot_case(case, run):
    return plot_cases([case], [run], case.image_key)


def accuracy_rows(records):
    rows = []
    for record in records:
        score = _score_values(record)
        rows.append({"model": record.model_id, "image": Path(record.image_key).name,
                     "variant": record.variant_id, "fit": record.fit_status, "score": record.score_status,
                     "mean (mm)": score.get("mean_mm"), "median (mm)": score.get("median_mm"),
                     "std (mm)": score.get("std_mm"), "p95 (mm)": score.get("p95_mm"),
                     "≤ 2.5 mm": score.get("within_2_5_mm"), "protocol": score.get("protocol")})
    return rows


def cost_rows(records):
    def mib(value):
        return value / (1024 ** 2) if value is not None else None

    return [
        {"model": row.model_id, "condition": row.variant_id, "device": row.cost.get("device"),
         "runtime": row.runtime_id, "timing scope": row.timing_scope,
         "observed (s)": row.cost.get("seconds"), "coarse forward (s)": row.cost.get("inference_seconds"),
         "CLI wall (s)": row.cost.get("cli_wall_seconds"),
         "peak allocated (MiB)": mib(row.cost.get("peak_allocated_bytes")),
         "peak reserved (MiB)": mib(row.cost.get("peak_reserved_bytes")),
         "detector": row.cost.get("detector"), "memory scope": row.cost.get("memory_scope")}
        for row in records
    ]


def flop_rows(directory):
    """Load scoped operator counts for the same original pilot portrait."""
    profiles = {
        "vits": ("vits",), "vitb": ("vitb",), "deca": ("deca",),
        "pixel3dmm": ("uv_map", "normals"), "mica": ("mica",),
    }
    rows = []
    for model, stages in profiles.items():
        paths = [Path(directory) / f"{stage}.json" for stage in stages]
        parts = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.is_file()]
        complete = len(parts) == len(stages)
        rows.append({
            "model": model,
            "counted GFLOPs": sum(part["counted_flops"] for part in parts) / 1e9 if complete else None,
            "input tensor": " + ".join("×".join(map(str, part["input_shape"])) for part in parts),
            "network forwards": sum(len(part["forwards"]) for part in parts),
            "profiled stages": " + ".join(part["scope"] for part in parts),
            "profile status": "measured" if complete else "incomplete",
        })
    return rows


def plot_ced(records):
    import matplotlib.pyplot as plt
    by_model = defaultdict(dict)
    for row in records:
        if row.score_file:
            score = _score_values(row)
            distances = Path(row.score_file).with_name("distances.npz")
            if distances.is_file():
                by_model[(row.model_id, score["protocol"])][row.image_key] = distances
    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    common = set.intersection(*(set(value) for value in by_model.values())) if by_model else set()
    protocols = {protocol for _, protocol in by_model}
    if common and len(protocols) == 1:
        for (model, protocol), data in by_model.items():
            arrays = []
            for image in sorted(common):
                with np.load(data[image], allow_pickle=False) as loaded:
                    arrays.append(loaded["distances_mm"])
            x = np.sort(np.concatenate(arrays))
            axis.plot(x, np.arange(1, 1 + len(x)) / len(x), label=model)
        axis.legend()
        axis.set_title(f"{protocol} · {len(common)} common scored inputs")
    else:
        axis.text(.5, .5, "No common qualified scan scores", transform=axis.transAxes, ha="center")
    axis.set(xlabel="Scan point → fitted surface (mm)", ylabel="Cumulative proportion")
    return fig


def plot_cost(records):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in records:
        if row.fit_status == "success":
            key = row.runtime_id, row.timing_scope
            groups[key].append(row)
    groups = groups or {("No fitted rows", ""): []}
    fig, axes = plt.subplots(len(groups), 2, squeeze=False,
                             figsize=(11, 3.5 * len(groups)), layout="constrained")
    for (runtime, scope), rows, (time_axis, memory_axis) in zip(groups, groups.values(), axes):
        names = [f"{row.model_id} · {row.variant_id}" for row in rows]
        seconds = [row.cost.get("seconds", row.cost.get("cli_wall_seconds", np.nan)) for row in rows]
        memory = []
        for row in rows:
            allocated = row.cost.get("peak_allocated_bytes", row.cost.get("max_memory_allocated"))
            memory.append(allocated / (1024 ** 2) if allocated is not None else np.nan)
        time_axis.set_title(f"{runtime}\n{scope}", fontsize=9)
        memory_axis.set_title(runtime, fontsize=9)
        for ax, values, label in [(time_axis, seconds, "Observed runtime (s)"),
                                  (memory_axis, memory, "Framework peak allocation (MiB)")]:
            ax.bar(range(len(names)), values, color="#315f86")
            ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
            ax.set_ylabel(label)
            if not any(np.isfinite(values)):
                ax.text(.5, .5, "No observations", transform=ax.transAxes, ha="center")
    return fig


def plot_degradation(pairs):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, prefix, xlabel in zip(axes, ("face-", "blur-"), ("Face width (px)", "Blur setting")):
        groups = defaultdict(list)
        for row in pairs:
            if row["condition"].startswith(prefix) and row["delta_mm"] is not None:
                x = row["face_width_px"] if prefix == "face-" else {"blur-mild": 1, "blur-heavy": 2}[row["condition"]]
                groups[(row["model"], row["subject"])].append((x, row["delta_mm"]))
        for (model, subject), values in groups.items():
            x, y = zip(*sorted(values))
            axis.plot(x, y, "o-", label=f"{model} · {subject}")
        axis.axhline(0, color="#aaa", linewidth=1)
        axis.set(xlabel=xlabel, ylabel="Degraded − original (mm)")
        if prefix == "blur-":
            axis.set_xticks([1, 2], ["Mild", "Heavy"])
        if groups:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "Paired scan scores unavailable", transform=axis.transAxes, ha="center")
    return fig


def plot_coverage(summaries):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    positions = np.arange(len(summaries))
    for offset, key, color in [(-.23, "requested", "#b6c3ce"), (0, "fitted", "#315f86"), (.23, "scored", "#3b8471")]:
        axis.bar(positions + offset, [row[key] for row in summaries], width=.23, label=key, color=color)
    labels = [row["model"] + "\n" + row["variant"] for row in summaries]
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Cases")
    axis.legend()
    return fig


def compare_prior_drift(original, variant, variant_metadata):
    """Compare ViT priors on shared valid in-frame vertices in source pixels."""
    required = {"vertex_coord", "vertex_depth", "vertex_coord_log_var",
                "vertex_depth_log_var", "input_size_hw"}
    with np.load(original, allow_pickle=False) as a, np.load(variant, allow_pickle=False) as b:
        if not required.issubset(a.files) or not required.issubset(b.files):
            return None
        xy_a, xy_b = a["vertex_coord"], b["vertex_coord"]
        depth_a, depth_b = a["vertex_depth"].reshape(-1), b["vertex_depth"].reshape(-1)
        valid = np.ones(len(xy_a), dtype=bool)
        for arrays, xy in ((a, xy_a), (b, xy_b)):
            height, width = arrays["input_size_hw"]
            valid &= (xy >= 0).all(axis=1) & (xy < [width, height]).all(axis=1)
            for key in ("vertex_coord_log_var", "vertex_depth_log_var"):
                log_var = arrays[key].reshape(-1)
                valid &= np.isfinite(log_var) & (log_var < 30.0)
        valid &= np.isfinite(depth_a) & np.isfinite(depth_b)
        frame = np.asarray(variant_metadata.get("variant_to_original", np.eye(3)))
        if not valid.any():
            return {"shared_valid": 0, "xy_median_px": None, "relative_depth_median_abs": None}
        same_frame_xy = xy_b[valid] @ frame[:2, :2].T + frame[:2, 2]
        drift = np.linalg.norm(same_frame_xy - xy_a[valid], axis=1)
        return {"shared_valid": int(np.sum(valid)), "xy_median_px": float(np.median(drift)),
                "relative_depth_median_abs": float(np.median(np.abs(depth_b[valid] - depth_a[valid])))}


def paired_neutral_drift(records, originals):
    """Measure prediction changes within a model with unchanged neutral mesh indexing."""
    from .evaluation import load_surface

    original_rows = {(row.model_id, row.image_key): row for row in originals}
    paired = {(row["model"], row["image"], row["condition"]): row["pair_status"]
              for row in pair_variants(records, originals)}
    outputs = []
    for row in records:
        key = row.model_id, row.image_key, row.variant_id
        if paired[key] not in {"scored", "score_unavailable"}:
            continue
        first = original_rows[key[:2]]
        before = _active_artifact(Path(first.case_dir), "neutral.obj")
        after = _active_artifact(Path(row.case_dir), "neutral.obj")
        if before is None or after is None:
            continue
        clean, degraded = load_surface(before), load_surface(after)
        if clean.vertices.shape != degraded.vertices.shape or not np.array_equal(clean.faces, degraded.faces):
            continue
        displacement = np.linalg.norm(degraded.vertices - clean.vertices, axis=1)
        outputs.append({"model": row.model_id, "condition": row.variant_id, "image": Path(row.image_key).name,
                        "corresponding vertices": len(displacement),
                        "mean change (native units)": float(displacement.mean()),
                        "p95 change (native units)": float(np.percentile(displacement, 95))})
    return outputs
