"""Create reproducible degradation controls from an RGB photograph."""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Degradation:
    divisor: int = 1
    blur: float = 0.0
    noise: float = 0.0
    jpeg_quality: int | None = None


def make_variant(image, spec, seed):
    if spec.divisor < 1 or not np.isfinite([spec.blur, spec.noise]).all() or min(spec.blur, spec.noise) < 0:
        raise ValueError("Invalid degradation settings")
    if spec.jpeg_quality is not None and not 1 <= spec.jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    raw = image.convert("RGB")
    original_size = raw.size
    if spec.blur:
        raw = raw.filter(ImageFilter.GaussianBlur(spec.blur))
    if spec.divisor != 1:
        raw = raw.resize(tuple(max(1, n // spec.divisor) for n in raw.size), Image.Resampling.BOX)
    if spec.noise:
        pixels = np.asarray(raw, dtype=np.float32)
        noisy = pixels + np.random.default_rng(seed).normal(0, spec.noise, pixels.shape)
        raw = Image.fromarray(np.clip(np.rint(noisy), 0, 255).astype(np.uint8))
    if spec.jpeg_quality is not None:
        buf = BytesIO()
        raw.save(buf, format="JPEG", quality=spec.jpeg_quality)
        buf.seek(0)
        with Image.open(buf) as decoded:
            raw = decoded.convert("RGB")
    return raw, {"original_size": original_size, "variant_size": raw.size, "seed": seed}


def bicubic_control(variant, target_size):
    return variant.resize(target_size, Image.Resampling.BICUBIC)


def make_face_width_variant(image, box, target_px):
    """Downsample the full RGB frame to an annotated face-box width."""
    box = np.asarray(box, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0]
            or box[3] <= box[1]):
        raise ValueError("Expected an ordered finite face box")
    width = box[2] - box[0]
    if not np.isfinite(target_px) or not 0 < target_px <= width:
        raise ValueError("Target face width must be positive and no larger than the original")
    size = tuple(max(1, round(length * target_px / width)) for length in image.size)
    sx, sy = (out / source for out, source in zip(size, image.size))
    transform = np.diag([sx, sy, 1.0])
    variant = image.convert("RGB").resize(size, Image.Resampling.BOX)
    return variant, {
        "original_size": image.size,
        "variant_size": size,
        "original_face_width_px": float(width),
        "target_face_width_px": float(target_px),
        "achieved_face_width_px": float(width * sx),
        "face_box": (box * [sx, sy, sx, sy]).tolist(),
        "original_to_variant": transform.tolist(),
        "variant_to_original": np.linalg.inv(transform).tolist(),
        "coordinate_convention": "pixel edges",
    }


COMPOUND_REFERENCE_FACE_PX = 460.0


def make_compound_variant(image, box, *, target_px=48, blur_px=1.5, noise=8.0, jpeg_quality=30, seed=0):
    """Blur -> downsample to a face width -> Gaussian noise -> JPEG, all deterministic.

    ``blur_px`` is a Gaussian sigma for a face 460 px wide at original resolution and is
    scaled in proportion to the annotated face-box width. ``noise`` is a sigma on the 0-255 scale.
    """
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0]:
        raise ValueError("Expected an ordered finite face box")
    width = float(box[2] - box[0])
    sigma = blur_px * width / COMPOUND_REFERENCE_FACE_PX
    blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(sigma))
    small, meta = make_face_width_variant(blurred, box, target_px)
    final, extra = make_variant(small, Degradation(noise=noise, jpeg_quality=jpeg_quality), seed)
    meta.update(variant_size=final.size, seed=seed, blur_sigma_px=sigma, noise_sigma=noise,
                jpeg_quality=jpeg_quality, original_size=image.size)
    return final, meta
