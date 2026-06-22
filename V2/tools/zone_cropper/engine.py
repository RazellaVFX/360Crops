from __future__ import annotations

import json
import importlib
import importlib.util
import math
import os
import queue as _queue
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
if importlib.util.find_spec("imagecodecs") is not None:
    imagecodecs = importlib.import_module("imagecodecs")
else:
    imagecodecs = None  # type: ignore
try:
    import torch
except ImportError:
    torch = None  # type: ignore
try:
    from torchvision.io import encode_jpeg as torchvision_encode_jpeg
except ImportError:
    torchvision_encode_jpeg = None  # type: ignore

try:
    from equilib.equi2pers.numpy import (
        convert_grid as convert_grid_numpy,
        create_rotation_matrices as create_rotation_matrices_numpy,
        matmul as matmul_numpy,
        prep_matrices as prep_matrices_numpy,
    )
    from equilib.equi2pers.torch import (
        convert_grid as convert_grid_torch,
        create_rotation_matrices as create_rotation_matrices_torch,
        matmul as matmul_torch,
        prep_matrices as prep_matrices_torch,
    )
    from equilib.grid_sample import numpy_grid_sample, torch_grid_sample
except ImportError:
    # Local clone layout: repository root folder named `equilib`.
    from equilib.equilib.equi2pers.numpy import (
        convert_grid as convert_grid_numpy,
        create_rotation_matrices as create_rotation_matrices_numpy,
        matmul as matmul_numpy,
        prep_matrices as prep_matrices_numpy,
    )
    from equilib.equilib.equi2pers.torch import (
        convert_grid as convert_grid_torch,
        create_rotation_matrices as create_rotation_matrices_torch,
        matmul as matmul_torch,
        prep_matrices as prep_matrices_torch,
    )
    from equilib.equilib.grid_sample import numpy_grid_sample, torch_grid_sample

from .config import ProgressInfo, SUPPORTED_EXTENSIONS, SequenceJobConfig
from .layout import Zone, compute_fov_y_deg, generate_zone_layout

ProgressCallback = Callable[[ProgressInfo], None]


def list_sequence_frames(input_dir: Path, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> List[Path]:
    exts = {e.lower() for e in extensions}
    frames = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(frames, key=lambda p: p.name.lower())


def load_equi_frame(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img)
    return np.transpose(arr, (2, 0, 1))


_PREFETCH_DONE = object()


def _prefetch_frames(
    frames: List[Path],
    out_queue: "_queue.Queue[object]",
    stop_event: Optional[Event],
) -> None:
    """Background thread: reads frames ahead and puts them into out_queue.

    Each successful item is (frame_path, ndarray). Errors are put as
    (frame_path, Exception). Signals end with the _PREFETCH_DONE sentinel.
    """
    for frame_path in frames:
        if stop_event and stop_event.is_set():
            break
        try:
            arr = load_equi_frame(frame_path)
            out_queue.put((frame_path, arr))
        except Exception as exc:
            out_queue.put((frame_path, exc))
    out_queue.put(_PREFETCH_DONE)


def _jpeg_params(compression_strength: float) -> tuple[int, int]:
    jpeg_quality = int(round(95 - compression_strength * 65))
    jpeg_quality = max(30, min(95, jpeg_quality))
    jpeg_subsampling = 0 if compression_strength <= 0.5 else 2
    return jpeg_quality, jpeg_subsampling


def _try_encode_jpeg_cuda_batch(crops_batch, *, compression_strength: float) -> Optional[list[bytes]]:
    """Encode a batch of CHW uint8 CUDA crops via nvJPEG when available."""
    if torch is None or torchvision_encode_jpeg is None:
        return None
    if not torch.is_tensor(crops_batch):
        return None
    if crops_batch.device.type != "cuda" or crops_batch.dtype != torch.uint8:
        return None

    jpeg_quality, _jpeg_subsampling = _jpeg_params(compression_strength)
    try:
        encoded = torchvision_encode_jpeg(list(crops_batch), quality=jpeg_quality)
    except Exception:
        return None

    out: list[bytes] = []
    for item in encoded:
        item_cpu = item.detach().cpu()
        out.append(item_cpu.numpy().tobytes())
    return out


def _encode_with_imagecodecs(crop_chw, *, suffix: str, compression_strength: float) -> Optional[bytes]:
    if imagecodecs is None:
        return None
    if torch is not None and torch.is_tensor(crop_chw):
        crop_chw = crop_chw.detach().cpu().numpy()
    if isinstance(crop_chw, np.ndarray) and crop_chw.ndim == 3 and crop_chw.shape[0] in {1, 3, 4}:
        hwc = np.transpose(crop_chw, (1, 2, 0))
    else:
        hwc = crop_chw
    if not isinstance(hwc, np.ndarray):
        return None
    hwc = np.ascontiguousarray(hwc)
    suffix_l = suffix.lower()

    if suffix_l in {".jpg", ".jpeg"}:
        jpeg_quality, _jpeg_subsampling = _jpeg_params(compression_strength)
        attempts = (
            {"level": jpeg_quality},
            {"quality": jpeg_quality},
            {},
        )
        for kwargs in attempts:
            try:
                encoded = imagecodecs.jpeg_encode(hwc, **kwargs)
                return bytes(encoded) if isinstance(encoded, (bytes, bytearray)) else encoded.tobytes()
            except Exception:
                continue
        return None

    if suffix_l == ".png":
        png_compress_level = int(round(compression_strength * 9))
        png_compress_level = max(0, min(9, png_compress_level))
        attempts = (
            {"level": png_compress_level},
            {"compression": png_compress_level},
            {},
        )
        for kwargs in attempts:
            try:
                encoded = imagecodecs.png_encode(hwc, **kwargs)
                return bytes(encoded) if isinstance(encoded, (bytes, bytearray)) else encoded.tobytes()
            except Exception:
                continue
        return None

    return None


def _save_crop(crop_chw, out_path: Path, *, compression_strength: float, use_imagecodecs: bool = False) -> None:
    if isinstance(crop_chw, memoryview):
        out_path.write_bytes(crop_chw.tobytes())
        return
    if isinstance(crop_chw, (bytes, bytearray)):
        out_path.write_bytes(bytes(crop_chw))
        return

    if use_imagecodecs:
        encoded_ic = _encode_with_imagecodecs(
            crop_chw,
            suffix=out_path.suffix.lower(),
            compression_strength=compression_strength,
        )
        if encoded_ic is not None:
            out_path.write_bytes(encoded_ic)
            return

    if torch is not None and torch.is_tensor(crop_chw):
        crop_chw = crop_chw.detach().cpu().numpy()
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        jpeg_quality, jpeg_subsampling = _jpeg_params(compression_strength)
        hwc = np.transpose(crop_chw, (1, 2, 0))
        image = Image.fromarray(hwc)
        image.save(out_path, quality=jpeg_quality, subsampling=jpeg_subsampling)
        return
    if out_path.suffix.lower() == ".png":
        png_compress_level = int(round(compression_strength * 9))
        png_compress_level = max(0, min(9, png_compress_level))
        hwc = np.transpose(crop_chw, (1, 2, 0))
        image = Image.fromarray(hwc)
        image.save(out_path, compress_level=png_compress_level)
        return
    hwc = np.transpose(crop_chw, (1, 2, 0))
    image = Image.fromarray(hwc)
    image.save(out_path)


def _chunked(seq: Sequence[Path], chunk_size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def _estimate_eta(start_ts: float, completed: int, total: int) -> Optional[float]:
    if completed <= 0:
        return None
    elapsed = time.perf_counter() - start_ts
    remaining = max(total - completed, 0)
    return (elapsed / completed) * remaining


def _round_half_away_from_zero(value: float) -> int:
    """Round to nearest integer using classic half-up semantics."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(abs(value) + 0.5))


def _format_pitch_label(value: float) -> str:
    rounded = _round_half_away_from_zero(value)
    if rounded == 0 and value < 0:
        return "-0"
    return f"{rounded}"


def _zone_id_tag(zone: Zone) -> str:
    zone_id = int(zone.zone_id)
    return f"Zone_{zone_id:03d}"


def _zone_folder_tag(zone: Zone) -> str:
    zone_id = int(zone.zone_id)
    yaw_i = _round_half_away_from_zero(zone.yaw_deg)
    pitch_s = _format_pitch_label(zone.pitch_deg)
    return f"Zone_{zone_id:03d}_Y{yaw_i:03d}_P{pitch_s}"


def _build_numpy_grid(
    *,
    rots: Sequence[dict[str, float]],
    source_height: int,
    source_width: int,
    crop_height: int,
    crop_width: int,
    fov_x_deg: float,
    z_down: bool,
) -> np.ndarray:
    dtype = np.dtype(np.float32)
    batch = len(rots)
    m, G = prep_matrices_numpy(
        height=crop_height,
        width=crop_width,
        batch=batch,
        fov_x=fov_x_deg,
        skew=0.0,
        dtype=dtype,
    )
    R = create_rotation_matrices_numpy(rots=list(rots), z_down=z_down, dtype=dtype)
    M = matmul_numpy(m, G, R)
    return convert_grid_numpy(M=M, h_equi=source_height, w_equi=source_width, method="robust")


def _build_torch_grid(
    *,
    rots: Sequence[dict[str, float]],
    source_height: int,
    source_width: int,
    crop_height: int,
    crop_width: int,
    fov_x_deg: float,
    z_down: bool,
    grid_dtype,
    device,
):
    tmp_device = torch.device("cpu")
    batch = len(rots)
    m, G = prep_matrices_torch(
        height=crop_height,
        width=crop_width,
        batch=batch,
        fov_x=fov_x_deg,
        skew=0.0,
        dtype=grid_dtype,
        device=tmp_device,
    )
    R = create_rotation_matrices_torch(rots=list(rots), z_down=z_down, dtype=grid_dtype, device=tmp_device)
    M = matmul_torch(m, G, R)
    grid = convert_grid_torch(M=M, h_equi=source_height, w_equi=source_width, method="robust")
    if device is not None:
        grid = grid.to(device=device)
    return grid


def _submit_save(
    *,
    save_pool: ThreadPoolExecutor,
    pending_writes: list[tuple[Future[None], int]],
    crop,
    out_path: Path,
    compression_strength: float,
    use_imagecodecs: bool = False,
    stop_event: Optional[Event] = None,
    max_pending: int = 64,
) -> None:
    if isinstance(crop, memoryview):
        crop = crop.tobytes()
    elif isinstance(crop, bytearray):
        crop = bytes(crop)
    elif torch is not None and torch.is_tensor(crop):
        crop = crop.detach().cpu().numpy()
    elif isinstance(crop, np.ndarray):
        crop = np.array(crop, copy=True)

    if isinstance(crop, bytes):
        crop_bytes = len(crop)
    else:
        crop_bytes = int(getattr(crop, "nbytes", 0))

    def drain_completed() -> None:
        still_pending: list[tuple[Future[None], int]] = []
        for future, bytes_size in pending_writes:
            if future.done():
                future.result()
            else:
                still_pending.append((future, bytes_size))
        pending_writes[:] = still_pending
        return None

    drain_completed()
    while pending_writes and len(pending_writes) >= max_pending:
        if stop_event and stop_event.is_set():
            return
        future, _bytes_size = pending_writes.pop(0)
        future.result()

    future = save_pool.submit(
        _save_crop,
        crop,
        out_path,
        compression_strength=compression_strength,
        use_imagecodecs=use_imagecodecs,
    )
    pending_writes.append((future, crop_bytes))


def _wait_pending_writes(pending_writes: list[tuple[Future[None], int]]) -> None:
    for future, _bytes_size in pending_writes:
        future.result()


def _cancel_pending_writes(pending_writes: list[tuple[Future[None], int]]) -> None:
    for future, _bytes_size in pending_writes:
        future.cancel()


def _stage_add(stats: Optional[dict], key: str, dt: float) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0.0) + dt


def _maybe_sync(enabled: bool) -> None:
    # Needed so GPU stage timers measure actual CUDA work, not async launch only.
    if enabled and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_frame_range(frame_count: int, frame_start: int, frame_end: int) -> tuple[int, int]:
    if frame_count <= 0:
        raise ValueError("No frames in sequence.")
    start = max(0, int(frame_start))
    end = frame_count - 1 if int(frame_end) < 0 else int(frame_end)
    end = min(end, frame_count - 1)
    if start > end:
        raise ValueError(
            f"Invalid frame range {start}..{end} for sequence with {frame_count} frame(s)."
        )
    return start, end


def process_sequence(
    config: SequenceJobConfig,
    *,
    stop_event: Optional[Event] = None,
    progress_cb: Optional[ProgressCallback] = None,
    stage_stats: Optional[dict] = None,
) -> Tuple[List[Zone], int]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    profile = stage_stats is not None
    debug_timings = os.environ.get("ZONE_CROPPER_DEBUG_TIMINGS", "").strip().lower() in {"1", "true", "yes", "on"}

    use_gpu = config.use_torch_gpu
    if use_gpu:
        if torch is None:
            raise ValueError("Torch is not installed, cannot enable GPU mode.")
        if not torch.cuda.is_available():
            raise ValueError("GPU mode is enabled, but CUDA is not available.")
        device = torch.device("cuda")
        backend = "native" if config.use_native_torch_backend else "pure"

    frames_all = list_sequence_frames(config.input_dir)
    if not frames_all:
        raise ValueError(f"No supported frames found in: {config.input_dir}")

    source_frame_start, source_frame_end = _resolve_frame_range(
        len(frames_all), config.frame_start, config.frame_end
    )
    frames = frames_all[source_frame_start : source_frame_end + 1]

    zones = generate_zone_layout(
        layout=config.layout,
        fov_x_deg=config.fov_x_deg,
        crop_width=config.crop_width,
        crop_height=config.crop_height,
    )
    rots = [z.to_rots() for z in zones]

    sample_frame = load_equi_frame(frames[0])
    source_height = int(sample_frame.shape[1])
    source_width = int(sample_frame.shape[2])
    numpy_grid: Optional[np.ndarray] = None
    torch_grid = None
    if use_gpu:
        assert torch is not None
        torch_grid = _build_torch_grid(
            rots=rots,
            source_height=source_height,
            source_width=source_width,
            crop_height=config.crop_height,
            crop_width=config.crop_width,
            fov_x_deg=config.fov_x_deg,
            z_down=config.z_down,
            grid_dtype=torch.float32,
            device=device,
        ).to(dtype=torch.float32)
    else:
        numpy_grid = _build_numpy_grid(
            rots=rots,
            source_height=source_height,
            source_width=source_width,
            crop_height=config.crop_height,
            crop_width=config.crop_width,
            fov_x_deg=config.fov_x_deg,
            z_down=config.z_down,
        )

    start_ts = time.perf_counter()
    total_frames = len(frames)
    written = 0
    fov_y = compute_fov_y_deg(config.fov_x_deg, config.crop_width, config.crop_height)

    # Output layout: one folder per zone containing crops from all frames.
    zone_dirs: dict[int, Path] = {}
    for z in zones:
        zone_dir_name = _zone_folder_tag(z)
        zone_dir = config.output_dir / zone_dir_name
        zone_dir.mkdir(parents=True, exist_ok=True)
        zone_dirs[z.zone_id] = zone_dir

    manifest = {
        "total_frames": total_frames,
        "source_frame_start": source_frame_start,
        "source_frame_end": source_frame_end,
        "source_total_frames": len(frames_all),
        "crop_size": {"width": config.crop_width, "height": config.crop_height},
        "fov_x_deg": config.fov_x_deg,
        "fov_y_deg": fov_y,
        "compression_strength": config.compression_strength,
        "use_imagecodecs": config.use_imagecodecs,
        "writer_threads": config.writer_threads,
        "gpu_zone_batch_size": config.gpu_zone_batch_size,
        "use_native_torch_backend": config.use_native_torch_backend,
        "projection_precomputed": True,
        "overlap": config.layout.overlap,
        "zones": [
            {
                "zone_id": z.zone_id,
                "yaw_deg": z.yaw_deg,
                "pitch_deg": z.pitch_deg,
                "roll_deg": z.roll_deg,
                "folder": zone_dirs[z.zone_id].name,
            }
            for z in zones
        ],
    }
    with (config.output_dir / "zones_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Prefetch frames from disk in a background thread while GPU/CPU processes.
    # Queue size = chunk_size so the reader stays at most that many frames ahead.
    prefetch_q: _queue.Queue = _queue.Queue(maxsize=max(2, config.chunk_size))
    prefetch_thread = threading.Thread(
        target=_prefetch_frames,
        args=(frames, prefetch_q, stop_event),
        daemon=True,
    )
    prefetch_thread.start()

    save_pool = ThreadPoolExecutor(max_workers=config.writer_threads)
    pending_writes: list[tuple[Future[None], int]] = []
    zone_batch_size = max(1, config.gpu_zone_batch_size) if use_gpu else len(zones)
    stopped_early = False
    try:
        for global_idx in range(total_frames):
            # Block only if prefetch is not yet ahead; ideally returns immediately.
            _t_read = time.perf_counter()
            item = prefetch_q.get()
            frame_wait_s = time.perf_counter() - _t_read
            _stage_add(stage_stats, "read", frame_wait_s)
            if debug_timings:
                print(
                    f"[timing] frame={global_idx + 1}/{total_frames} "
                    f"wait_for_frame_ms={frame_wait_s * 1000.0:.2f}"
                )

            if item is _PREFETCH_DONE:
                break

            frame_path, equi_or_err = item
            if isinstance(equi_or_err, Exception):
                raise equi_or_err
            equi: np.ndarray = equi_or_err

            if stop_event and stop_event.is_set():
                stopped_early = True
                break

            if int(equi.shape[1]) != source_height or int(equi.shape[2]) != source_width:
                raise ValueError(
                    f"Frame {frame_path.name} size {equi.shape[2]}x{equi.shape[1]} "
                    f"does not match first frame size {source_width}x{source_height}."
                )

            if use_gpu:
                assert torch is not None
                assert torch_grid is not None
                _t_dev = time.perf_counter()
                equi_input = torch.from_numpy(np.ascontiguousarray(equi)).to(device=device, non_blocking=True)
                _maybe_sync(profile)
                _stage_add(stage_stats, "to_dev", time.perf_counter() - _t_dev)
                for start_idx in range(0, len(zones), zone_batch_size):
                    if stop_event and stop_event.is_set():
                        stopped_early = True
                        break
                    end_idx = min(start_idx + zone_batch_size, len(zones))
                    grid_batch = torch_grid[start_idx:end_idx]
                    batch_len = end_idx - start_idx
                    _t_sample = time.perf_counter()
                    equi_batch = equi_input.unsqueeze(0).expand(batch_len, -1, -1, -1).to(dtype=torch.float32)
                    if backend == "native":
                        crops_batch = torch_grid_sample(
                            img=equi_batch,
                            grid=grid_batch,
                            out=None,
                            mode=config.mode,
                            backend="native",
                        )
                    else:
                        out_ref = torch.empty(
                            (batch_len, equi_batch.shape[1], config.crop_height, config.crop_width),
                            dtype=equi_batch.dtype,
                            device=equi_batch.device,
                        )
                        crops_batch = torch_grid_sample(
                            img=equi_batch,
                            grid=grid_batch,
                            out=out_ref,
                            mode=config.mode,
                            backend="pure",
                        )
                    _maybe_sync(profile)
                    _stage_add(stage_stats, "sample", time.perf_counter() - _t_sample)

                    _t_post = time.perf_counter()
                    if config.clip_output:
                        crops_batch = torch.clamp(crops_batch, 0.0, 255.0)
                    crops_batch = crops_batch.to(dtype=torch.uint8)
                    _maybe_sync(profile)
                    _stage_add(stage_stats, "post", time.perf_counter() - _t_post)

                    encoded_jpeg_batch: Optional[list[bytes]] = None
                    if (
                        not config.use_imagecodecs
                        and config.image_ext.lower() in {".jpg", ".jpeg"}
                    ):
                        # Fast path: nvJPEG on CUDA, then transfer compressed bytes only.
                        # Explicitly disabled when imagecodecs checkbox is enabled.
                        encoded_jpeg_batch = _try_encode_jpeg_cuda_batch(
                            crops_batch,
                            compression_strength=config.compression_strength,
                        )
                        _maybe_sync(profile)

                    if encoded_jpeg_batch is None:
                        # Fallback path for PNG/CPU or when nvJPEG is unavailable.
                        crops_batch_np = crops_batch.detach().cpu().numpy()

                    for local_idx, zone in enumerate(zones[start_idx:end_idx]):
                        if stop_event and stop_event.is_set():
                            stopped_early = True
                            break
                        zone_i = start_idx + local_idx
                        out_name = f"{_zone_id_tag(zone)}_{frame_path.stem}{config.image_ext}"
                        _t_save = time.perf_counter()
                        _submit_save(
                            save_pool=save_pool,
                            pending_writes=pending_writes,
                            crop=(
                                encoded_jpeg_batch[local_idx]
                                if encoded_jpeg_batch is not None
                                else crops_batch_np[local_idx]
                            ),
                            out_path=zone_dirs[zone.zone_id] / out_name,
                            compression_strength=config.compression_strength,
                            use_imagecodecs=config.use_imagecodecs,
                            stop_event=stop_event,
                        )
                        _stage_add(stage_stats, "save", time.perf_counter() - _t_save)
                        written += 1

                        if progress_cb:
                            progress_cb(
                                ProgressInfo(
                                    total_frames=total_frames,
                                    frame_index=global_idx + 1,
                                    frame_name=frame_path.name,
                                    total_zones=len(zones),
                                    zone_index=zone_i + 1,
                                    eta_seconds=_estimate_eta(start_ts, global_idx + 1, total_frames),
                                    message=f"Frame {global_idx + 1}/{total_frames}, zone {zone_i + 1}/{len(zones)}",
                                )
                            )
                    if stopped_early:
                        break
                if stopped_early:
                    break
            else:
                assert numpy_grid is not None
                _t_dev = time.perf_counter()
                equi_batch = np.repeat(equi[np.newaxis, ...], len(zones), axis=0)
                out_ref_np = np.empty(
                    (len(zones), equi_batch.shape[1], config.crop_height, config.crop_width),
                    dtype=np.float32,
                )
                _stage_add(stage_stats, "to_dev", time.perf_counter() - _t_dev)

                _t_sample = time.perf_counter()
                crops_batch = numpy_grid_sample(
                    img=equi_batch,
                    grid=numpy_grid,
                    out=out_ref_np,
                    mode=config.mode,
                )
                _stage_add(stage_stats, "sample", time.perf_counter() - _t_sample)

                _t_post = time.perf_counter()
                if config.clip_output:
                    np.clip(crops_batch, 0, 255, out=crops_batch)
                crops_batch = crops_batch.astype(np.uint8, copy=False)
                _stage_add(stage_stats, "post", time.perf_counter() - _t_post)

                for zone_i, zone in enumerate(zones):
                    if stop_event and stop_event.is_set():
                        stopped_early = True
                        break
                    out_name = f"{_zone_id_tag(zone)}_{frame_path.stem}{config.image_ext}"
                    _t_save = time.perf_counter()
                    _submit_save(
                        save_pool=save_pool,
                        pending_writes=pending_writes,
                        crop=crops_batch[zone_i],
                        out_path=zone_dirs[zone.zone_id] / out_name,
                        compression_strength=config.compression_strength,
                        use_imagecodecs=config.use_imagecodecs,
                        stop_event=stop_event,
                    )
                    _stage_add(stage_stats, "save", time.perf_counter() - _t_save)
                    written += 1

                    if progress_cb:
                        progress_cb(
                            ProgressInfo(
                                total_frames=total_frames,
                                frame_index=global_idx + 1,
                                frame_name=frame_path.name,
                                total_zones=len(zones),
                                zone_index=zone_i + 1,
                                eta_seconds=_estimate_eta(start_ts, global_idx + 1, total_frames),
                                message=f"Frame {global_idx + 1}/{total_frames}, zone {zone_i + 1}/{len(zones)}",
                            )
                        )
                if stopped_early:
                    break

        if stopped_early:
            _cancel_pending_writes(pending_writes)
            save_pool.shutdown(wait=False, cancel_futures=True)
            return zones, written

        _wait_pending_writes(pending_writes)
        save_pool.shutdown(wait=True)
        return zones, written
    finally:
        if stopped_early:
            # Best-effort no-wait shutdown for responsive stop.
            try:
                save_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


def render_layout_overlay(
    size: tuple[int, int],
    zones: Sequence[Zone],
    *,
    marker_radius_px: int = 5,
    show_borders: bool = False,
    fov_x_deg: Optional[float] = None,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
) -> Image.Image:
    width, height = size
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    border_rgba = (64, 255, 64, 115)
    marker_scale = max(height / 416.0, 1.0)
    marker_radius = max(2, int(round(marker_radius_px * marker_scale)))
    outline_width = max(2, int(round(2 * marker_scale)))
    text_gap = max(2, int(round(2 * marker_scale)))
    font_size = max(8, int(round(8 * marker_scale)))
    text_stroke = max(1, int(round(marker_scale)))
    border_width = max(1, int(round(2 * marker_scale)))
    font = None
    font_candidates: list[str] = ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "SegoeUI.ttf", "segoeui.ttf"]
    if sys.platform == "win32":
        windows_fonts = Path("C:/Windows/Fonts")
        font_candidates.extend(
            [
                str(windows_fonts / "arial.ttf"),
                str(windows_fonts / "ARIAL.TTF"),
                str(windows_fonts / "segoeui.ttf"),
                str(windows_fonts / "SEGOEUI.TTF"),
                str(windows_fonts / "tahoma.ttf"),
                str(windows_fonts / "TAHOMA.TTF"),
            ]
        )
    for font_name in font_candidates:
        try:
            font = ImageFont.truetype(font_name, size=font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    def map_to_xy(yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        x = (yaw_deg % 360.0) / 360.0 * width
        y = (0.5 - (pitch_deg / 180.0)) * height
        return x, y

    def spherical_border_points(
        *,
        yaw_deg: float,
        pitch_deg: float,
        fov_x_deg_local: float,
        fov_y_deg_local: float,
        samples_per_edge: int = 48,
    ) -> list[tuple[float, float]]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        tan_x = math.tan(math.radians(fov_x_deg_local) * 0.5)
        tan_y = math.tan(math.radians(fov_y_deg_local) * 0.5)

        fwd = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=np.float64,
        )
        right = np.array(
            [
                -math.sin(yaw),
                math.cos(yaw),
                0.0,
            ],
            dtype=np.float64,
        )
        up = np.array(
            [
                -math.sin(pitch) * math.cos(yaw),
                -math.sin(pitch) * math.sin(yaw),
                math.cos(pitch),
            ],
            dtype=np.float64,
        )

        def ray_to_xy(u: float, v: float) -> tuple[float, float]:
            ray = fwd + u * right + v * up
            ray /= np.linalg.norm(ray)
            ray_yaw = math.degrees(math.atan2(float(ray[1]), float(ray[0])))
            ray_pitch = math.degrees(math.asin(float(max(-1.0, min(1.0, ray[2])))))
            return map_to_xy(ray_yaw, ray_pitch)

        points: list[tuple[float, float]] = []
        edge_steps = max(4, samples_per_edge)

        for i in range(edge_steps):
            t = i / (edge_steps - 1)
            u = -tan_x + 2.0 * tan_x * t
            points.append(ray_to_xy(u, tan_y))
        for i in range(1, edge_steps):
            t = i / (edge_steps - 1)
            v = tan_y - 2.0 * tan_y * t
            points.append(ray_to_xy(tan_x, v))
        for i in range(1, edge_steps):
            t = i / (edge_steps - 1)
            u = tan_x - 2.0 * tan_x * t
            points.append(ray_to_xy(u, -tan_y))
        for i in range(1, edge_steps - 1):
            t = i / (edge_steps - 1)
            v = -tan_y + 2.0 * tan_y * t
            points.append(ray_to_xy(-tan_x, v))

        return points

    def draw_polyline(draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        max_dx = width * 0.5
        max_dy = height * 0.5
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1) % len(points)]
            if abs(x1 - x0) > max_dx or abs(y1 - y0) > max_dy:
                continue
            draw.line((x0, y0, x1, y1), fill=border_rgba, width=border_width)

    if show_borders and fov_x_deg is not None and crop_width is not None and crop_height is not None:
        fov_y_deg = compute_fov_y_deg(fov_x_deg=fov_x_deg, width=crop_width, height=crop_height)
        border_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for z in zones:
            zone_border = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            zone_draw = ImageDraw.Draw(zone_border)
            border_points = spherical_border_points(
                yaw_deg=z.yaw_deg,
                pitch_deg=z.pitch_deg,
                fov_x_deg_local=fov_x_deg,
                fov_y_deg_local=fov_y_deg,
            )
            draw_polyline(zone_draw, border_points)
            border_overlay = Image.alpha_composite(border_overlay, zone_border)
        base = Image.alpha_composite(base, border_overlay)

    draw = ImageDraw.Draw(base)
    for z in zones:
        x, y = map_to_xy(z.yaw_deg, z.pitch_deg)
        x0 = x - marker_radius
        y0 = y - marker_radius
        x1 = x + marker_radius
        y1 = y + marker_radius
        draw.ellipse((x0, y0, x1, y1), outline=(255, 64, 64, 255), width=outline_width)
        draw.text(
            (x + marker_radius + text_gap, y - marker_radius - text_gap),
            f"{z.zone_id}",
            fill=(255, 255, 64, 255),
            font=font,
            stroke_width=text_stroke,
            stroke_fill=(0, 0, 0, 255),
        )
    return base


def render_layout_preview(
    frame_path: Path,
    zones: Sequence[Zone],
    *,
    marker_radius_px: int = 5,
    show_borders: bool = False,
    fov_x_deg: Optional[float] = None,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
) -> Image.Image:
    with Image.open(frame_path) as img:
        base = img.convert("RGB")
    draw = ImageDraw.Draw(base)
    width, height = base.size
    marker_scale = max(height / 416.0, 1.0)
    marker_radius = max(2, int(round(marker_radius_px * marker_scale)))
    outline_width = max(2, int(round(2 * marker_scale)))
    text_gap = max(2, int(round(2 * marker_scale)))
    font_size = max(8, int(round(8 * marker_scale)))
    text_stroke = max(1, int(round(marker_scale)))
    border_width = max(1, int(round(2 * marker_scale)))
    font = None
    font_candidates: list[str] = ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "SegoeUI.ttf", "segoeui.ttf"]
    if sys.platform == "win32":
        windows_fonts = Path("C:/Windows/Fonts")
        font_candidates.extend(
            [
                str(windows_fonts / "arial.ttf"),
                str(windows_fonts / "ARIAL.TTF"),
                str(windows_fonts / "segoeui.ttf"),
                str(windows_fonts / "SEGOEUI.TTF"),
                str(windows_fonts / "tahoma.ttf"),
                str(windows_fonts / "TAHOMA.TTF"),
            ]
        )
    for font_name in font_candidates:
        try:
            font = ImageFont.truetype(font_name, size=font_size)
            break
        except OSError:
            continue
    if font is None:
        # Last-resort bitmap font (fixed size).
        font = ImageFont.load_default()

    def map_to_xy(yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        x = (yaw_deg % 360.0) / 360.0 * width
        y = (0.5 - (pitch_deg / 180.0)) * height
        return x, y

    def spherical_border_points(
        *,
        yaw_deg: float,
        pitch_deg: float,
        fov_x_deg_local: float,
        fov_y_deg_local: float,
        samples_per_edge: int = 48,
    ) -> list[tuple[float, float]]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        tan_x = math.tan(math.radians(fov_x_deg_local) * 0.5)
        tan_y = math.tan(math.radians(fov_y_deg_local) * 0.5)

        # Local camera basis on sphere, aligned with the same yaw/pitch center
        # mapping used by preview center points.
        fwd = np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=np.float64,
        )
        right = np.array(
            [
                -math.sin(yaw),
                math.cos(yaw),
                0.0,
            ],
            dtype=np.float64,
        )
        up = np.array(
            [
                -math.sin(pitch) * math.cos(yaw),
                -math.sin(pitch) * math.sin(yaw),
                math.cos(pitch),
            ],
            dtype=np.float64,
        )

        def ray_to_xy(u: float, v: float) -> tuple[float, float]:
            ray = fwd + u * right + v * up
            ray /= np.linalg.norm(ray)
            ray_yaw = math.degrees(math.atan2(float(ray[1]), float(ray[0])))
            ray_pitch = math.degrees(math.asin(float(max(-1.0, min(1.0, ray[2])))))
            return map_to_xy(ray_yaw, ray_pitch)

        points: list[tuple[float, float]] = []
        edge_steps = max(4, samples_per_edge)

        # Top edge (left -> right)
        for i in range(edge_steps):
            t = i / (edge_steps - 1)
            u = -tan_x + 2.0 * tan_x * t
            points.append(ray_to_xy(u, tan_y))
        # Right edge (top -> bottom)
        for i in range(1, edge_steps):
            t = i / (edge_steps - 1)
            v = tan_y - 2.0 * tan_y * t
            points.append(ray_to_xy(tan_x, v))
        # Bottom edge (right -> left)
        for i in range(1, edge_steps):
            t = i / (edge_steps - 1)
            u = tan_x - 2.0 * tan_x * t
            points.append(ray_to_xy(u, -tan_y))
        # Left edge (bottom -> top)
        for i in range(1, edge_steps - 1):
            t = i / (edge_steps - 1)
            v = -tan_y + 2.0 * tan_y * t
            points.append(ray_to_xy(-tan_x, v))

        return points

    def draw_polyline(points: Sequence[tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        max_dx = width * 0.5
        max_dy = height * 0.5
        for i, (x0, y0) in enumerate(points):
            x1, y1 = points[(i + 1) % len(points)]
            if abs(x1 - x0) > max_dx or abs(y1 - y0) > max_dy:
                continue
            draw.line((x0, y0, x1, y1), fill=(64, 255, 64), width=border_width)

    if show_borders and fov_x_deg is not None and crop_width is not None and crop_height is not None:
        fov_y_deg = compute_fov_y_deg(fov_x_deg=fov_x_deg, width=crop_width, height=crop_height)
        for z in zones:
            border_points = spherical_border_points(
                yaw_deg=z.yaw_deg,
                pitch_deg=z.pitch_deg,
                fov_x_deg_local=fov_x_deg,
                fov_y_deg_local=fov_y_deg,
            )
            draw_polyline(border_points)

    for z in zones:
        x, y = map_to_xy(z.yaw_deg, z.pitch_deg)
        x0 = x - marker_radius
        y0 = y - marker_radius
        x1 = x + marker_radius
        y1 = y + marker_radius
        draw.ellipse((x0, y0, x1, y1), outline=(255, 64, 64), width=outline_width)
        draw.text(
            (x + marker_radius + text_gap, y - marker_radius - text_gap),
            f"{z.zone_id}",
            fill=(255, 255, 64),
            font=font,
            stroke_width=text_stroke,
            stroke_fill=(0, 0, 0),
        )
    return base


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds_i = max(0, int(math.ceil(seconds)))
    m, s = divmod(seconds_i, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
