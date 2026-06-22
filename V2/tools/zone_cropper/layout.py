from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .config import ZoneLayoutConfig


@dataclass(frozen=True)
class Zone:
    zone_id: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0

    def to_rots(self) -> Dict[str, float]:
        return {
            "yaw": math.radians(self.yaw_deg),
            "pitch": math.radians(self.pitch_deg),
            "roll": math.radians(self.roll_deg),
        }


def compute_fov_y_deg(fov_x_deg: float, width: int, height: int) -> float:
    half_fov_x = math.radians(fov_x_deg) / 2.0
    half_fov_y = math.atan((height / float(width)) * math.tan(half_fov_x))
    return math.degrees(half_fov_y * 2.0)


def _base_steps(layout: ZoneLayoutConfig, fov_x_deg: float, fov_y_deg: float) -> tuple[float, float]:
    if layout.mode == "step_mode":
        return layout.yaw_step_deg, layout.pitch_step_deg
    overlap_scale = max(1.0 - layout.overlap, 0.01)
    return fov_x_deg * overlap_scale, fov_y_deg * overlap_scale


def _generate_zones_for_steps(
    *,
    yaw_step_deg: float,
    pitch_step_deg: float,
    max_center_pitch_deg: float,
    include_poles: bool,
) -> List[Zone]:
    eps = 1e-6
    max_center_pitch_deg = max(0.0, min(90.0, max_center_pitch_deg))

    pitch_levels: List[float] = []
    p = -max_center_pitch_deg
    while p <= max_center_pitch_deg + eps:
        pitch_levels.append(p)
        p += pitch_step_deg
    if not pitch_levels:
        pitch_levels = [0.0]
    if abs(pitch_levels[-1] - max_center_pitch_deg) > 0.5:
        pitch_levels.append(max_center_pitch_deg)

    zones: List[Zone] = []
    zone_id = 0
    for ring_idx, pitch in enumerate(pitch_levels):
        cos_pitch = max(math.cos(math.radians(abs(pitch))), 0.15)
        ring_yaw_step = min(360.0, yaw_step_deg / cos_pitch)
        ring_count = max(1, int(math.ceil(360.0 / ring_yaw_step)))
        yaw_offset = (180.0 / ring_count) if (ring_idx % 2 == 1) else 0.0
        for i in range(ring_count):
            yaw = (i * (360.0 / ring_count) + yaw_offset) % 360.0
            zones.append(Zone(zone_id=zone_id, yaw_deg=yaw, pitch_deg=pitch))
            zone_id += 1

    if include_poles and max_center_pitch_deg < 90.0:
        zones.append(Zone(zone_id=zone_id, yaw_deg=0.0, pitch_deg=max_center_pitch_deg))
        zone_id += 1
        zones.append(Zone(zone_id=zone_id, yaw_deg=0.0, pitch_deg=-max_center_pitch_deg))

    return zones


def generate_zone_layout(
    *,
    layout: ZoneLayoutConfig,
    fov_x_deg: float,
    crop_width: int,
    crop_height: int,
) -> List[Zone]:
    fov_y_deg = compute_fov_y_deg(fov_x_deg=fov_x_deg, width=crop_width, height=crop_height)
    if layout.allow_pole_center:
        max_center_pitch_deg = min(layout.max_pitch_deg, 90.0)
    else:
        max_center_pitch_deg = min(layout.max_pitch_deg, 90.0 - fov_y_deg / 2.0)

    base_yaw_step, base_pitch_step = _base_steps(layout, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)
    base_yaw_step = max(base_yaw_step, 0.5)
    base_pitch_step = max(base_pitch_step, 0.5)

    if layout.mode == "step_mode":
        return _generate_zones_for_steps(
            yaw_step_deg=base_yaw_step,
            pitch_step_deg=base_pitch_step,
            max_center_pitch_deg=max_center_pitch_deg,
            include_poles=layout.include_poles,
        )

    # Count mode: adjust ring spacing scale to approach desired zone count.
    low = 0.25
    high = 4.0
    best = _generate_zones_for_steps(
        yaw_step_deg=base_yaw_step,
        pitch_step_deg=base_pitch_step,
        max_center_pitch_deg=max_center_pitch_deg,
        include_poles=layout.include_poles,
    )

    target = layout.target_zones
    for _ in range(18):
        mid = (low + high) / 2.0
        candidate = _generate_zones_for_steps(
            yaw_step_deg=base_yaw_step * mid,
            pitch_step_deg=base_pitch_step * mid,
            max_center_pitch_deg=max_center_pitch_deg,
            include_poles=layout.include_poles,
        )
        if abs(len(candidate) - target) < abs(len(best) - target):
            best = candidate
        if len(candidate) > target:
            low = mid
        else:
            high = mid

    return best
