"""两阶段交换用的完整飞越路线；目标内部编号为0到34。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Route:
    order: list[int]
    times_s: list[float]
    branches: list[list[int]]
    initial_state_km_km_s: list[float]
    cost_km_s: float
    arrival_velocity_km_s: list[float]
    minimum_radius_km: float
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            order_1based=[target + 1 for target in self.order],
            target_count=len(set(self.order)),
            end_time_days=self.times_s[-1] / 86400.0 if self.times_s else None,
            units="km,s,km/s",
            status="two_body_screening_only",
        )
        return value
