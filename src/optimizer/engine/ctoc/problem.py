"""Immutable, portable problem data; no ATK or Windows imports."""

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from .storage import read_json, digest


def default_data():
    return Path(__file__).resolve().parents[3] / 'data'


@dataclass
class Problem:
    path: Path
    manifest: dict
    states: np.ndarray
    pole: np.ndarray

    @classmethod
    def load(cls, path=None, verify=True):
        path = Path(path or default_data()).resolve()
        manifest = read_json(path / "manifest.json")
        if verify:
            for name, expected in manifest["files"].items():
                if Path(name).name != name or digest(path / name) != expected:
                    raise ValueError(f"Problem data checksum mismatch: {name}")
        states = np.load(path / "states.npy", mmap_mode="r", allow_pickle=False)
        times = np.load(path / "time_s.npy", mmap_mode="r", allow_pickle=False)
        if states.shape != (35, 14401, 6) or not np.array_equal(times, np.arange(14401) * 60):
            raise ValueError("Unsupported ephemeris grid")
        if not np.isfinite(states).all():
            raise ValueError("Non-finite ephemeris")
        from .kernels.physics import MU, RE
        from .kernels.j2_refine import J2

        if any(
            not np.isclose(a, b, rtol=1e-14, atol=0)
            for a, b in [
                (MU, manifest["mu_km3_s2"]),
                (RE, manifest["radius_km"]),
                (J2, manifest["j2"]),
            ]
        ):
            raise ValueError("Data constants disagree with compiled numerical kernels")
        pole = np.array(read_json(path / "calibration.json")["pole_frame_to_icrf"])
        if (
            pole.shape != (3, 3)
            or not np.allclose(pole.T @ pole, np.eye(3), atol=1e-12)
            or np.linalg.det(pole) < 0
        ):
            raise ValueError("Invalid fixed-pole rotation")
        return cls(path, manifest, states, pole)

    @property
    def fingerprint(self):
        return digest(self.path / "manifest.json")

    def pole_states(self, cache=None):
        if cache is None:
            return np.concatenate(
                (self.states[:, :, :3] @ self.pole, self.states[:, :, 3:] @ self.pole), axis=2
            )
        directory = Path(cache)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (self.fingerprint + ".pole.npy")
        if not destination.exists():
            import os, uuid

            tmp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
            with tmp.open("wb") as f:
                np.save(f, self.pole_states(), allow_pickle=False)
            os.replace(tmp, destination)
        return np.load(destination, mmap_mode="r", allow_pickle=False)
