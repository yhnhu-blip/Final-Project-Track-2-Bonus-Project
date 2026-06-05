"""High-level planner for the 200 m track bonus (LEARNED version).

Maps the official compact 5D track observation to the joystick command consumed
by the HW1 Go2 low-level policy:

    5D track observation -> [vx, vy, yaw_rate]

This file keeps the SAME public interface the evaluator relies on:
    - a class with `.command(obs, t)` returning an np.array of shape (3,)
    - a `.load(path)` classmethod
    - a `.config` attribute exposing the official track fields for validation

Two planner types are supported, selected by `planner_type` in the JSON config:
    - "starter_pd"  : the original conservative rule baseline (kept for reference)
    - "mlp_residual": a small learned MLP that outputs (vx_residual, yaw_residual)
                      on top of a geometric feed-forward. The MLP weights are the
                      "learned parameters" required for the leaderboard route.

The MLP weights are stored in a separate .npz file referenced by the config
field `weights_path`. If no weights are given (or the file is missing), the MLP
residual is zero, so the planner degrades gracefully to a pure-rule controller
that already completes the lap. This makes the planner safe to load at any stage
of training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


# --------------------------------------------------------------------------- #
# MLP utilities (pure numpy; arch is fixed so CEM searches a flat weight vector)
# --------------------------------------------------------------------------- #
MLP_IN = 5
MLP_HIDDEN = 16
MLP_OUT = 2  # (vx_residual, yaw_residual), both in [-1, 1] via tanh


def mlp_param_count(in_: int = MLP_IN, h: int = MLP_HIDDEN, out: int = MLP_OUT) -> int:
    return in_ * h + h + h * h + h + h * out + out


def mlp_unpack(theta: np.ndarray, in_: int = MLP_IN, h: int = MLP_HIDDEN, out: int = MLP_OUT):
    theta = np.asarray(theta, dtype=np.float64).ravel()
    i = 0
    W1 = theta[i:i + in_ * h].reshape(in_, h); i += in_ * h
    b1 = theta[i:i + h]; i += h
    W2 = theta[i:i + h * h].reshape(h, h); i += h * h
    b2 = theta[i:i + h]; i += h
    W3 = theta[i:i + h * out].reshape(h, out); i += h * out
    b3 = theta[i:i + out]; i += out
    if i != theta.size:
        raise ValueError(f"theta size {theta.size} does not match expected {i}")
    return W1, b1, W2, b2, W3, b3


def mlp_forward(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    W1, b1, W2, b2, W3, b3 = mlp_unpack(theta)
    x = np.asarray(x, dtype=np.float64)
    hh = np.tanh(x @ W1 + b1)
    hh = np.tanh(hh @ W2 + b2)
    out = np.tanh(hh @ W3 + b3)  # bounded in [-1, 1]
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackPlannerConfig:
    planner_type: str = "mlp_residual"

    # --- geometric / rule parameters (kept from the starter) ---
    min_speed_mps: float = 0.30
    max_lateral_speed_mps: float = 0.10
    max_yaw_rate_radps: float = 0.30
    k_heading: float = 0.55
    k_lateral: float = 0.08
    stand_seconds: float = 1.0

    # --- learned-planner parameters ---
    vx_lo: float = 0.60          # min forward speed the MLP can command
    vx_hi: float = 1.30          # max forward speed (stay <= V7 stable ceiling)
    yaw_res_scale: float = 0.15  # how much the MLP can bend yaw beyond the rule
    weights_path: str = ""       # path to .npz holding the MLP weight vector "theta"

    # --- starter_pd fallback fields (only used if planner_type == "starter_pd") ---
    speed_mps: float = 0.85
    heading_slowdown: float = 0.45

    # official track fields are intentionally NOT stored here; the evaluator
    # validates them against the planner config dict, and absent == skip-check.

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {k: payload[k] for k in valid if k in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "TrackPlannerConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class StarterTrackPlanner:
    """Track planner. Keeps the class name the evaluator imports.

    Despite the name, this supports a learned MLP residual planner. The name is
    preserved because `run_track_bonus.py` does `from track_bonus.planner import
    StarterTrackPlanner` and calls `.load(...)` / `.command(...)`.
    """

    def __init__(self, config: TrackPlannerConfig) -> None:
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._theta: np.ndarray | None = None

        if config.planner_type not in ("starter_pd", "mlp_residual"):
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")

        if config.planner_type == "mlp_residual":
            self._theta = self._load_weights(config.weights_path)

    # -- weight loading (graceful: missing weights -> zero residual) --
    def _load_weights(self, weights_path: str) -> np.ndarray:
        n = mlp_param_count()
        if weights_path:
            p = Path(weights_path)
            if not p.is_absolute():
                # resolve relative to this file's repo root
                p = Path(__file__).resolve().parent.parent / weights_path
            if p.exists():
                data = np.load(p)
                theta = np.asarray(data["theta"], dtype=np.float64).ravel()
                if theta.size != n:
                    raise ValueError(
                        f"weights theta has {theta.size} params, expected {n} "
                        f"(arch {MLP_IN}->{MLP_HIDDEN}->{MLP_HIDDEN}->{MLP_OUT})"
                    )
                return theta
        # no weights -> zero residual -> safe pure-rule behavior
        return np.zeros(n, dtype=np.float64)

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        return cls(TrackPlannerConfig.load(path))

    # -- main entry point called every control step by the evaluator --
    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        if self.config.planner_type == "starter_pd":
            return self._command_rule(obs)
        return self._command_mlp(obs)

    # -- learned MLP residual planner --
    def _command_mlp(self, obs: TrackControllerObservation) -> np.ndarray:
        cfg = self.config
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        heading_error = float(obs.heading_error_rad)

        res = mlp_forward(self._theta, obs.as_array())  # in [-1, 1]^2
        vx_res, yaw_res = float(res[0]), float(res[1])

        # vx: midpoint + residual maps to [vx_lo, vx_hi]
        vx = 0.5 * (cfg.vx_lo + cfg.vx_hi) + 0.5 * (cfg.vx_hi - cfg.vx_lo) * vx_res
        vx = float(np.clip(vx, cfg.vx_lo, cfg.vx_hi))

        # vy: pure rule lateral correction (already near-perfect in baseline)
        vy = float(np.clip(-cfg.k_lateral * lateral_error,
                           -cfg.max_lateral_speed_mps, cfg.max_lateral_speed_mps))

        # yaw: geometric feed-forward + heading P + learned residual
        yaw = curvature * vx + cfg.k_heading * heading_error + cfg.yaw_res_scale * yaw_res
        yaw = float(np.clip(yaw, -cfg.max_yaw_rate_radps, cfg.max_yaw_rate_radps))

        return np.asarray([vx, vy, yaw], dtype=np.float32)

    # -- original rule baseline (kept for reference / ablation) --
    def _command_rule(self, obs: TrackControllerObservation) -> np.ndarray:
        cfg = self.config
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(cfg.k_lateral * lateral_error, max(cfg.speed_mps, 1e-3))
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)
        speed_scale = 1.0 - cfg.heading_slowdown * min(abs(heading_error), math.pi) / math.pi
        vx = float(np.clip(cfg.speed_mps * speed_scale, cfg.min_speed_mps, cfg.speed_mps))
        vy = float(np.clip(-cfg.k_lateral * lateral_error,
                           -cfg.max_lateral_speed_mps, cfg.max_lateral_speed_mps))
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw = float(np.clip(curvature * vx + cfg.k_heading * heading_error,
                            -cfg.max_yaw_rate_radps, cfg.max_yaw_rate_radps))
        return np.asarray([vx, vy, yaw], dtype=np.float32)
