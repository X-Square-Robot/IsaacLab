# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Headless benchmark: measure per-step frame rate for a given robot + solver pair.

Supports three robots (``ex001``, ``ex001_6r``, ``cx002``) and two physics
backends (``mjwarp``, ``physx``). Designed to be driven by an
orchestrator (``run_benchmark_report.py``) that runs every (robot, solver)
combination in an isolated subprocess and aggregates the JSON payloads that
this script prints on stdout.

Usage::

    ./isaaclab.sh -p scripts/benchmarks/benchmark_solvers.py \
        --robot ex001_6r --solver mjwarp --num_steps 1000 --headless

Output (last line on stdout)::

    BENCHMARK_RESULT:{"robot": "...", "solver": "...", "joint_dof": 7, ...}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI — parse before AppLauncher boots the simulator
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark solver frame rate for a single (robot, solver) pair.")
parser.add_argument(
    "--robot",
    type=str,
    required=True,
    choices=["ex001", "ex001_6r", "cx002"],
    help="Robot asset to benchmark.",
)
parser.add_argument(
    "--solver",
    type=str,
    required=True,
    choices=["mjwarp", "physx"],
    help="Physics backend to benchmark.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000,
    help="Number of timed simulation steps (default: 1000).",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=20,
    help="Number of warmup steps excluded from timing (default: 20).",
)
parser.add_argument(
    "--output_json",
    type=str,
    default="",
    help="Optional path to write the JSON result; stdout marker is always emitted.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force headless mode for reproducible benchmarking.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that require the simulation app to be running
# ---------------------------------------------------------------------------
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.physics.physics_manager_cfg import PhysicsCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets.robots.cx002 import CX002_CFG
from isaaclab_assets.robots.ex001 import EX001_6R_CFG, EX001_CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot registry
# ---------------------------------------------------------------------------

_ROBOT_REGISTRY: dict[str, ArticulationCfg] = {
    "ex001": EX001_CFG,
    "ex001_6r": EX001_6R_CFG,
    "cx002": CX002_CFG,
}


# ---------------------------------------------------------------------------
# Solver config factory
# ---------------------------------------------------------------------------


def _make_physics_cfg(robot_name: str, solver_name: str) -> PhysicsCfg:
    """Build a :class:`PhysicsCfg` for the requested solver backend."""
    if solver_name == "mjwarp":
        return NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=150,
                nconmax=70,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
            ),
            num_substeps=2,
            use_cuda_graph=False,
        )
    if solver_name == "physx":
        return PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
    raise ValueError(f"Unknown solver: {solver_name!r}")


def _make_sim_cfg(robot_name: str, solver_name: str) -> SimulationCfg:
    """Build a full :class:`SimulationCfg` for the requested solver backend."""
    physics_cfg = _make_physics_cfg(robot_name, solver_name)
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        physics=physics_cfg,
        visualizer_cfgs=NewtonVisualizerCfg(
            headless=True,
            window_width=640,
            window_height=480,
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def _wp_synchronize() -> None:
    """Synchronize the default CUDA device if one is available.

    Falls back silently when running on CPU-only Warp builds.
    """
    with contextlib.suppress(Exception):  # pragma: no cover - CPU fallback path
        wp.synchronize_device()


def main() -> dict:
    """Run ``num_steps`` physics steps and return the timing report."""
    robot_name = args_cli.robot
    solver_name = args_cli.solver

    logger.info("Benchmark: robot=%s solver=%s num_steps=%d", robot_name, solver_name, args_cli.num_steps)

    # -- simulation ------------------------------------------------------
    sim_cfg = _make_sim_cfg(robot_name, solver_name)
    sim = SimulationContext(sim_cfg)

    # Ground + light (same as the interactive demos, so collision workload matches)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # -- robot ------------------------------------------------------------
    prim_path = f"/World/{robot_name}"
    robot_cfg = _ROBOT_REGISTRY[robot_name].replace(prim_path=prim_path)
    robot = Articulation(robot_cfg)

    # Reset — required before reading articulation data
    sim.reset()

    joint_dof = int(robot.num_joints)
    default_target = wp.to_torch(robot.data.default_joint_pos).clone()

    # -- warmup ----------------------------------------------------------
    for _ in range(max(0, args_cli.warmup_steps)):
        robot.set_joint_position_target_index(target=default_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    _wp_synchronize()

    # -- timed loop ------------------------------------------------------
    num_steps = int(args_cli.num_steps)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        robot.set_joint_position_target_index(target=default_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    _wp_synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_time = total_time / num_steps if num_steps > 0 else float("nan")
    avg_fps = num_steps / total_time if total_time > 0 else float("nan")

    result: dict = {
        "robot": robot_name,
        "solver": solver_name,
        "joint_dof": joint_dof,
        "num_steps": num_steps,
        "warmup_steps": int(args_cli.warmup_steps),
        "total_time": total_time,
        "avg_time": avg_time,
        "avg_fps": avg_fps,
    }

    # -- emit result -----------------------------------------------------
    marker_line = "BENCHMARK_RESULT:" + json.dumps(result)
    print(marker_line, flush=True)
    if args_cli.output_json:
        with open(args_cli.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    # Tear down explicitly so solver-owned resources are released.
    sim.clear_instance()
    simulation_app.close()
    return result


if __name__ == "__main__":
    main()
