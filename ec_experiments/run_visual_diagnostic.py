# Example usage:
# 1) Registry mode:
#    uv run python -m ec_experiments.run_visual_diagnostic --algorithm-id pso --objective-id rosenbrock --n-dim 4
# 2) Plan mode:
#    uv run python -m ec_experiments.run_visual_diagnostic --config config/experiments/pso_coefficient_analysis.toml --task-index 0
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sko.DE import DE
from sko.base import SkoBase

from ec_experiments.demo_pso import Configuration
from ec_experiments.launcher.algorithms import ALGORITHMS
from ec_experiments.launcher.config_loader import load_experiment_plan
from ec_experiments.launcher.objectives import OBJECTIVES
from ec_experiments.launcher.restrictions import validate_restrictions
from ec_experiments.visualization import run_diagnostic_animation


class DEWithHistoryX(DE):
    """DE variant for diagnostics that records full population history (all_history_X)."""

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        self.all_history_X = []

        for _ in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.selection()

            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_X.append(self.X.copy())

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


def _resolve_algorithm_id(algorithm_id: str | None, algo_legacy: str | None) -> str:
    if algorithm_id:
        return algorithm_id
    if algo_legacy:
        return algo_legacy
    return "pso"


def _resolve_objective_id(objective_id: str | None, func_legacy: str | None, algorithm_id: str) -> str:
    if objective_id:
        return objective_id
    if func_legacy:
        return func_legacy
    if algorithm_id in {"pso_tsp", "ga.tsp.modified.v1"}:
        return "tsp.berlin52.rounded"
    return "rosenbrock"


def _default_params_for_algorithm(algorithm_id: str) -> dict[str, Any]:
    if algorithm_id == "pso":
        return {"w": 0.8, "c1": 0.5, "c2": 0.5, "pop": 60, "max_iter": 150}
    if algorithm_id == "de":
        return {"size_pop": 100, "max_iter": 600, "prob_mut": 1, "F": 0.8}
    if algorithm_id in {"ga", "ga.modified.v1", "ga.binary.v1", "rcga.modified.v1"}:
        return {"size_pop": 60, "max_iter": 150, "prob_mut": 0.001, "precision": 1e-7}
    if algorithm_id in {"pso_tsp", "ga.tsp.modified.v1"}:
        return {"w": 0.8, "c1": 0.5, "c2": 0.5, "size_pop": 200, "max_iter": 300}
    raise ValueError(f"No default params configured for algorithm_id='{algorithm_id}'")


def _pick_variant(
    variants: list[dict[str, Any]],
    base_params: dict[str, Any],
    restrictions: list[str],
    on_violation: str,
    variant_index: int,
) -> dict[str, Any]:
    if not variants:
        return dict(base_params)

    start_idx = max(0, variant_index)
    for idx in range(start_idx, len(variants)):
        merged = dict(base_params)
        merged.update(variants[idx])
        if not restrictions:
            return merged

        is_valid, violations = validate_restrictions(merged, restrictions)
        if is_valid:
            return merged

        details = "; ".join(violations)
        message = f"Variant index {idx} violates restrictions: {details}"
        if on_violation == "error":
            raise ValueError(message)
        if on_violation == "warn":
            print(f"[WARN] {message}")
            return merged
        print(f"[SKIP] {message}")

    raise ValueError("No valid variant found after applying restrictions with skip policy")


def _build_registry_setup(
    algorithm_id: str,
    objective_id: str,
    n_dim: int,
    tsp_base_path: str,
    use_de_historyx: bool,
) -> tuple[
    Configuration,
    Any,
    int,
    np.ndarray | None,
    type[SkoBase],
    Any,
    float,
    np.ndarray | None,
    np.ndarray | None,
]:
    if algorithm_id not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm_id '{algorithm_id}'")
    if objective_id not in OBJECTIVES:
        raise ValueError(f"Unknown objective_id '{objective_id}'")

    algo_spec = ALGORITHMS[algorithm_id]
    obj_spec = OBJECTIVES[objective_id]

    objective_params: dict[str, Any] = {}
    if obj_spec.problem_type == "continuous":
        objective_params["n_dim"] = n_dim
    if obj_spec.problem_type == "tsp":
        objective_params["tsp_base_path"] = tsp_base_path

    func, resolved_n_dim, objective_ctx = obj_spec.builder(objective_params)
    params = _default_params_for_algorithm(algorithm_id)
    config = Configuration(params=params, name_map={"max_iter": "iter"})

    algo_cls: type[SkoBase] = DEWithHistoryX if use_de_historyx and algorithm_id == "de" else algo_spec.algo_cls
    city_coords = objective_ctx.get("city_coords") if isinstance(objective_ctx, dict) else None

    return config, func, resolved_n_dim, city_coords, algo_cls, algo_spec.config_filter, None, None


def _build_plan_setup(
    config_path: str,
    task_index: int,
    set_name: str | None,
    variant_index: int,
    use_de_historyx: bool,
) -> tuple[
    Configuration,
    Any,
    int,
    np.ndarray | None,
    type[SkoBase],
    Any,
    np.ndarray | None,
    np.ndarray | None,
]:
    plan = load_experiment_plan(config_path)
    if task_index < 0 or task_index >= len(plan.tasks):
        raise ValueError(f"task_index out of range: {task_index}. Available: 0..{len(plan.tasks)-1}")

    task = plan.tasks[task_index]
    if task.objective_id not in OBJECTIVES:
        raise ValueError(f"Unknown objective_id '{task.objective_id}' in plan")
    if task.algorithm_id not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm_id '{task.algorithm_id}' in plan")

    algo_spec = ALGORITHMS[task.algorithm_id]
    algo_plan = plan.algorithms.get(task.algorithm_id)
    if algo_plan is None:
        raise ValueError(f"No [algorithms.{task.algorithm_id}] section found in plan")

    selected_set = set_name if set_name is not None else next(iter(algo_plan.config_sets.keys()))
    if selected_set not in algo_plan.config_sets:
        raise ValueError(
            f"Unknown set_name '{selected_set}' for algorithm '{task.algorithm_id}'. "
            f"Available: {', '.join(algo_plan.config_sets.keys())}"
        )

    variants = algo_plan.config_sets[selected_set]
    merged_params = _pick_variant(
        variants=variants,
        base_params=algo_plan.base_params,
        restrictions=algo_plan.restrictions,
        on_violation=algo_plan.on_restriction_violation,
        variant_index=variant_index,
    )

    config = Configuration(params=merged_params, name_map=algo_plan.name_map)
    obj_spec = OBJECTIVES[task.objective_id]
    func, resolved_n_dim, objective_ctx = obj_spec.builder(task.objective_params)
    city_coords = objective_ctx.get("city_coords") if isinstance(objective_ctx, dict) else None

    algo_cls: type[SkoBase] = DEWithHistoryX if use_de_historyx and task.algorithm_id == "de" else algo_spec.algo_cls

    lb_override = np.full(resolved_n_dim, float(task.lb), dtype=float) if task.lb is not None else None
    ub_override = np.full(resolved_n_dim, float(task.ub), dtype=float) if task.ub is not None else None

    return (
        config,
        func,
        resolved_n_dim,
        city_coords,
        algo_cls,
        algo_spec.config_filter,
        lb_override,
        ub_override,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-run visual diagnostic for scikit-opt algorithms.")
    parser.add_argument("--config", type=str, default=None, help="Optional experiment TOML plan")
    parser.add_argument("--task-index", type=int, default=0, help="Task index to load from TOML plan")
    parser.add_argument("--set-name", type=str, default=None, help="Configuration set name inside algorithm plan")
    parser.add_argument("--variant-index", type=int, default=0, help="Variant index inside selected set")
    parser.add_argument("--algorithm-id", type=str, default=None, help="Algorithm id from launcher registry")
    parser.add_argument("--objective-id", type=str, default=None, help="Objective id from launcher registry")
    # Legacy aliases for backward compatibility.
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--func", type=str, default=None)
    parser.add_argument("--n-dim", type=int, default=4)
    parser.add_argument("--out", type=str, default="diagnostics/diagnostic.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-dims", type=int, default=16)
    parser.add_argument("--tsp-base-path", type=str, default="data/berlin52")
    parser.add_argument("--slice-anchor-mode", choices=["zero", "best"], default="zero")
    parser.add_argument("--de-historyx", action="store_true", help="Use DE diagnostic class with all_history_X")
    args = parser.parse_args()

    if args.config:
        (
            config,
            func,
            n_dim,
            city_coords,
            algo_cls,
            config_filter,
            lb_override,
            ub_override,
        ) = _build_plan_setup(
            config_path=args.config,
            task_index=args.task_index,
            set_name=args.set_name,
            variant_index=args.variant_index,
            use_de_historyx=args.de_historyx,
        )
    else:
        algorithm_id = _resolve_algorithm_id(args.algorithm_id, args.algo)
        objective_id = _resolve_objective_id(args.objective_id, args.func, algorithm_id)
        (
            config,
            func,
            n_dim,
            city_coords,
            algo_cls,
            config_filter,
            lb_override,
            ub_override,
        ) = _build_registry_setup(
            algorithm_id=algorithm_id,
            objective_id=objective_id,
            n_dim=args.n_dim,
            tsp_base_path=args.tsp_base_path,
            use_de_historyx=args.de_historyx,
        )

    result = run_diagnostic_animation(
        config=config,
        func=func,
        n_dim=n_dim,
        sko_to_use=algo_cls,
        out_path=Path(args.out),
        config_filter=config_filter,
        fps=args.fps,
        max_dims=args.max_dims,
        slice_anchor_mode=args.slice_anchor_mode,
        city_coords=city_coords,
        verbose=1,
        lb_override=lb_override,
        ub_override=ub_override,
    )

    print(f"mode={result['mode']} best_y={result['best_y']:.6e} out={result['out_path']}")


if __name__ == "__main__":
    main()
