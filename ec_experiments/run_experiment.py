from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ec_experiments.demo_pso import Configuration, train_multiple_configs
from ec_experiments.launcher.algorithms import ALGORITHMS
from ec_experiments.launcher.config_loader import load_experiment_plan
from ec_experiments.launcher.objectives import OBJECTIVES
from ec_experiments.launcher.restrictions import validate_restrictions


def _build_config_sets(
    base_params: dict,
    name_map: dict[str, str],
    variants_by_set: dict[str, list[dict]],
    restrictions: list[str],
    on_violation: str,
) -> dict[str, list[Configuration]]:
    base = Configuration(params=base_params, name_map=name_map)
    output: dict[str, list[Configuration]] = {}

    for set_name, variants in variants_by_set.items():
        built_configs: list[Configuration] = []
        for variant in variants:
            cfg = base.add_config_and_name_maps(variant, {})
            if restrictions:
                is_valid, violations = validate_restrictions(cfg.params, restrictions)
                if not is_valid:
                    details = "; ".join(violations)
                    message = (
                        f"Restriction violation in set '{set_name}' for config {cfg.params}: {details}"
                    )
                    if on_violation == "error":
                        raise ValueError(message)
                    if on_violation == "warn":
                        print(f"[WARN] {message}")
                    if on_violation == "skip":
                        print(f"[SKIP] {message}")
                        continue
            built_configs.append(cfg)

        if built_configs:
            output[set_name] = built_configs

    if not output and on_violation == "skip":
        return output

    if not output:
        raise ValueError(
            "No valid configurations available after restriction processing. "
            "Adjust restrictions or set on_restriction_violation='warn'."
        )

    return output


def _resolve_subdir(template: str, objective_id: str, algorithm_id: str) -> str:
    objective = objective_id.replace(".", "_")
    algorithm = algorithm_id.replace(".", "_")
    return template.format(objective=objective, algorithm=algorithm)


def run_from_plan(
    config_path: str,
    runs_override: int | None = None,
    output_override: str | None = None,
    verbose_override: int | None = None,
    only_objectives: set[str] | None = None,
    only_algorithms: set[str] | None = None,
    dry_run: bool = False,
) -> None:
    plan = load_experiment_plan(config_path)

    runs = runs_override if runs_override is not None else plan.runs
    verbose = verbose_override if verbose_override is not None else plan.verbose
    output_dir = output_override if output_override is not None else plan.output_dir

    for task in plan.tasks:
        if only_objectives is not None and task.objective_id not in only_objectives:
            continue
        if only_algorithms is not None and task.algorithm_id not in only_algorithms:
            continue

        if task.objective_id not in OBJECTIVES:
            raise ValueError(f"Unknown objective_id '{task.objective_id}'")
        if task.algorithm_id not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm_id '{task.algorithm_id}'")

        objective_spec = OBJECTIVES[task.objective_id]
        algorithm_spec = ALGORITHMS[task.algorithm_id]
        algorithm_plan = plan.algorithms.get(task.algorithm_id)

        if algorithm_plan is None:
            raise ValueError(f"Task references algorithm '{task.algorithm_id}' but no config was defined")
        if objective_spec.problem_type != algorithm_spec.problem_type:
            raise ValueError(
                f"Incompatible task '{task.objective_id}' + '{task.algorithm_id}': "
                f"{objective_spec.problem_type} vs {algorithm_spec.problem_type}"
            )

        func, n_dim, _ = objective_spec.builder(task.objective_params)
        config_sets = _build_config_sets(
            base_params=algorithm_plan.base_params,
            name_map=algorithm_plan.name_map,
            variants_by_set=algorithm_plan.config_sets,
            restrictions=algorithm_plan.restrictions,
            on_violation=algorithm_plan.on_restriction_violation,
        )

        if not config_sets:
            print(
                f"[SKIP] task={task.objective_id}/{task.algorithm_id} has no valid configurations "
                "after applying restrictions"
            )
            continue

        task_runs = task.runs if task.runs is not None else runs
        task_verbose = task.verbose if task.verbose is not None else verbose
        task_log_scale = task.log_scale if task.log_scale is not None else plan.log_scale
        task_track_performance = (
            task.track_performance if task.track_performance is not None else plan.track_performance
        )

        lb = [task.lb] * n_dim if task.lb is not None else [-2]
        ub = [task.ub] * n_dim if task.ub is not None else [2]

        relative_dir = _resolve_subdir(task.subdir_template, task.objective_id, task.algorithm_id)
        folder = str(Path(output_dir) / relative_dir)

        if dry_run:
            print(
                f"[DRY RUN] task={task.objective_id}/{task.algorithm_id} "
                f"runs={task_runs} n_dim={n_dim} out={folder}"
            )
            continue

        print(
            f"[RUN] objective={task.objective_id} algorithm={task.algorithm_id} "
            f"runs={task_runs} n_dim={n_dim}"
        )
        train_multiple_configs(
            config_sets,
            runs=task_runs,
            folder=folder,
            func=func,
            n_dim=n_dim,
            Sko_to_use=algorithm_spec.algo_cls,
            config_filter=algorithm_spec.config_filter,
            verbose=task_verbose,
            log_scale=task_log_scale,
            track_performance=task_track_performance,
            lb=lb,
            ub=ub,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiments from TOML plans")
    parser.add_argument("--config", required=True, help="Path to TOML experiment plan")
    parser.add_argument("--runs", type=int, default=None, help="Override runs value")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=None, help="Override verbosity")
    parser.add_argument("--only-objective", action="append", default=None, help="Run only selected objective_id")
    parser.add_argument("--only-algorithm", action="append", default=None, help="Run only selected algorithm_id")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print execution plan without running")
    args = parser.parse_args(argv)

    run_from_plan(
        config_path=args.config,
        runs_override=args.runs,
        output_override=args.output_dir,
        verbose_override=args.verbose,
        only_objectives=set(args.only_objective) if args.only_objective else None,
        only_algorithms=set(args.only_algorithm) if args.only_algorithm else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
