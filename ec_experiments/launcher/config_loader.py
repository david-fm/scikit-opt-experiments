from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class AlgorithmPlan:
    algorithm_id: str
    base_params: dict[str, Any]
    name_map: dict[str, str]
    config_sets: dict[str, list[dict[str, Any]]]
    restrictions: list[str]
    on_restriction_violation: str


@dataclass(frozen=True)
class TaskSpec:
    objective_id: str
    algorithm_id: str
    objective_params: dict[str, Any]
    subdir_template: str
    lb: float | None
    ub: float | None
    runs: int | None
    verbose: int | None
    log_scale: bool | None
    track_performance: bool | None


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    output_dir: str
    runs: int
    verbose: int
    log_scale: bool
    track_performance: bool
    tasks: list[TaskSpec]
    algorithms: dict[str, AlgorithmPlan]


def _parse_algorithm_plan(algorithm_id: str, payload: dict[str, Any]) -> AlgorithmPlan:
    base_params = dict(payload.get("base_params", {}))
    name_map = dict(payload.get("name_map", {}))
    restrictions = [str(expr) for expr in payload.get("restrictions", [])]
    on_restriction_violation = str(payload.get("on_restriction_violation", "warn")).lower()

    if on_restriction_violation not in {"warn", "skip", "error"}:
        raise ValueError(
            "on_restriction_violation must be one of: warn, skip, error "
            f"(algorithm '{algorithm_id}')"
        )

    sets: dict[str, list[dict[str, Any]]] = {}
    for set_item in payload.get("sets", []):
        name = str(set_item["name"])
        variants = [dict(item) for item in set_item.get("variants", [])]
        sets[name] = variants

    if not sets:
        raise ValueError(f"Algorithm '{algorithm_id}' does not define any config sets")

    return AlgorithmPlan(
        algorithm_id=algorithm_id,
        base_params=base_params,
        name_map=name_map,
        config_sets=sets,
        restrictions=restrictions,
        on_restriction_violation=on_restriction_violation,
    )


def load_experiment_plan(config_path: str | Path) -> ExperimentPlan:
    path = Path(config_path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    experiment = raw.get("experiment", {})
    name = str(experiment.get("name", path.stem))
    output_dir = str(experiment.get("output_dir", f"outputs/runs/{name}"))
    runs = int(experiment.get("runs", 5))
    verbose = int(experiment.get("verbose", 1))
    log_scale = bool(experiment.get("log_scale", False))
    track_performance = bool(experiment.get("track_performance", False))

    algorithms_raw: dict[str, Any] = raw.get("algorithms", {})
    if not algorithms_raw:
        raise ValueError("No algorithms section found in TOML file")

    algorithms = {
        algorithm_id: _parse_algorithm_plan(algorithm_id, payload)
        for algorithm_id, payload in algorithms_raw.items()
    }

    tasks: list[TaskSpec] = []
    for task_item in raw.get("tasks", []):
        task = TaskSpec(
            objective_id=str(task_item["objective_id"]),
            algorithm_id=str(task_item["algorithm_id"]),
            objective_params=dict(task_item.get("objective_params", {})),
            subdir_template=str(task_item.get("subdir_template", "{objective}_{algorithm}")),
            lb=float(task_item["lb"]) if "lb" in task_item else None,
            ub=float(task_item["ub"]) if "ub" in task_item else None,
            runs=int(task_item["runs"]) if "runs" in task_item else None,
            verbose=int(task_item["verbose"]) if "verbose" in task_item else None,
            log_scale=bool(task_item["log_scale"]) if "log_scale" in task_item else None,
            track_performance=bool(task_item["track_performance"]) if "track_performance" in task_item else None,
        )
        tasks.append(task)

    if not tasks:
        raise ValueError("No tasks were defined in TOML file")

    return ExperimentPlan(
        name=name,
        output_dir=output_dir,
        runs=runs,
        verbose=verbose,
        log_scale=log_scale,
        track_performance=track_performance,
        tasks=tasks,
        algorithms=algorithms,
    )
