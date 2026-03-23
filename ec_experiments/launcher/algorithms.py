from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO, PSO_TSP
from sko.base import SkoBase

from ec_experiments.custom_algorithms import GA_Binary, GA_TSP_modified, GA_modified, RCGA_modified


ConfigFilter = Callable[[dict], dict]


@dataclass(frozen=True)
class AlgorithmSpec:
    algo_cls: type[SkoBase]
    config_filter: ConfigFilter
    problem_type: str


ALGORITHMS: dict[str, AlgorithmSpec] = {}


def register_algorithm(algorithm_id: str, spec: AlgorithmSpec) -> None:
    if algorithm_id in ALGORITHMS:
        raise ValueError(f"Algorithm id already registered: {algorithm_id}")
    ALGORITHMS[algorithm_id] = spec


def _identity(cfg: dict) -> dict:
    return dict(cfg)


def _keep_keys(keys: set[str]) -> ConfigFilter:
    def _filter(cfg: dict) -> dict:
        return {k: v for k, v in cfg.items() if k in keys}

    return _filter


register_algorithm(
    "pso",
    AlgorithmSpec(
        algo_cls=PSO,
        config_filter=_keep_keys({"w", "c1", "c2", "pop", "max_iter"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "de",
    AlgorithmSpec(
        algo_cls=DE,
        config_filter=_keep_keys({"size_pop", "max_iter", "F", "prob_mut"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "ga",
    AlgorithmSpec(
        algo_cls=GA,
        config_filter=_keep_keys({"size_pop", "max_iter", "prob_mut", "precision"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "ga.modified.v1",
    AlgorithmSpec(
        algo_cls=GA_modified,
        config_filter=_keep_keys({"size_pop", "max_iter", "prob_mut", "precision", "window_size"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "rcga.modified.v1",
    AlgorithmSpec(
        algo_cls=RCGA_modified,
        config_filter=_keep_keys({"size_pop", "max_iter", "prob_mut", "precision", "window_size"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "ga.binary.v1",
    AlgorithmSpec(
        algo_cls=GA_Binary,
        config_filter=_keep_keys({"size_pop", "max_iter", "prob_mut", "precision", "window_size"}),
        problem_type="continuous",
    ),
)

register_algorithm(
    "ga.tsp.modified.v1",
    AlgorithmSpec(
        algo_cls=GA_TSP_modified,
        config_filter=_keep_keys({"size_pop", "max_iter", "prob_mut", "window_size"}),
        problem_type="tsp",
    ),
)

register_algorithm(
    "pso_tsp",
    AlgorithmSpec(
        algo_cls=PSO_TSP,
        config_filter=_identity,
        problem_type="tsp",
    ),
)
