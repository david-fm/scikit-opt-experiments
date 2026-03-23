from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ec_experiments.demo_pso import griewank_creator, read_tsp, rosenbrock_creator


ObjectiveCallable = Callable[[np.ndarray], float] | Callable[[Any], float]
ObjectiveBuilder = Callable[[dict[str, Any]], tuple[ObjectiveCallable, int, dict[str, Any]]]


@dataclass(frozen=True)
class ObjectiveSpec:
    builder: ObjectiveBuilder
    problem_type: str


OBJECTIVES: dict[str, ObjectiveSpec] = {}


def register_objective(objective_id: str, spec: ObjectiveSpec) -> None:
    if objective_id in OBJECTIVES:
        raise ValueError(f"Objective id already registered: {objective_id}")
    OBJECTIVES[objective_id] = spec


def _build_rosenbrock(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    n_dim = int(params.get("n_dim", 4))
    a = int(params.get("a", 1))
    b = int(params.get("b", 100))
    return rosenbrock_creator(a=a, b=b), n_dim, {}


def _build_griewank(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    n_dim = int(params.get("n_dim", 4))
    a = int(params.get("a", 4000))
    return griewank_creator(a=a), n_dim, {}


def _build_tsp_berlin52(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    tsp_base_path = Path(params.get("tsp_base_path", "data/berlin52"))
    cities, _, _ = read_tsp(str(tsp_base_path))
    coords = np.array([(c[0], c[1]) for c in cities], dtype=float)
    n_dim = len(coords)

    distance_matrix = np.zeros((n_dim, n_dim), dtype=float)
    for i in range(n_dim):
        for j in range(n_dim):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            distance_matrix[i, j] = int(d + 0.5)

    def cal_total_distance(routine: np.ndarray) -> float:
        n_points, = routine.shape
        return float(
            sum(distance_matrix[routine[i % n_points], routine[(i + 1) % n_points]] for i in range(n_points))
        )

    return cal_total_distance, n_dim, {"city_coords": coords}


def _load_tsp_points_for_continuous_objectives(params: dict[str, Any]) -> tuple[np.ndarray, int]:
    tsp_base_path = Path(params.get("tsp_base_path", "data/berlin52"))
    cities, _, _ = read_tsp(str(tsp_base_path))
    # Keep original project behavior: use full city tuple as in legacy scripts.
    points_coordinate = np.asarray(cities, dtype=float)
    n_dim = len(points_coordinate)
    return points_coordinate, n_dim


def _build_tsp_berlin52_euclidean_argsort(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    points_coordinate, n_dim = _load_tsp_points_for_continuous_objectives(params)

    def objective_function(p: np.ndarray) -> float:
        route = np.argsort(p)
        vec_c1 = points_coordinate[route]
        vec_c2 = points_coordinate[np.roll(route, -1)]
        return float(np.sum(np.sqrt(np.sum((vec_c1 - vec_c2) ** 2, axis=1))))

    return objective_function, n_dim, {"city_coords": points_coordinate}


def _build_tsp_berlin52_rankbased(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    points_coordinate, n_dim = _load_tsp_points_for_continuous_objectives(params)

    def objective_function(p: np.ndarray) -> float:
        rank1 = np.argsort(p)
        rank_continuous = np.argsort(rank1).astype(float) / (n_dim - 1)
        route = np.argsort(rank_continuous)
        vec_c1 = points_coordinate[route]
        vec_c2 = points_coordinate[np.roll(route, -1)]
        return float(np.sum(np.sqrt(np.sum((vec_c1 - vec_c2) ** 2, axis=1))))

    return objective_function, n_dim, {"city_coords": points_coordinate}


def _build_tsp_berlin52_exponential_sigmoid(params: dict[str, Any]) -> tuple[ObjectiveCallable, int, dict[str, Any]]:
    points_coordinate, n_dim = _load_tsp_points_for_continuous_objectives(params)

    def objective_function(p: np.ndarray) -> float:
        p_scaled = np.exp(10 * (p - 0.5)) / (1 + np.exp(10 * (p - 0.5)))
        route = np.argsort(p_scaled)
        vec_c1 = points_coordinate[route]
        vec_c2 = points_coordinate[np.roll(route, -1)]
        return float(np.sum(np.sqrt(np.sum((vec_c1 - vec_c2) ** 2, axis=1))))

    return objective_function, n_dim, {"city_coords": points_coordinate}


register_objective("rosenbrock", ObjectiveSpec(builder=_build_rosenbrock, problem_type="continuous"))
register_objective("griewank", ObjectiveSpec(builder=_build_griewank, problem_type="continuous"))
register_objective("tsp.berlin52", ObjectiveSpec(builder=_build_tsp_berlin52, problem_type="tsp"))
register_objective("tsp.berlin52.rounded", ObjectiveSpec(builder=_build_tsp_berlin52, problem_type="tsp"))
register_objective(
    "tsp.berlin52.euclidean_argsort",
    ObjectiveSpec(builder=_build_tsp_berlin52_euclidean_argsort, problem_type="continuous"),
)
register_objective(
    "tsp.berlin52.rankbased",
    ObjectiveSpec(builder=_build_tsp_berlin52_rankbased, problem_type="continuous"),
)
register_objective(
    "tsp.berlin52.exponential_sigmoid",
    ObjectiveSpec(builder=_build_tsp_berlin52_exponential_sigmoid, problem_type="continuous"),
)
