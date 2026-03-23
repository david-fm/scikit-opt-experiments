from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Type
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sko.PSO import PSO_TSP
from sko.base import SkoBase


def _reshape_history_y(history: Any) -> np.ndarray:
    arr = np.asarray(history)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _extract_fitness_curves(sko_model: SkoBase) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(sko_model, "all_history_Y"):
        hist_y = _reshape_history_y(sko_model.all_history_Y)
        mean_curve = np.mean(hist_y, axis=1)
        min_curve = np.minimum.accumulate(np.min(hist_y, axis=1))
        return mean_curve, min_curve

    if hasattr(sko_model, "gbest_y_hist"):
        hist = np.asarray(sko_model.gbest_y_hist, dtype=float).reshape(-1)
        return hist, hist

    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def _extract_population_history(sko_model: SkoBase) -> list[np.ndarray]:
    # PSO/PSO_TSP in scikit-opt can expose per-iteration X under record_value.
    if hasattr(sko_model, "record_value"):
        record_value = getattr(sko_model, "record_value")
        if isinstance(record_value, dict) and "X" in record_value:
            return [np.asarray(x) for x in record_value["X"]]

    # Some GA variants can expose all_history_X.
    if hasattr(sko_model, "all_history_X"):
        history_x = getattr(sko_model, "all_history_X")
        return [np.asarray(x) for x in history_x]

    # DE (and some other algorithms) expose only the best individual per generation.
    # This is not equivalent to full population history, but it is a useful
    # trajectory fallback when all_history_X is unavailable.
    if hasattr(sko_model, "generation_best_X"):
        generation_best_x = getattr(sko_model, "generation_best_X")
        return [np.asarray(x).reshape(1, -1) for x in generation_best_x]

    return []


def _extract_tsp_route_history(sko_model: SkoBase, pop_history: list[np.ndarray]) -> list[np.ndarray]:
    if hasattr(sko_model, "best_x_history"):
        best_x_history = getattr(sko_model, "best_x_history")
        return [np.asarray(x) for x in best_x_history]

    if pop_history:
        return [np.asarray(x[0]) for x in pop_history if np.asarray(x).ndim >= 2 and len(x) > 0]

    return []


def _build_model(
    config: Any,
    func: Callable,
    n_dim: int,
    sko_to_use: Type[SkoBase],
    config_filter: Callable[[dict], dict],
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
) -> SkoBase:
    filtered_params = config.get_filtered_params(config_filter)

    if issubclass(sko_to_use, PSO_TSP):
        return sko_to_use(func=func, n_dim=n_dim, **filtered_params)

    if lb_override is not None and ub_override is not None:
        return sko_to_use(
            func=func,
            n_dim=n_dim,
            lb=lb_override.tolist(),
            ub=ub_override.tolist(),
            **filtered_params,
        )

    return sko_to_use(
        func=func,
        n_dim=n_dim,
        lb=[-2] * n_dim,
        ub=[2] * n_dim,
        **filtered_params,
    )


def _extract_bounds(sko_model: SkoBase, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(sko_model, "lb") and hasattr(sko_model, "ub"):
        lb = np.asarray(getattr(sko_model, "lb"), dtype=float).reshape(-1)
        ub = np.asarray(getattr(sko_model, "ub"), dtype=float).reshape(-1)
        if lb.size == n_dim and ub.size == n_dim:
            return lb, ub

    lb = np.full(n_dim, -2.0, dtype=float)
    ub = np.full(n_dim, 2.0, dtype=float)
    return lb, ub


def _save_animation(fig: plt.Figure, update_func: Callable[[int], None], n_frames: int, out_path: Path, fps: int) -> None:
    anim = FuncAnimation(fig, update_func, frames=n_frames, interval=max(1, int(1000 / fps)), repeat=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)


def _render_fitness_fallback(
    mean_curve: np.ndarray,
    min_curve: np.ndarray,
    out_path: Path,
    fps: int,
    title: str,
) -> None:
    if len(mean_curve) == 0 and len(min_curve) == 0:
        raise ValueError("No fitness history available to render fallback animation.")

    n_frames = max(len(mean_curve), len(min_curve))
    x = np.arange(n_frames)
    mean_full = np.pad(mean_curve, (0, max(0, n_frames - len(mean_curve))), mode="edge") if len(mean_curve) else np.zeros(n_frames)
    min_full = np.pad(min_curve, (0, max(0, n_frames - len(min_curve))), mode="edge") if len(min_curve) else np.zeros(n_frames)

    fig, ax = plt.subplots(figsize=(9, 5))

    y_min = float(np.nanmin(np.concatenate([mean_full, min_full])))
    y_max = float(np.nanmax(np.concatenate([mean_full, min_full])))
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def update(i: int) -> None:
        ax.clear()
        ax.plot(x[: i + 1], mean_full[: i + 1], color="green", label="mean")
        ax.plot(x[: i + 1], min_full[: i + 1], color="blue", label="min(best-so-far)")
        ax.scatter([i], [mean_full[i]], color="green", s=40)
        ax.scatter([i], [min_full[i]], color="blue", s=40)
        ax.set_title(f"{title}\nFallback mode (fitness-only) - iter {i + 1}/{n_frames}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    _save_animation(fig, update, n_frames=n_frames, out_path=out_path, fps=fps)
    plt.close(fig)


def _render_population_grid(
    pop_history: list[np.ndarray],
    mean_curve: np.ndarray,
    min_curve: np.ndarray,
    out_path: Path,
    fps: int,
    title: str,
    max_dims: int,
) -> None:
    if not pop_history:
        raise ValueError("Population history is empty.")

    normalized: list[np.ndarray] = []
    for raw in pop_history:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        normalized.append(arr)

    n_frames = len(normalized)
    n_particles = normalized[0].shape[0]
    n_dim = normalized[0].shape[1]
    dim_count = min(n_dim, max_dims)

    rows = int(math.ceil(math.sqrt(dim_count)))
    cols = int(math.ceil(dim_count / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    flat_axes = axes.ravel()

    per_dim_min = np.full(dim_count, np.inf)
    per_dim_max = np.full(dim_count, -np.inf)
    for arr in normalized:
        for d in range(dim_count):
            per_dim_min[d] = min(per_dim_min[d], float(np.min(arr[:, d])))
            per_dim_max[d] = max(per_dim_max[d], float(np.max(arr[:, d])))

    for d in range(dim_count):
        if math.isclose(per_dim_min[d], per_dim_max[d]):
            per_dim_min[d] -= 1.0
            per_dim_max[d] += 1.0

    particle_index = np.arange(n_particles)

    def update(i: int) -> None:
        arr = normalized[i]
        mean_txt = f"{mean_curve[i]:.4e}" if i < len(mean_curve) else "n/a"
        min_txt = f"{min_curve[i]:.4e}" if i < len(min_curve) else "n/a"

        for d in range(dim_count):
            ax = flat_axes[d]
            ax.clear()
            ax.scatter(particle_index, arr[:, d], s=10, alpha=0.75)
            ax.set_title(f"dim {d}")
            ax.set_xlabel("particle")
            ax.set_ylabel("value")
            ax.set_ylim(per_dim_min[d], per_dim_max[d])
            ax.grid(alpha=0.2)

        for idx in range(dim_count, len(flat_axes)):
            flat_axes[idx].axis("off")

        fig.suptitle(
            f"{title}\niter {i + 1}/{n_frames} | mean={mean_txt} | min={min_txt}",
            fontsize=12,
        )

    _save_animation(fig, update, n_frames=n_frames, out_path=out_path, fps=fps)
    plt.close(fig)


def _render_tsp_routes(
    route_history: list[np.ndarray],
    city_coords: np.ndarray,
    mean_curve: np.ndarray,
    min_curve: np.ndarray,
    out_path: Path,
    fps: int,
    title: str,
) -> None:
    if not route_history:
        raise ValueError("No TSP routes available for animation.")

    coords = np.asarray(city_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("city_coords must be an array shaped (n_cities, 2).")

    n_frames = len(route_history)
    n_cities = len(coords)

    x_min, x_max = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    y_min, y_max = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(i: int) -> None:
        route = np.asarray(route_history[i]).astype(int).reshape(-1)
        if route.size != n_cities or np.min(route) < 0 or np.max(route) >= n_cities:
            route = np.argsort(route)

        closed_route = np.append(route, route[0])
        route_points = coords[closed_route]

        mean_txt = f"{mean_curve[i]:.4e}" if i < len(mean_curve) else "n/a"
        min_txt = f"{min_curve[i]:.4e}" if i < len(min_curve) else "n/a"

        ax.clear()
        ax.scatter(coords[:, 0], coords[:, 1], s=14, color="black")
        ax.plot(route_points[:, 0], route_points[:, 1], color="tab:blue", linewidth=1.4, alpha=0.9)
        ax.set_title(f"{title}\niter {i + 1}/{n_frames} | mean={mean_txt} | min={min_txt}")
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    _save_animation(fig, update, n_frames=n_frames, out_path=out_path, fps=fps)
    plt.close(fig)


def _render_landscape_2d(
    pop_history: list[np.ndarray],
    func: Callable,
    mean_curve: np.ndarray,
    min_curve: np.ndarray,
    out_path: Path,
    fps: int,
    title: str,
    lb: np.ndarray,
    ub: np.ndarray,
    grid_size: int = 120,
) -> None:
    if not pop_history:
        raise ValueError("Population history is empty.")

    normalized: list[np.ndarray] = []
    for raw in pop_history:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            raise ValueError("Landscape mode requires at least 2 dimensions.")
        normalized.append(arr)

    n_frames = len(normalized)
    x_lb, x_ub = float(lb[0]), float(ub[0])
    y_lb, y_ub = float(lb[1]), float(ub[1])

    x = np.linspace(x_lb, x_ub, grid_size)
    y = np.linspace(y_lb, y_ub, grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    zz_flat = np.array([float(func(np.array(p, dtype=float))) for p in grid_points], dtype=float)
    zz = zz_flat.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 7))

    z_min = float(np.nanmin(zz))
    z_max = float(np.nanmax(zz))
    if math.isclose(z_min, z_max):
        z_min -= 1.0
        z_max += 1.0
    levels = np.linspace(z_min, z_max, 25)

    def update(i: int) -> None:
        arr = normalized[i]
        xy = arr[:, :2]
        particle_vals = np.array([float(func(np.array(p, dtype=float))) for p in xy], dtype=float)

        mean_txt = f"{mean_curve[i]:.4e}" if i < len(mean_curve) else "n/a"
        min_txt = f"{min_curve[i]:.4e}" if i < len(min_curve) else "n/a"

        ax.clear()
        ax.contourf(xx, yy, zz, levels=levels, cmap="turbo", alpha=0.85)
        ax.contour(xx, yy, zz, levels=12, colors="white", linewidths=0.4, alpha=0.55)

        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=particle_vals,
            cmap="turbo",
            edgecolors="black",
            linewidths=0.4,
            s=28,
            zorder=3,
        )

        best_idx = int(np.argmin(particle_vals))
        ax.scatter(
            [xy[best_idx, 0]],
            [xy[best_idx, 1]],
            marker="*",
            s=140,
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        ax.set_title(f"{title}\niter {i + 1}/{n_frames} | mean={mean_txt} | min={min_txt}")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(y_lb, y_ub)
        ax.grid(alpha=0.15)

        # Keep color range stable across frames for visual comparability.
        sc.set_clim(z_min, z_max)

    _save_animation(fig, update, n_frames=n_frames, out_path=out_path, fps=fps)
    plt.close(fig)


def _render_dimension_slices(
    pop_history: list[np.ndarray],
    func: Callable,
    mean_curve: np.ndarray,
    min_curve: np.ndarray,
    out_path: Path,
    fps: int,
    title: str,
    lb: np.ndarray,
    ub: np.ndarray,
    max_dims: int,
    slice_anchor_mode: str,
    curve_points: int = 120,
) -> None:
    if not pop_history:
        raise ValueError("Population history is empty.")

    normalized: list[np.ndarray] = []
    for raw in pop_history:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        normalized.append(arr)

    n_frames = len(normalized)
    n_dim = normalized[0].shape[1]
    dim_count = min(n_dim, max_dims)

    rows = int(math.ceil(math.sqrt(dim_count)))
    cols = int(math.ceil(dim_count / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), squeeze=False)
    flat_axes = axes.ravel()

    slice_x = [np.linspace(float(lb[d]), float(ub[d]), curve_points) for d in range(dim_count)]

    def update(i: int) -> None:
        arr = normalized[i]
        particle_vals = np.array([float(func(np.array(p, dtype=float))) for p in arr], dtype=float)
        if slice_anchor_mode == "zero":
            anchor = np.zeros(n_dim, dtype=float)
        else:
            best_idx = int(np.argmin(particle_vals))
            anchor = arr[best_idx].copy()

        mean_txt = f"{mean_curve[i]:.4e}" if i < len(mean_curve) else "n/a"
        min_txt = f"{min_curve[i]:.4e}" if i < len(min_curve) else "n/a"

        for d in range(dim_count):
            ax = flat_axes[d]
            ax.clear()

            x_axis = slice_x[d]
            samples = np.repeat(anchor.reshape(1, -1), curve_points, axis=0)
            samples[:, d] = x_axis
            y_axis = np.array([float(func(np.array(p, dtype=float))) for p in samples], dtype=float)

            ax.scatter(x_axis, y_axis, c=y_axis, cmap="turbo", s=10, alpha=0.9)
            ax.plot(x_axis, y_axis, color="black", linewidth=0.8, alpha=0.5)

            y_best_idx = int(np.argmin(y_axis))
            ax.scatter(
                [x_axis[y_best_idx]],
                [y_axis[y_best_idx]],
                marker="*",
                s=120,
                c="yellow",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )

            ax.scatter(
                arr[:, d],
                particle_vals,
                c="deepskyblue",
                s=68,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                zorder=3,
            )

            ax.set_title(f"dim {d} slice (others fixed)")
            ax.set_xlabel(f"x{d}")
            ax.set_ylabel("f(x)")
            ax.grid(alpha=0.2)

        for idx in range(dim_count, len(flat_axes)):
            flat_axes[idx].axis("off")

        fig.suptitle(
            f"{title}\niter {i + 1}/{n_frames} | mean={mean_txt} | min={min_txt}",
            fontsize=12,
        )

    _save_animation(fig, update, n_frames=n_frames, out_path=out_path, fps=fps)
    plt.close(fig)


def run_diagnostic_animation(
    config: Any,
    func: Callable,
    n_dim: int,
    sko_to_use: Type[SkoBase],
    out_path: str | Path,
    *,
    config_filter: Callable[[dict], dict] = lambda x: x,
    lb_override: np.ndarray | list[float] | None = None,
    ub_override: np.ndarray | list[float] | None = None,
    fps: int = 10,
    max_dims: int = 16,
    slice_anchor_mode: str = "best",
    city_coords: np.ndarray | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """
    Run one diagnostic optimization and export a GIF.

    This function is intentionally independent from experiment loops so it can be
    used for detailed visual inspection of a single run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sko_model = _build_model(
        config=config,
        func=func,
        n_dim=n_dim,
        sko_to_use=sko_to_use,
        config_filter=config_filter,
        lb_override=np.asarray(lb_override, dtype=float).reshape(-1) if lb_override is not None else None,
        ub_override=np.asarray(ub_override, dtype=float).reshape(-1) if ub_override is not None else None,
    )

    if hasattr(sko_model, "record_mode"):
        setattr(sko_model, "record_mode", True)

    best_x, best_y = sko_model.run()

    mean_curve, min_curve = _extract_fitness_curves(sko_model)
    pop_history = _extract_population_history(sko_model)
    lb, ub = _extract_bounds(sko_model, n_dim=n_dim)

    mode = "population-grid"
    title = f"{sko_to_use.__name__} diagnostic"

    if issubclass(sko_to_use, PSO_TSP):
        route_history = _extract_tsp_route_history(sko_model, pop_history)
        if city_coords is not None and route_history:
            _render_tsp_routes(route_history, np.asarray(city_coords), mean_curve, min_curve, out_path, fps, title)
            mode = "tsp-route"
        else:
            _render_fitness_fallback(mean_curve, min_curve, out_path, fps, title)
            mode = "fitness-fallback"
            if verbose:
                print("[warning] No city_coords or route history. Using fitness-only fallback.")
    elif pop_history:
        sample = np.asarray(pop_history[0])
        if sample.ndim == 2 and sample.shape[1] == 2:
            _render_landscape_2d(
                pop_history=pop_history,
                func=func,
                mean_curve=mean_curve,
                min_curve=min_curve,
                out_path=out_path,
                fps=fps,
                title=title,
                lb=lb,
                ub=ub,
            )
            mode = "landscape-2d"
        elif sample.ndim == 2 and sample.shape[1] > 2:
            _render_dimension_slices(
                pop_history=pop_history,
                func=func,
                mean_curve=mean_curve,
                min_curve=min_curve,
                out_path=out_path,
                fps=fps,
                title=title,
                lb=lb,
                ub=ub,
                max_dims=max_dims,
                slice_anchor_mode=slice_anchor_mode,
            )
            mode = "dimension-slices"
        else:
            _render_population_grid(pop_history, mean_curve, min_curve, out_path, fps, title, max_dims=max_dims)
    else:
        _render_fitness_fallback(mean_curve, min_curve, out_path, fps, title)
        mode = "fitness-fallback"
        if verbose:
            print("[warning] Population history is unavailable. Using fitness-only fallback.")

    if verbose:
        print(f"[saved] {out_path}")

    return {
        "best_x": np.asarray(best_x),
        "best_y": float(np.asarray(best_y).reshape(-1)[0]),
        "out_path": str(out_path),
        "mode": mode,
        "n_frames": int(max(len(pop_history), len(mean_curve), len(min_curve))),
    }
