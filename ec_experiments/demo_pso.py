from scipy import spatial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sko.PSO import PSO, PSO_TSP
from sko.DE import DE
from sko.base import SkoBase
from sko.operators import crossover, mutation, ranking, selection
import csv
import os
from pathlib import Path
from typing import Any, Type, Callable
from tqdm import tqdm
import math
import time





def DistanceRounded(p, q):
  return int(math.hypot(p[0]-q[0], p[1]-q[1])+0.5)

def TourLength(T):
    t = len(T)
    if t == 0:
        return 0.0
    prev = T[t-1]
    length = 0.0
    for i in range(0, t):
        curr = T[i]
        length += DistanceRounded(prev, curr)
        prev = curr
    return length


def TourError(T, best_length):
    return 100.0*(TourLength(T)-best_length)/best_length


def read_tsp(fn):
    f = open(fn+".tsp", "rt")
    reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
    coordinates = False
    cities = []
    best = 0.0
    index = 0
    for row in reader:
        if row[0] == "EOF":
            break
        elif coordinates:
            cities.append((float(row[1]), float(row[2]), index))
            index += 1
        elif row[0] == "BEST:" or row[0] == "OPTIMUM:":
            best = float(row[1])
        elif row[0] == "BEST" or row[0] == "OPTIMUM":
            best = float(row[2])
        elif row[0] == "NAME:":
            print("reading tour", row[1])
        elif row[0] == "NAME":
            print("reading tour", row[2])
        elif row[0] == "NODE_COORD_SECTION":
            coordinates = True
    f.close()
    opt = []
    try:
        f = open(fn+".opt.tour", "rt")
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        coordinates = False
        for row in reader:
            if row[0] == "EOF" or row[0] == "-1":
                break
            elif coordinates:
                index = int(row[0])
                opt.append((cities[index-1][0], cities[index-1][1], index))
            elif row[0] == "NAME:":
                print("reading optimal tour", row[1])
            elif row[0] == "TOUR_SECTION":
                coordinates = True
    except FileNotFoundError:
        print("optimal tour not available")
    finally:
        f.close()

    print("reading done, optimal tour length", best)
    return cities, best, opt





def rosenbrock_creator(a: int = 1, b: int = 100):

    # def rosenbrock(p):
    #     sum = 0
    #     for i in range(len(p)-1):
    #         xi = p[i]
    #         xii = p[i+1]
    #         part = b*(xii-xi**2)**2 + (xi-a)**2
    #         sum += part
    #     return sum

    def rosenbrock(p):
        p = np.asarray(p)
        return np.sum(
            b * (p[1:] - p[:-1]**2)**2 +
            (p[:-1] - a)**2
        )

    return rosenbrock

def griewank_creator(a: int = 4000):

    def griewank(p):
        p = np.asarray(p)
        sum_part = np.sum(p**2) / a
        prod_part = np.prod(np.cos(p / np.sqrt(np.arange(1, len(p)+1))))
        return sum_part - prod_part + 1

    return griewank


class Configuration:
    def __init__(
            self,
            params: dict[str, Any] | None = None,
            name_map: dict[str, str] | None = None,
            **kwargs: Any):
        default_params = {}
        default_name_map = {}

        merged_params = dict(default_params)
        if params is not None:
            merged_params.update(params)
        if kwargs:
            merged_params.update(kwargs)

        self.params = merged_params
        self.name_map = dict(default_name_map)
        if name_map is not None:
            self.name_map.update(name_map)

    def get_filtered_params(self, config_filter: Callable[[dict], dict] = lambda x: x) -> dict:
        """
        Get configuration parameters after applying the filter.

        Args:
            config_filter: Function to filter the configuration dictionary

        Returns:
            Filtered configuration dictionary
        """
        return config_filter(dict(self.params))

    def to_string(
            self,
            config_filter: Callable[[dict], dict] = lambda x: x,
            separator=" ",
            name_map: dict[str, str] | None = None,
            params: dict[str, Any] | None = None) -> str:
        """
        Generate a single-line string representation of filtered configuration.
        Useful for CSV rows and inline prints.

        Args:
            config_filter: Function to filter the configuration dictionary
            separator: String to use between parameters (default: " ")
            name_map: Optional mapping for display names
            params: Optional parameter dictionary to render

        Returns:
            String like "pop=50 mut=0.01 prec=0.01 win=3"
        """
        source_params = dict(self.params) if params is None else dict(params)
        filtered_params = config_filter(source_params)
        effective_name_map = self.name_map if name_map is None else name_map
        parts = []

        for key, value in filtered_params.items():
            display_name = effective_name_map.get(key, key)
            parts.append(f"{display_name}={value}")

        return separator.join(parts)

    def to_label(
            self,
            config_filter: Callable[[dict], dict] = lambda x: x,
            name_map: dict[str, str] | None = None,
            params: dict[str, Any] | None = None) -> str:
        """
        Generate a multi-line string representation for plot labels.

        Args:
            config_filter: Function to filter the configuration dictionary
            name_map: Optional mapping for display names
            params: Optional parameter dictionary to render

        Returns:
            String like "pop=50\\nmut=0.01\\nprec=0.01\\nwin=3"
        """
        return self.to_string(config_filter, separator="\n", name_map=name_map, params=params)
    
    def add_config_and_name_maps(self, new_params: dict[str, Any], new_name_map: dict[str, str]) -> "Configuration":
        """
        Add new parameters and name mappings to a copy of the current configuration.

        Args:
            new_params: Dictionary of new parameters to add
            new_name_map: Dictionary of new name mappings to add
        Returns:
            New configuration object with updated parameters and name mappings
        """
        updated_params = dict(self.params)
        updated_params.update(new_params)

        updated_name_map = dict(self.name_map)
        updated_name_map.update(new_name_map)

        return Configuration(params=updated_params, name_map=updated_name_map)
        


def trainer(
    config: Configuration, runs: int = 5,
    func=rosenbrock_creator(), n_dim=2, a_val=1,
    Sko_to_use: Type[SkoBase] = PSO,
    config_filter: Callable[[dict], dict] = lambda x: x,
    lb=[-2], ub=[2],
    verbose: int = 2,
    track_performance: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float, float], dict[str, Any]]:
    """
    For a configuration, return a mean of the histories of several SKO runs

    Args:
        verbose: 0 = no output, 1 = only tqdm progress bars, 2 = tqdm + print statements
        track_performance: If True, track execution time metrics
    """
    all_history = []
    all_mins = []
    all_means = []
    all_best_x = []
    all_best_y = []
    all_run_times = []
    config_start_time = time.time()
    runs_iter = tqdm(range(runs), desc="  Runs", leave=False,
                     disable=verbose == 0, position=1) if verbose >= 1 else range(runs)
    for _ in runs_iter:
        if issubclass(Sko_to_use, PSO_TSP):
            sko_model = Sko_to_use(
                func=func,
                n_dim=n_dim,
                **config.get_filtered_params(config_filter)
            )
        else:
            sko_model = Sko_to_use(
                func=func,
                n_dim=n_dim,
                lb=lb,
                ub=ub,
                **config.get_filtered_params(config_filter)
            )

        run_start_time = time.time()
        best_x, best_y = sko_model.run()
        run_end_time = time.time()
        if track_performance:
            all_run_times.append(run_end_time - run_start_time)
        all_best_x.append(np.asarray(best_x))
        all_best_y.append(float(np.asarray(best_y).reshape(-1)[0]))
        if hasattr(sko_model, "all_history_Y"):
            history = sko_model.all_history_Y
        elif hasattr(sko_model, "gbest_y_hist"):
            history = sko_model.gbest_y_hist
        else:
            history = []
        means = np.mean(history, axis=1)
        mins = np.minimum.accumulate(np.min(history, axis=1))
        all_history.append(history)
        all_mins.append(mins)
        all_means.append(means)

    history = np.mean(all_history, axis=0)
    mins = np.mean(all_mins, axis=0)
    min_std = np.std(all_mins, axis=0)
    means = np.mean(all_means, axis=0)
    mean_std = np.std(all_means, axis=0)
    df = pd.DataFrame({
        'min': mins,
        'mean': means
    })
    df_history = pd.DataFrame(history)
    best_x_mean = np.mean(all_best_x, axis=0)
    best_x_std = np.std(all_best_x, axis=0)
    best_y_mean = np.mean(all_best_y)
    best_y_std = np.std(all_best_y)
    
    # Calculate performance metrics if requested
    performance_metrics = {}
    if track_performance:
        config_end_time = time.time()
        total_config_time = config_end_time - config_start_time
        avg_run_time = np.mean(all_run_times) if all_run_times else 0.0
        
        performance_metrics = {
            'total_time': total_config_time,
            'avg_run_time': avg_run_time,
            'min_run_time': np.min(all_run_times) if all_run_times else 0.0,
            'max_run_time': np.max(all_run_times) if all_run_times else 0.0,
            'runs': runs
        }
    
    return df, df_history, (min_std, mean_std), (best_x_mean, best_x_std, best_y_mean, best_y_std), performance_metrics


def train_multiple_configs(
        config_sets: dict[str, list[Configuration]],
        runs: int = 5, folder: os.PathLike[str] = "",
        func=rosenbrock_creator(), n_dim=2, a_val=1,
        lb=[-2], ub=[2],
        Sko_to_use: Type[SkoBase] = PSO,
        config_filter: Callable[[Configuration], dict] = lambda x: x,
    log_scale: bool = False,
        verbose: int = 2,
        track_performance: bool = False) -> None:
    """
    For configuration sets, create consolidated plots with 2 rows (min/mean)
    and as many columns as configs in each set

    Args:
        log_scale: If True, use logarithmic scale on Y axis for min/mean plots.
        verbose: 0 = no output, 1 = only tqdm progress bars, 2 = tqdm + print statements
        track_performance: If True, save performance metrics to a separate CSV file
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    # Open files for writing and keep them open throughout the loop
    summary_file = open(folder / f"summary.csv", mode="w")
    writer = csv.writer(summary_file, delimiter=",")
    writer.writerow(["Config Name", "Best Y Mean", "Best Y std"])
    
    # Initialize performance CSV if tracking
    perf_file = None
    perf_writer = None
    if track_performance:
        perf_file = open(folder / f"performance.csv", mode="w")
        perf_writer = csv.writer(perf_file, delimiter=",")
        perf_writer.writerow(["Config Name", "Total Time (s)", "Avg Run Time (s)", "Min Run Time (s)", "Max Run Time (s)", "Runs"])

    try:
        # Calculate total configs for overall progress
        total_configs = sum(len(configs) for configs in config_sets.values())
        config_counter = 0

        set_iter = tqdm(config_sets.items(), desc="Config Sets", unit="set",
                        leave=True, position=0) if verbose >= 1 else config_sets.items()
        for set_name, configs in set_iter:
            # Train all configurations in the set
            if verbose >= 1:
                set_iter.set_postfix_str(f"Current: {set_name[:40]}")
            if verbose == 2:
                print(f"\n[Set] {set_name} ({len(configs)} configs)")
            results = []
            config_desc = f"{set_name[:30]:30s}" if len(
                set_name) > 30 else f"{set_name:30s}"
            config_progress = tqdm(configs, desc=config_desc, leave=False,
                                   position=0, unit="config") if verbose >= 1 else configs
            for config in config_progress:
                config_counter += 1
                if verbose == 2:
                    config_str = config.to_string(
                        config_filter, separator=", ")
                    print(f"  [Config] {config_str}")
                df, df_history, stds, best_stats, perf_metrics = trainer(
                    config, runs, func, n_dim, a_val, Sko_to_use, 
                    config_filter=config_filter, verbose=verbose,
                    lb=lb, ub=ub,
                    track_performance=track_performance)
                best_x_mean, best_x_std, best_y_mean, best_y_std = best_stats
                best_x_mean_str = np.array2string(
                    best_x_mean, precision=4, separator=', ')
                best_x_std_str = np.array2string(
                    best_x_std, precision=4, separator=', ')
                if verbose == 2:
                    print(f"    best_x mean: {best_x_mean_str}")
                    print(f"    best_x std : {best_x_std_str}")
                    print(f"    best_y mean: {best_y_mean:.6e}")
                    print(f"    best_y std : {best_y_std:.6e}")
                if verbose >= 1:
                    config_progress.set_postfix_str(
                        f"best_y={best_y_mean:.4e}±{best_y_std:.4e}")
                config_label = config.to_label(config_filter)
                config_string = config.to_string(config_filter)
                writer.writerow([
                    config_string,
                    f"{best_y_mean:.6e}",
                    f"{best_y_std:.6e}"
                ])
                summary_file.flush()
                
                # Write performance metrics if tracking
                if track_performance and perf_metrics and perf_writer is not None:
                    perf_writer.writerow([
                        config_string,
                        f"{perf_metrics['total_time']:.6f}",
                        f"{perf_metrics['avg_run_time']:.6f}",
                        f"{perf_metrics['min_run_time']:.6f}",
                        f"{perf_metrics['max_run_time']:.6f}",
                        perf_metrics['runs']
                    ])
                    perf_file.flush()
                results.append((df, stds, config_label))

            # Create consolidated plot: 2 rows x len(configs) columns
            n_configs = len(configs)
            fig, axes = plt.subplots(2, n_configs, figsize=(5*n_configs, 8))

            # Handle case where n_configs == 1
            if n_configs == 1:
                axes = axes.reshape(2, 1)

            fig.suptitle(set_name, fontsize=16, fontweight='bold')

            for col, (df, stds, config_label) in enumerate(results):
                min_std, mean_std = stds

                # Row 0: Minimum fitness
                ax_min = axes[0, col]
                df["min"].plot(
                    ax=ax_min, label='Mean Min Fitness', color='blue')
                ax_min.fill_between(
                    df.index,
                    df["min"] - min_std,
                    df["min"] + min_std,
                    color='blue', alpha=0.2, label='Std Dev'
                )
                ax_min.set_title(config_label, fontsize=10)
                ax_min.set_xlabel('Iteration')
                ax_min.set_ylabel('Min Fitness')
                if log_scale:
                    ax_min.set_yscale('log')
                ax_min.legend(fontsize=8)
                ax_min.grid(True, alpha=0.3)

                # Annotation for last value
                last_idx = len(df["min"]) - 1
                last_val = df["min"].iloc[last_idx]
                last_std_val = min_std[last_idx]
                ax_min.annotate(
                    f'{last_val:.4e}±{last_std_val:.4e}',
                    xy=(last_idx, last_val), xycoords='data',
                    xytext=(-40, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='black'),
                    bbox=dict(boxstyle="round", fc='white', alpha=0.8),
                    fontsize=8
                )

                # Row 1: Mean fitness
                ax_mean = axes[1, col]
                df["mean"].plot(
                    ax=ax_mean, label='Mean Avg Fitness', color='green')
                ax_mean.fill_between(
                    df.index,
                    df["mean"] - mean_std,
                    df["mean"] + mean_std,
                    color='green', alpha=0.2, label='Std Dev'
                )
                ax_mean.set_xlabel('Iteration')
                ax_mean.set_ylabel('Mean Fitness')
                if log_scale:
                    ax_mean.set_yscale('log')
                ax_mean.legend(fontsize=8)
                ax_mean.grid(True, alpha=0.3)

                # Annotation for last value
                last_val = df["mean"].iloc[last_idx]
                last_std_val = mean_std[last_idx]
                ax_mean.annotate(
                    f'{last_val:.4e}±{last_std_val:.4e}',
                    xy=(last_idx, last_val), xycoords='data',
                    xytext=(-40, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='black'),
                    bbox=dict(boxstyle="round", fc='white', alpha=0.8),
                    fontsize=8
                )

            plt.tight_layout()
            # Save with sanitized set name
            safe_name = set_name.replace(" ", "_").replace("/", "_")
            plt.savefig(folder / f'{safe_name}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            if verbose == 2:
                print(f"  [Saved] {safe_name}.png")
    finally:
        # Close all open files
        summary_file.close()
        if perf_file is not None:
            perf_file.close()


def run_single_visual_diagnostic(
    config: Configuration,
    func: Callable,
    n_dim: int,
    Sko_to_use: Type[SkoBase] = PSO,
    out_path: str | os.PathLike[str] = "diagnostics/diagnostic.gif",
    config_filter: Callable[[dict], dict] = lambda x: x,
    fps: int = 10,
    max_dims: int = 16,
    city_coords: np.ndarray | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """
    Run a single optimization and export a diagnostic GIF.

    The function is opt-in and externalized so existing experiment loops remain unchanged.
    """
    from ec_experiments.visualization import run_diagnostic_animation

    return run_diagnostic_animation(
        config=config,
        func=func,
        n_dim=n_dim,
        sko_to_use=Sko_to_use,
        out_path=Path(out_path),
        config_filter=config_filter,
        fps=fps,
        max_dims=max_dims,
        city_coords=city_coords,
        verbose=verbose,
    )



