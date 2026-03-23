# scikit-opt-experiments

Minimal framework to run reproducible optimization experiments with scikit-opt using TOML plans.

Package name: `ec_experiments`

CLI commands:

1. `run-experiment`
2. `run-visual-diagnostic`

## Quick Start

1. Install and sync dependencies:

```bash
uv sync
```

2. Run a plan (dry run):

```bash
uv run run-experiment --config config/experiments/pso_coefficient_analysis.toml --dry-run
```

3. Run the same plan for real:

```bash
uv run run-experiment --config config/experiments/pso_coefficient_analysis.toml
```

## Project Layout

- `config/experiments/`: experiment plans in TOML
- `ec_experiments/`: source package
- `ec_experiments/launcher/`: config loading + algorithm/objective registries + restrictions
- `ec_experiments/visualization/`: diagnostic GIF rendering
- `data/`: datasets (for example berlin52)
- `outputs/`: experiment outputs
- `diagnostics/`: generated diagnostic GIFs

## Running Experiments

Main command:

```bash
uv run run-experiment --config <plan.toml>
```

Useful overrides:

```bash
uv run run-experiment \
  --config config/experiments/pso_coefficient_analysis.toml \
  --runs 20 \
  --verbose 1
```

Available plans:

1. `config/experiments/de_F_prob_mut.toml`
2. `config/experiments/ga_best_configs.toml`
3. `config/experiments/pso_anton.toml`
4. `config/experiments/pso_coefficient_analysis.toml`
5. `config/experiments/pso_complete_tsp_analysis.toml`
6. `config/experiments/pso_de_initial_analysis.toml`
7. `config/experiments/pso_tsp_analysis.toml`
8. `config/experiments/pso_tsp_improved_analysis.toml`
9. `config/experiments/rosenbrock_comparison_de_pso_ga.toml`

## Diagnostic Visualizer

Main command:

```bash
uv run run-visual-diagnostic --help
```

Mode A: registry mode (no TOML required)

```bash
uv run run-visual-diagnostic \
  --algorithm-id pso \
  --objective-id rosenbrock \
  --n-dim 2 \
  --out diagnostics/diag_registry.gif
```

Mode B: plan mode (from TOML)

```bash
uv run run-visual-diagnostic \
  --config config/experiments/pso_coefficient_analysis.toml \
  --task-index 0 \
  --set-name "Medium w tests" \
  --variant-index 0 \
  --out diagnostics/diag_plan.gif
```

Common flags:

```bash
--fps 10
--max-dims 16
--slice-anchor-mode zero|best
--de-historyx
```

## TOML Essentials

Each plan contains:

1. `[experiment]`: global settings (runs, output_dir, verbosity)
2. `[[tasks]]`: objective + algorithm pair to run
3. `[algorithms.<id>]`: base params, name map, config sets, restrictions

Restrictions are declarative expressions, for example:

```toml
restrictions = [
  "c1 + c2 <= 4",
  "0.5 * (c1 + c2) - 1 <= w <= 1",
]
on_restriction_violation = "warn"
```

Allowed violation policies:

1. `warn`
2. `skip`
3. `error`

## Development Notes

How to add a custom algorithm:

1. Implement class in `ec_experiments/custom_algorithms.py` (or another module)
2. Register it in `ec_experiments/launcher/algorithms.py`
3. Reference `algorithm_id` in TOML

How to add a custom objective:

1. Add builder in `ec_experiments/launcher/objectives.py`
2. Register objective id
3. Use it from `[[tasks]]` in TOML

## Troubleshooting

- Command not found (`run-experiment` / `run-visual-diagnostic`): run `uv sync`
- Using old module path (`tasks.*`): use `ec_experiments.*`
- Validate a plan without executing: add `--dry-run`
