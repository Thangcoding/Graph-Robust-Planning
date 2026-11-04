# RobustMCGS

Robust online planning under model mismatch. The project combines Monte Carlo Graph Search, which merges nodes that share a state, with a robust Bellman backup over a total-variation (TV) uncertainty set.

**Setting:** the planner plans on a model `θ_plan`, but its actions are executed in the real environment `θ_exec ≠ θ_plan`. Performance is measured by regret against Q*(θ_exec) and by success rate.

## Installation


###  Create the environment and install the requirements

From the repository root:

```bash
conda create -n robustGraph python=3.12 -y
conda activate robustGraph
pip install -r requirements.txt     # numpy, gymnasium, matplotlib, pandas, ipykernel, pytest
```

Check the installation:

```bash
python -c "import src.algorithms, src.envs; print('ok')"
```


```bash
python -m ipykernel install --user --name robustGraph --display-name "Python (robustGraph)"
```

Then select the kernel "Python (robustGraph)" in Jupyter or VS Code. Run `conda activate robustGraph` in every new terminal before running the scripts.

## Layout

```
src/
  algorithms/
    base/           BasePlanner (MCTS loop: select, expand, rollout, backup), Node, Action, config
    planner/        planners, registered with @register_planner
    registry.py     make_planner(name, env, **config)
  envs/             gambler, frozen_lake (8x8), cliff_walking, american_option; make_env(name, **kw)
  utils/            visualization
experiments/        tune.py, run.py, value_iteration.py, result_analysis.ipynb (see experiments/README.md)
scripts/            run_pipeline.sh: tune then run for every env × algo × N × ε
tests/              pytest
```

## Planners

| Name (`make_planner`) | Search structure | Backup |
|---|---|---|
| `uct` | tree | mean (UCT) |
| `stochastic-power-uct` | tree | power mean |
| `gs-power-uct` | graph, merges states at the same depth | power mean |
| `gsf-power-uct` | graph, merges all identical states | power mean |
| `robust-uct` | tree | robust (TV ball of radius ε) + power mean |
| `gs-robust-uct` | graph, same depth | robust |
| `gsf-robust-uct` | graph, full merge | robust |

Main hyper-parameters:
- `num_simulations` (N)
- `exploration_const` (c)
- `p`: exponent of the power mean (`inf` is equivalent to taking the max)
- `uncertainty_budget`: radius ε of the TV ball
- `gamma`, `max_depth`, `rollout_depth`, `seed`

## Example

```python
from src.envs import make_env
from src.algorithms import make_planner

env = make_env("frozen_lake", slip_rate=0.2)
env.reset(seed=0)
planner = make_planner("gsf-robust-uct", env, num_simulations=200,
                       uncertainty_budget=0.2, p=float("inf"), seed=0)
action = planner.plan(env)
```

To add a new planner, subclass `BasePlanner`, override `select_action`, `expand` and `backup`, and decorate the class with `@register_planner("new-name")`.

## Experiments

```bash
# tune + run; results are written to experiments/results/<algo>/<env>/sims<N>[/eps<e>]/
ENVS=frozen_lake ALGOS=gsf-robust-uct SIMS="100 1000" EPS="0.2" bash scripts/run_pipeline.sh
```

Then open `experiments/result_analysis.ipynb` to analyse the results. Details on the parameter grids, the result file columns and caveats are in [`experiments/README.md`](experiments/README.md).

## Main results (11/2026)

- **FrozenLake:** GSF Robust UCT is the best method at every N, in both regret and success rate. The robust backup only helps once states are merged: on a plain tree it is worse than the non-robust backup.
- **Gambler:** GS Robust UCT is the most stable method. GSF Robust UCT gets worse when `θ_plan = 0.4` and the real game is more favourable than the model.

## Tests

```bash
pytest tests/
```
