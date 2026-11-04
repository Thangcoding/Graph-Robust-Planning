#!/usr/bin/env bash 
# Pipeline: for every env x algorithm x simulation budget,
#   1. tune.py  -> experiments/tune_config/<algo>/<env>/sims<N>/   (top configs by root regret)
#   2. run.py   -> experiments/results/<algo>/<env>/sims<N>/       (best config: plan on theta_plan, execute on theta_exec)
#    
#   bash experiments/run_pipeline.sh                                    # every env, algorithm, budget below
#   ENVS="gambler" ALGOS="uct robust-uct" SIMS="100 400" bash experiments/run_pipeline.sh
#   SKIP_TUNE=1 ALGOS="uct" bash experiments/run_pipeline.sh           # only run.py, reuse tune_config/
#   ALGOS="gs-robust-uct" EPS="0.05 0.1" bash experiments/run_pipeline.sh  # one tune+run per fixed eps (sims<N>/eps<e>/)
#   
# Logs: experiments/logs/<env>_<algo>_sims<N>[_eps<e>]_{tune,run}.log
set -euo pipefail
cd "$(dirname "$0")/.."

ENVS="${ENVS:-gambler frozen_lake}"
ALGOS="${ALGOS:-uct stochastic-power-uct gs-power-uct gsf-power-uct robust-uct gs-robust-uct gsf-robust-uct}"
SIMS="${SIMS:-100}"
EPS="${EPS:-none}"   # "none" = eps tuned with the other hyper-parameters
WORKERS="${WORKERS:-$(nproc)}"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"

for ENV in $ENVS; do
  for ALGO in $ALGOS; do
    for B in $SIMS; do
     for E in $EPS; do
      NAME="${ENV}_${ALGO}_sims${B}"
      ARGS=(--env "$ENV" --algo "$ALGO" --sims "$B" --workers "$WORKERS")
      if [ "$E" != "none" ]; then
        NAME="${NAME}_eps${E}"
        ARGS+=(--eps "$E")
      fi

      if [ -z "${SKIP_TUNE:-}" ]; then
        echo "================ tune | $ENV | $ALGO | sims $B | eps $E ================"
        python experiments/tune.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/${NAME}_tune.log"
      fi

      echo "================ run  | $ENV | $ALGO | sims $B | eps $E ================"
      python experiments/run.py "${ARGS[@]}" 2>&1 | tee "$LOG_DIR/${NAME}_run.log"
     done
    done 
  done 
done
echo "pipeline done: envs [$ENVS], algos [$ALGOS], sims [$SIMS], eps [$EPS]"




