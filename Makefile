PY := .venv/bin/python -E -s   # -E: ignore PYTHONPATH (ROS on it breaks imports)
.PHONY: download prepare selftest baselines train ablate dreams poseeval steer figures
download: ; bash download.sh
prepare:  ; $(PY) prepare.py --delete-raw
selftest: ; $(PY) selftest.py
baselines: ; $(PY) baselines.py
train:    ; $(PY) train.py
ablate:   ; $(PY) ablate.py --run $(RUN)
dreams:   ; $(PY) -u dreams.py --run $(RUN)
poseeval: ; $(PY) poseeval.py --run pose
steer:    ; for st in real right left stop; do $(PY) -u dreams.py --run pose --steer $$st --out dreams_$$st || exit 1; done
figures:  ; $(PY) curve.py && $(PY) steer_figure.py
# Long run in the background; survives closing the terminal. Stop any time, resume later.
# stop uses TERM: background jobs started from a non-interactive shell ignore SIGINT.
# last.pt is written via tmp+rename, so a kill mid-save cannot corrupt it.
RUN ?= base
ARGS ?=          # e.g. make start RUN=pose ARGS=--pose  (resume restores them from args.json)
start:  ; mkdir -p runs/$(RUN) && setsid nohup $(PY) -u train.py --name $(RUN) $(ARGS) > runs/$(RUN)/train.out 2>&1 & echo started
resume: ; setsid nohup $(PY) -u train.py --name $(RUN) --resume >> runs/$(RUN)/train.out 2>&1 & echo resumed
stop:   ; pkill -TERM -f "^.venv/bin/python .*train.py --name $(RUN)( |$$)" && echo "stopped; progress kept up to the last eval (runs/$(RUN)/last.pt)"
status: ; tail -n 3 runs/$(RUN)/log.jsonl; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
