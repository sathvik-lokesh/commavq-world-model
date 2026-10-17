# Experiment design

The data, baselines and tests for this project, written before the results they apply to. Results are in [README.md](README.md).

## Data
- commaVQ (`commaai/commavq`, MIT): 40 shards x 2,500 one-minute segments, 20 fps, each frame =
  8x16 VQ tokens (vocab 1024). Also has a per-frame 6-d `pose` (used in M4).
- Train = data-0000..0002 (7,500 segments, 125 h). Test = data-0039 (whole shard held out; the
  scores use its first 400 segments). The dataset ships only a "train" split, so the split is ours.

## Metric
Mean cross-entropy (nats/token) of frame t given frames < t, scored only on frames with >=1
frame of context. The baselines (`runs/baselines.json`) and the model are scored on the same 400 test segments.

| Baseline (held-out) | nats/token |
|---|---|
| uniform | 6.93 |
| unigram | 6.85 |
| copy-mix (40% "same token as last frame") | 4.78 |
| **prev-token** P(tok \| same position, last frame) | **3.30** <- bar to beat |
| prev-token+pos | 3.31 |
Test repeat rate: 39.8% of tokens are identical to the previous frame at the same position.

## Milestones
1. **M1 base model**: 8-layer, d512, 16-frame context. Beat 3.30 clearly. Report the
   learning curve and wall-clock time.
2. **M2 does context help?** Score the same model with 1, 2, 4, 8 and 15 frames of context. If
   the gain stops at 1-2 frames, it is a smart copy machine, not a world model. Say so.
3. **M3 dreams**: `render.py` real-vs-imagined videos. Clips are fixed by seed before generating, plus a "worst" one.
4. **M4 action conditioning (stretch)**: feed `pose` as a per-frame token/embedding. Same
   start, different steering -> different dreams? Tests whether the model is action-conditioned. Check what the 6 pose values mean first (unverified).

## M4 tests and pose model
- **M4 pre-registered prediction**: with the real pose fed in, dream-vs-real pixel MAE should drop
  BELOW frozen at short horizons on the same 16 clips (`dreams.py` rerun). If it still ties, the
  model ignores the pose.
- **M4 tests, fixed before results:** (1) pose-shuf worse than pose-real (CI excludes 0) = uses pose;
  (2) `dreams_real` beats frozen pixel MAE at short horizons on the same 16 clips (M3 prediction);
  (3) `dreams_right` turn_px > 0 > `dreams_left` turn_px (yaw +-0.2 rad/s, other dims real);
  (4) `dreams_stop` motion rate < `dreams_real` motion rate.
- **M4 step 1 DONE: pose verified from the data** (undocumented on HF card + GitHub README):
  [vx, vy, vz] m/s, [roll, pitch, yaw] rate rad/s, openpilot car frame (x fwd, y right, z down).
  vx: 0-36 m/s (p1-p99), 5% stopped, lag-1 autocorr 1.000; yaw p1-p99 +-0.22, autocorr 0.999;
  roll/pitch noisy (0.86). pose[1199] always NaN -> pose[t] = motion t->t+1. **Yaw > 0 = right turn**:
  40 strong-yaw frame pairs, image shift sign matches 65%/5%, shift sign calibrated on a synthetic
  roll. Caveat: poses were estimated from the video, so they carry info about the target frame.
- **M4 model**: per-frame Linear(6->512) of normalised pose[t-1], added to every token of frame t
  (frame 0 none). Built after init + zero-init -> same seed gives exactly the base model at step 0
  (selftest 19/19; step-0 test NLL identical to base, 7.047275513410568). `make start RUN=pose ARGS=--pose`.
  Resume restores flags from args.json; flags typed on the command line still win.

## Known traps
- 20 fps: adjacent frames are near-identical, so low loss can be pure copying. M2 exists for that.
- Do not compare against comma's leaderboard numbers unless the scoring (context length, which
  frames count) is identical.
- `-E -s` on every python call: ROS on PYTHONPATH breaks imports (see Makefile).
