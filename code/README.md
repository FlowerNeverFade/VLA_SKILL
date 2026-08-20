# Single-Skill PI0.5 LoRA

This directory is the standalone single-skill implementation. It converts
LeRobot task slices into validated skill directories, computes per-skill
statistics, inserts a selected LoRA target group into PI0.5, and writes adapter
and processor artifacts for evaluation or runtime inference.

Run commands from this directory or pass explicit `--skill-root`,
`--base-model-path`, and `--output-root` values. See the repository README for
the external runtime layout and environment variables.
