# Task Evidence NVS

完整中文研究方案与实现状态：[任务语义与场景证据约束技术路线（2026-11-10）](../研究点一_任务语义与场景证据约束技术路线_20261110.md)。包含采样孔洞与未观测区域的处理设计、当前试跑局限和后续验证计划。

`task_evidence_nvs` implements the first executable layer of the task-semantic,
scene-evidence-constrained multi-view augmentation route. It is intentionally
model-neutral: a geometry condition is written to disk and can be passed to a
ViewCrafter, ZeroNVS, or another generator through an adapter. The package does
not claim that a projected point is a correct generated label.

## Pipeline

1. `parse_task` converts a Chinese or English inspection description into a
   controlled vocabulary of categories, attributes, view conditions, and
   relations.
2. `SceneBuilder` joins reference-image observations with an optional labeled
   point cloud. Explicit `instance_id` values are required for high-confidence
   cross-view identity.
3. `build_evidence_condition` projects RGB, depth, support, instance identity,
   and task attention into the target camera. Task attention is gated by
   geometric support and is marked as neither a generation reliability label
   nor a training target.
4. `select_candidates` applies a reliability threshold and separate generation
   and annotation budgets. The CLI uses transparent geometric proxies until
   human-reviewed trial generations and real meta-task updates are available.
5. `review_generated_view` rebuilds supervision from the generated image. It
   never inherits source boxes automatically; unconfirmed identity is recorded
   as unknown or ignored.

## Quick start

The checked-in demo contains two reference records, a small point cloud, and a
relative camera. It is deliberately synthetic and only tests the data flow:

```bash
python tools/build_task_evidence_conditions.py \
  --manifest configs/task_evidence_demo.json \
  --output outputs/task_evidence_demo \
  --max-generation-cost 2 \
  --max-annotation-cost 2
```

The output contains `task.json`, `scene.json`, a selection `manifest.json`, and
one `evidence_condition.npz` plus metadata file per candidate. Conditions use
the arrays `rgb`, `depth`, `support`, `task_attention`, and `instance_map`.

The manifest camera uses OpenCV-style positive camera-Z. Real unknown-pose
scenes must keep the scale relative; the package does not infer physical UAV
altitude from a normalized point cloud.

## Connecting a generator

`generator.GenerationRequest` contains the scene, parsed task, candidate camera,
and condition arrays. Wrap an existing model runner with
`generator.CallableGenerator`, or use `DryRunGenerator` to persist requests
without generating images. After a model produces a view, create proposals from
geometry or a detector, pass independent review confirmations to
`review_generated_view`, and call `rebuild_grounded_text` only on reviewed
objects.

The same operation is available as a command-line step:

```bash
python tools/review_task_evidence.py \
  --task "检测道路附近的车辆和行人" \
  --proposals proposals.json \
  --confirmations confirmations.json \
  --view-id yawp10_pitch0_radiusp0 \
  --output reviewed_view.json
```

`losses.evidence_conditioned_loss` implements the planned objective:

`diffusion + lambda_task * task_evidence + lambda_cross * cross_view`.

The PyTorch loss is optional and does not load a diffusion model. It expects
training-side target views and correspondence masks; it must not be evaluated
against final test labels during candidate selection.

## Limits and research protocol

- The parser is a controlled vocabulary, not a general-purpose language model.
- Projected boxes and attention maps are proposals, not labels.
- `reliability_is_proxy` and `utility_is_proxy` are recorded in selection
  metadata and must be replaced by trial-generation review and scene-disjoint
  real meta-task feedback for formal experiments.
- Generated objects, preserved source instances, reasonable occlusions, and
  identity-unknown regions have separate statuses.
- All generated descendants should inherit the parent scene split; the package
  does not perform dataset leakage detection automatically.
