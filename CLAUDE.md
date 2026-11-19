# HD Map Change Detection Agent — Portfolio Project

## Goal
Portfolio project for Physical / Spatial AI and AV roles (Waymo, Rivian, NVIDIA, Deloitte Physical & Spatial AI).
Answer a natural-language question like "Is the crosswalk at this intersection still where the map says?"
by comparing vehicle LiDAR + camera data against the HD map, and report **deviation + confidence**.

Pipeline: Narrative (NL question) → agent → LiDAR + camera vs HD map → geometry → error + confidence.

Bita's background: transportation planning + telemetry. Lean on that intuition.

## Working style
- Bita is learning the fundamentals as she builds: explain each transformation briefly, prefer examples over long text.
- Keep responses concise. Build small, working increments; don't over-engineer.
- Evaluation from day one; no fine-tuning until a baseline shows a measurable failure.

## Data
- **Argoverse 2 Map Change / Trust but Verify (TbV)**: ~1,043 vehicle logs (~54 s each), LiDAR sweeps,
  7 ring cameras, poses, calibration, HD vector maps; some logs contain real-world map changes
  (labeled match / mismatch). Tabular data in .feather files.
  - Guide: https://argoverse.github.io/user-guide/datasets/map_change_detection.html
  - API: https://github.com/argoverse/av2-api (`pip install av2`)
  - Download one shard first: https://s3.amazonaws.com/argoverse/datasets/av2/tars/tbv/TbV_v1.0_shard0.tar.gz
  - Reference code: https://github.com/johnwlambert/tbv
- Later: Argoverse 2 Sensor dataset (same map format, 3D annotations) or nuScenes.

## MVP 1 — "Does the map still match the road?" (no training)
1. Load one log: LiDAR sweeps, ego poses, calibration, HD map (lane boundaries, crosswalks) → av2 API
2. Transform LiDAR into the city/map frame using ego poses → coordinate transforms (SE(3))
3. Extract ground points + high-intensity points (lane paint) → Open3D / NumPy
4. Per map element: distance between map polylines/polygons and observed paint points → deviation
5. Confidence: point density/coverage per element, occlusion, pose uncertainty, intensity noise
6. Agent: NL question → tool calls (load log, query map element, compute deviation) → short report → LangGraph + LLM

Eval:
- Real changes: use TbV match/mismatch labels → precision / recall of change detection
- Synthetic: shift map elements by known amounts (e.g., 10 / 25 / 50 cm) → check recovered error + confidence calibration

## Later phases (post-training, only where baseline fails)
1. Confidence calibration (temperature scaling / isotonic) — cheap, stats-heavy
2. Fine-tune pretrained segmentation model for lane markings / crosswalks (camera or bird's-eye-view LiDAR)
3. Optional: LoRA fine-tune small LLM/VLM for tool use or change-type labeling (moved / removed / added / occluded)
4. Optional: temporal memory across multiple drives of the same area

Parked for now: YOLO, NVIDIA Cosmos, Vertex AI.

## Stack
Python, NumPy, pandas/pyarrow (.feather), av2, Open3D, PyTorch (later), LangGraph, MLflow (experiment tracking).
Mac (no MS Office; use Numbers/CSV for tables).
